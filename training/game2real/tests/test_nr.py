"""Independent mathematical and graph checks against the repository NR sources.

These tests check the trainable floating-point reconstruction. They do not
establish bit parity with the FP8/FP16 Vulkan instruction schedule. The window
oracle below explicitly enumerates native zero-border windows; it does not use
the implementation's partition, padding, or attention helpers.
"""

from __future__ import annotations

from collections import Counter
import math

import pytest
import torch
from torch import nn

from gta2real.nr import (
    NRBlock, NRConfig, NRFFN, NRGenerator, NRGeometry,
    NRGlobalAttention, NRTransition, NRWindowAttention, mp_cubic_silu, pool2x_align4,
)


PHASE_SHIFTS = ((0, 0), (4, 4), (4, 0), (0, 4))
WIDTHS_BY_NUMBER = {
    **{index: 32 for index in (0, 1, 2, 3, 4, 66, 67, 68, 69, 70)},
    **{index: 64 for index in (*range(5, 9), *range(62, 66))},
    **{index: 128 for index in (*range(9, 15), *range(56, 62))},
    **{index: 256 for index in (*range(15, 23), *range(48, 56))},
    **{index: 512 for index in (*range(23, 31), *range(40, 48))},
    **{index: 1024 for index in range(31, 39)},
}


def _identity_attention(module, *, query_key: bool = False) -> None:
    """Set head-major QKV matrices without relying on the forward reshape."""
    channels = module.qkv.in_features
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
        module.attention_scale.fill_(1)
        module.projection.weight.copy_(torch.eye(channels))
        for channel in range(channels):
            head, within_head = divmod(channel, 32)
            module.qkv.weight[head * 96 + 64 + within_head, channel] = 1
            if query_key:
                module.qkv.weight[head * 96 + within_head, channel] = 1
                module.qkv.weight[head * 96 + 32 + within_head, channel] = 1


def _uniform_zero_border_oracle(image: torch.Tensor, phase: int) -> torch.Tensor:
    result = torch.empty_like(image)
    _, height, width, _ = image.shape
    shift_x, shift_y = PHASE_SHIFTS[phase]
    for y in range(height):
        origin_y = ((y + shift_y) // 8) * 8 - shift_y
        for x in range(width):
            origin_x = ((x + shift_x) // 8) * 8 - shift_x
            patch = image[:, max(0, origin_y):min(height, origin_y + 8),
                          max(0, origin_x):min(width, origin_x + 8)]
            # Even an edge window has 64 keys. Out-of-field K/V are zero, but
            # their exp(0) still contributes to the attention denominator.
            result[:, y, x] = patch.sum((1, 2)) / 64
    return result


@pytest.mark.parametrize("phase", range(4))
def test_all_four_window_phases_use_zero_border_not_wrap_or_key_mask(phase):
    attention = NRWindowAttention(64)
    _identity_attention(attention)
    spatial = torch.arange(9 * 11, dtype=torch.float32).reshape(1, 9, 11, 1) / 100
    image = spatial + torch.arange(64, dtype=torch.float32).reshape(1, 1, 1, 64) / 1000
    actual = attention(image, phase=phase)
    expected = _uniform_zero_border_oracle(image, phase)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)


def test_window_cosine_attention_uses_learned_scale_without_sqrt_head_divisor():
    attention = NRWindowAttention(32)
    _identity_attention(attention, query_key=True)
    with torch.no_grad():
        attention.attention_scale.fill_(2)
    image = torch.zeros(1, 1, 2, 32)
    image[0, 0, 0, 0] = 1
    image[0, 0, 1, 1] = 1
    actual = attention(image, phase=0)
    denominator = math.exp(2) + 63  # One self key and 63 other/zero-border keys.
    assert actual[0, 0, 0, 0].item() == pytest.approx(math.exp(2) / denominator, rel=1e-5)
    assert actual[0, 0, 0, 1].item() == pytest.approx(1 / denominator, rel=1e-5)
    assert actual[0, 0, 1, 0].item() == pytest.approx(1 / denominator, rel=1e-5)


def test_prior_is_full_query_key_matrix_not_relative_offset_table():
    attention = NRWindowAttention(32)
    _identity_attention(attention)
    assert tuple(attention.prior.shape) == (1, 64, 64)
    with torch.no_grad():
        attention.prior[0, 0, 0] = math.log(3)
    image = torch.ones(1, 8, 8, 32)
    image[0, 0, 0, 0] = 2
    actual = attention(image, phase=0)
    assert actual[0, 0, 0, 0].item() == pytest.approx(69 / 66, rel=2e-6)
    assert actual[0, 0, 1, 0].item() == pytest.approx(65 / 64, rel=2e-6)


def test_global_attention_keeps_real_tokens_and_has_sqrt32_q_scale():
    attention = NRGlobalAttention(1024)
    _identity_attention(attention, query_key=True)
    with torch.no_grad():
        attention.attention_scale.fill_(0.2)
    image = torch.zeros(1, 1, 2, 1024)
    image[0, 0, 0, 0] = 1
    image[0, 0, 1, 1] = 1
    actual = attention(image)
    self_score = math.exp(math.sqrt(32) * 0.2)
    assert actual[0, 0, 0, 0].item() == pytest.approx(self_score / (self_score + 1), rel=1e-5)
    assert actual[0, 0, 0, 1].item() == pytest.approx(1 / (self_score + 1), rel=1e-5)
    assert not any("prior" in name for name, _ in attention.named_parameters())


def test_native_cubic_silu_coefficients_and_unclipped_final_multiply():
    values = torch.tensor([-10.0, -4.0, 0.0, 1.0, 4.0, 10.0], requires_grad=True)
    actual = mp_cubic_silu(values)
    expected = torch.tensor([-0.0, -0.0, 0.0, 1.285888671875, 7.15625, 17.890625])
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    actual.sum().backward()
    assert torch.isfinite(values.grad).all()
    assert values.grad[-1].item() == pytest.approx(1.7890625)


@pytest.mark.parametrize("channels,expected_parameters", [
    (32, 8192), (64, 28672), (128, 98304), (256, 360448),
    (512, 786432), (1024, 8388608),
])
def test_ffn_exact_parameter_topology_and_gradient(channels, expected_parameters):
    ffn = NRFFN(channels)
    assert sum(parameter.numel() for parameter in ffn.parameters()) == expected_parameters
    assert not any(name.endswith("bias") for name, _ in ffn.named_parameters())
    image = torch.randn(1, 2, 3, channels, requires_grad=True) * 0.03
    image.retain_grad()
    output = ffn(image)
    assert output.shape == image.shape
    output.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    for name, parameter in ffn.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_grouped_ffn_all_experts_run_and_merge_in_order():
    ffn = NRFFN(64)
    with torch.no_grad():
        for parameter in ffn.parameters():
            parameter.zero_()
        # Expert 0 and expert 1 both read channel 0 of the full input.
        ffn.expand.weight[0, 0] = 1
        ffn.expand.weight[128, 0] = 2
        ffn.contract.weight[0, 0, 0] = 1
        ffn.contract.weight[1, 0, 0] = 1
        ffn.project.weight[0, 0] = 1
        ffn.project.weight[1, 32] = 1
    image = torch.zeros(1, 1, 1, 64)
    image[..., 0] = 1
    result = ffn(image)
    assert result[0, 0, 0, 0].item() == pytest.approx(1.285888671875)
    assert result[0, 0, 0, 1].item() == pytest.approx(3.130859375)
    assert torch.count_nonzero(result).item() == 2


def test_split512_activation_occurs_only_inside_branch_middle():
    ffn = NRFFN(512)
    with torch.no_grad():
        for parameter in ffn.parameters():
            parameter.zero_()
        ffn.expand.weight.copy_(torch.eye(512))
        for branch in range(8):
            ffn.branch_expand.weight[branch, 0, 0] = -1
            ffn.branch_contract.weight[branch, 0, 0] = 1
        ffn.contract.weight.copy_(torch.eye(512))
    image = -torch.ones(1, 1, 1, 512)
    result = ffn(image)
    assert torch.count_nonzero(result).item() == 8
    for channel in range(0, 512, 64):
        assert result[0, 0, 0, channel].item() == pytest.approx(1.285888671875)


def test_block_scaled_residuals_are_ffn_then_attention():
    block = NRBlock(32, phase=3)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
        block.ffn_scale.fill_(2)
        block.attention_skip.fill_(0.25)
    image = torch.randn(1, 3, 5, 32, requires_grad=True)
    result = block(image)
    torch.testing.assert_close(result, image * 0.5)
    result.sum().backward()
    torch.testing.assert_close(image.grad, torch.full_like(image, 0.5))


def test_pooling_requires_a_complete_2x2_and_zero_fills_align4_border():
    # ops.comp writes zero unless sx+1<width and sy+1<height. A generic ceil-mode
    # average pool would incorrectly retain the incomplete final row/column.
    image = torch.arange(5 * 7, dtype=torch.float32).reshape(1, 5, 7, 1)
    actual = pool2x_align4(image)
    expected = torch.zeros(1, 4, 4, 1)
    for y in range(2):
        for x in range(3):
            expected[0, y, x] = image[0, y * 2:y * 2 + 2, x * 2:x * 2 + 2].sum((0, 1)) / 4
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("valid,field", [
    ((256, 256), (320, 320)), ((512, 512), (576, 512)),
    ((768, 768), (832, 768)), ((832, 480), (832, 512)),
    ((1920, 1080), (1920, 1152)), ((3840, 2160), (3840, 2176)),
])
def test_geometry_matches_recorded_native_fields(valid, field):
    geometry = NRGeometry.from_valid(*valid)
    assert (geometry.full_width, geometry.full_height) == field
    assert len(geometry.levels) == 6
    assert geometry.levels[0][0] % 8 == geometry.levels[0][1] % 8 == 0


def test_complete_meta_network_has_exact_original_numbered_topology():
    with torch.device("meta"):
        model = NRGenerator(NRConfig(checkpoint_blocks=False))
    assert set(model.blocks) == {str(index) for index in range(71)}
    assert sum(parameter.numel() for parameter in model.parameters()) == 145755115
    assert sum(isinstance(module, NRBlock) for module in model.blocks.values()) == 70
    assert not isinstance(model.blocks["39"], NRBlock)
    assert not any(isinstance(module, (nn.LayerNorm, nn.Conv2d)) for module in model.modules())
    seen_widths = Counter()
    for number, width in WIDTHS_BY_NUMBER.items():
        block = model.blocks[str(number)]
        attention = next(module for module in block.modules()
                         if isinstance(module, (NRWindowAttention, NRGlobalAttention)))
        assert attention.qkv.in_features == width, number
        assert attention.qkv.out_features == width * 3, number
        assert tuple(block.ffn_scale.shape) == (width,), number
        assert tuple(block.attention_skip.shape) == (width,), number
        assert tuple(attention.attention_scale.shape) == (width // 32,), number
        assert attention.qkv.bias is None and attention.projection.bias is None
        if 31 <= number <= 38:
            assert isinstance(attention, NRGlobalAttention)
            assert not any("prior" in name for name, _ in attention.named_parameters())
        else:
            assert tuple(attention.prior.shape) == (width // 32, 64, 64), number
        seen_widths[width] += 1
    assert seen_widths == {32: 10, 64: 8, 128: 12, 256: 16, 512: 16, 1024: 8}
    assert tuple(model.input_adapter.weight.shape) == (32, 16)
    assert tuple(model.head.weight.shape) == (4, 32)
    assert model.input_adapter.bias is None and model.head.bias is None
    for end, channels in ((4, 32), (8, 64), (14, 128), (22, 256), (30, 512)):
        assert tuple(model.encoder_projections[str(end)].weight.shape) == (channels * 2, channels)
    assert tuple(model.blocks["39"].projection.weight.shape) == (512, 1024)
    assert tuple(model.blocks["39"].skip_scale.shape) == (512,)
    for start, channels in ((48, 256), (56, 128), (62, 64), (66, 32)):
        assert tuple(model.decoder_projections[str(start)].weight.shape) == (channels, channels * 2)
        assert tuple(model.decoder_skip_scales[str(start)].shape) == (channels,)
    assert model.blocks["0"].phase == 0 and model.blocks["70"].phase == 1
    # Decoder window phases continue their encoder scale's counter.
    assert model.blocks["40"].phase == 0
    assert model.blocks["48"].phase == 0
    assert model.blocks["56"].phase == 2
    assert model.blocks["62"].phase == model.blocks["66"].phase == 0


def _meta_generator() -> NRGenerator:
    with torch.device("meta"):
        model = NRGenerator(NRConfig(checkpoint_blocks=False))
    return model


def test_meta_forward_visits_all_numbered_entries_at_native_stage_sizes():
    model = _meta_generator()
    visits = []
    handles = []
    for number, block in model.blocks.items():
        def record(module, inputs, output, index=int(number)):
            visits.append((index, tuple(output.shape)))
        handles.append(block.register_forward_hook(record))
    try:
        # Recorded 512x512 input uses a 576x512 field. These stage sizes come
        # from the native geometry, not from introspecting the new graph.
        features = torch.empty(1, 16, 512, 576, device="meta")
        head = model.forward_features(features)
    finally:
        for handle in handles:
            handle.remove()
    assert head.shape == (1, 4, 512, 576)
    assert [index for index, _ in visits] == list(range(71))
    shapes = dict(visits)
    expected = {
        0: (1, 512, 576, 32), 1: (1, 256, 288, 32),
        5: (1, 128, 144, 64), 9: (1, 64, 72, 128),
        15: (1, 32, 36, 256), 23: (1, 16, 20, 512),
        31: (1, 8, 12, 1024), 38: (1, 8, 12, 1024),
        39: (1, 16, 20, 512), 40: (1, 16, 20, 512),
        48: (1, 32, 36, 256), 56: (1, 64, 72, 128),
        62: (1, 128, 144, 64), 66: (1, 256, 288, 32),
        70: (1, 512, 576, 32),
    }
    for number, shape in expected.items():
        assert shapes[number] == shape, number


def test_transition39_projects_then_replicates_and_adds_scaled_skip():
    transition = NRTransition()
    with torch.no_grad():
        transition.projection.weight.zero_()
        transition.projection.weight[:, :512].copy_(torch.eye(512))
        transition.skip_scale.fill_(0.25)
    low = torch.zeros(1, 2, 2, 1024)
    low[0, 0, 0, :512] = 1
    low[0, 0, 1, :512] = 2
    low[0, 1, 0, :512] = 3
    low[0, 1, 1, :512] = 4
    skip = torch.ones(1, 3, 4, 512)
    actual = transition(low, skip)
    expected = torch.tensor([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4]], dtype=torch.float32) + 0.25
    torch.testing.assert_close(actual[0, :, :, 0], expected)
    assert torch.equal(actual[0, :, :, 0], actual[0, :, :, -1])


def test_preprocessing_sixteen_lanes_native_controls_and_reflection():
    model = _meta_generator()
    rgb = torch.linspace(0, 1, 256).reshape(1, 1, 1, 256).expand(1, 3, 256, 256)
    noise = torch.linspace(-1, 1, 320).reshape(1, 1, 1, 320).expand(1, 3, 320, 320)
    features = model.prepare_features(rgb, noise=noise, controls={
        "style": 64, "local_tone": 0.2, "local_structure": 0.6,
        "skin_structure": -1, "auto_mask": True,
    })
    assert features.shape == (1, 16, 320, 320)
    torch.testing.assert_close(features[:, :3], noise)
    assert torch.all(features[:, 3] == 1)
    torch.testing.assert_close(features[:, 4:7, :256, :256], (rgb - 0.5) * 0.125)
    torch.testing.assert_close(features[:, 7:10], features[:, 4:7])
    # x=256 reflects to 254 without repeating the edge. Explicit full-field
    # noise is independent from this reflected RGB coordinate.
    torch.testing.assert_close(features[:, 4:7, 256, 256], (rgb[:, :, 254, 254] - 0.5) * 0.125)
    assert features[0, 0, 256, 256] != features[0, 0, 254, 254]
    for lane, value in ((10, 0.5), (11, 0.2), (12, 1), (13, 0.6), (14, 0.6), (15, 0)):
        torch.testing.assert_close(features[:, lane], torch.full_like(features[:, lane], value))
    manual = model.prepare_features(rgb, controls={"auto_mask": False, "local_structure": 0.4})
    for lane, value in ((12, 0.4), (13, -1), (14, -1)):
        torch.testing.assert_close(manual[:, lane], torch.full_like(manual[:, lane], value))


def test_history_mask_falls_back_to_current_in_features_and_preserves_gradients():
    model = _meta_generator()
    current = torch.full((1, 3, 256, 256), 0.4, requires_grad=True)
    previous = torch.full_like(current, 0.1)
    mask = torch.ones(1, 1, 256, 256)
    mask[..., :128] = 0
    features = model.prepare_features(current, previous=previous, history_mask=mask)
    expected_history = torch.where(mask.bool(), previous, current)
    torch.testing.assert_close(features[:, 7:10, :256, :256], (expected_history - 0.5) * 0.125)
    features[:, 4:10, :256, :256].sum().backward()
    assert current.grad is not None and torch.isfinite(current.grad).all()
    assert torch.all(current.grad[..., :128] > current.grad[..., 128:])


def test_native_padding_reflects_once_then_clamps_far_coordinates():
    model = _meta_generator()
    horizontal = 0.1 + torch.arange(128, dtype=torch.float32) / 256
    rgb = horizontal.reshape(1, 1, 1, 128).expand(1, 3, 128, 128)
    features = model.prepare_features(rgb)
    assert features.shape == (1, 16, 320, 320)
    # preprocess.wgsl uses one mirror, followed by source-coordinate clamping.
    # Beyond 2*valid-2, repeated reflection would reintroduce image content and
    # therefore produce a different input than the original shader.
    indices = [min(127, max(0, x if x < 128 else 2 * 128 - x - 2)) for x in range(320)]
    expected_row = (horizontal[indices] - 0.5) * 0.125
    torch.testing.assert_close(features[0, 4, 0], expected_row)
    torch.testing.assert_close(features[0, 4, 319], expected_row)
    assert features[0, 4, 0, 128].item() == pytest.approx((horizontal[126].item() - 0.5) * 0.125)
    torch.testing.assert_close(features[0, 4, :, 255:],
                               torch.full_like(features[0, 4, :, 255:], (0.1 - 0.5) * 0.125))


def test_residual_and_temporal_head_composition_and_gradients(monkeypatch):
    model = _meta_generator()
    model.blend_scale = nn.Parameter(torch.tensor(0.5))
    current = torch.full((1, 3, 2, 3), 0.4, requires_grad=True)
    previous = torch.full((1, 3, 2, 3), 0.1, requires_grad=True)
    head = torch.zeros(1, 4, 2, 3)
    head[:, :3] = 0.8
    head.requires_grad_()
    monkeypatch.setattr(model, "raw_head", lambda *args, **kwargs: head)
    mask = torch.ones(1, 1, 2, 3)
    mask[..., 0] = 0
    actual = model(current, previous=previous, history_mask=mask)
    neural = 0.4 + 0.8 / 4
    mixed = neural * 0.75 + 0.1 * 0.25  # sigmoid(0) * blend_scale(0.5) = .25
    expected = torch.full_like(current, mixed)
    expected[..., 0] = neural
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(current.grad).all() and torch.isfinite(previous.grad).all()
    assert model.blend_scale.grad is not None and torch.isfinite(model.blend_scale.grad)
    assert torch.count_nonzero(head.grad[:, 3, :, 1:]).item() > 0


def test_no_history_blend_has_zero_gradient_but_remains_in_autograd():
    model = _meta_generator()
    model.blend_scale = nn.Parameter(torch.tensor(0.5))
    current = torch.full((1, 3, 2, 3), 0.4, requires_grad=True)
    head = torch.zeros(1, 4, 2, 3, requires_grad=True)
    actual = model.compose_head(current, head)
    torch.testing.assert_close(actual, current)
    actual.sum().backward()
    assert model.blend_scale.grad is not None
    assert model.blend_scale.grad.item() == 0
    assert head.grad is not None and torch.isfinite(head.grad).all()
