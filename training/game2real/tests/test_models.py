"""Small CPU tests for shape, initialization, masking, and trainable gradients."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from gta2real.models import (
    GeneratorConfig,
    MultiScaleDiscriminator,
    SwinBlock,
    TransformerUNet,
    _make_window_mask,
    _window_partition,
)


def tiny_config(**changes):
    config = GeneratorConfig(dim=8, depths=(2, 1, 1, 1), heads=(1, 2, 4, 8),
                             window_size=4, mlp_ratio=2.0, checkpoint_blocks=False)
    return replace(config, **changes)


@pytest.mark.parametrize("shape", [(1, 3, 17, 23), (1, 3, 64, 80), (1, 3, 1, 3)])
def test_generator_preserves_odd_shape_and_initial_identity(shape):
    model = TransformerUNet(tiny_config()).eval()
    source = torch.rand(shape)
    with torch.no_grad():
        output = model(source)
    assert output.shape == source.shape
    assert torch.equal(output, source)


def test_identity_head_receives_nonzero_training_gradient():
    model = TransformerUNet(tiny_config())
    source = torch.rand(1, 3, 13, 19) * 0.8 + 0.1
    target = (source + 0.05).clamp(0, 1)
    loss = torch.nn.functional.mse_loss(model(source), target)
    loss.backward()
    gradient = model.residual_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_nonzero_head_trains_encoder_with_checkpointing():
    torch.manual_seed(7)
    model = TransformerUNet(tiny_config(checkpoint_blocks=True))
    torch.nn.init.normal_(model.residual_head.weight, std=0.01)
    source = torch.rand(1, 3, 15, 21) * 0.8 + 0.1
    output = model(source)
    output.square().mean().backward()
    assert model.stem.weight.grad is not None
    assert torch.isfinite(model.stem.weight.grad).all()
    assert model.stem.weight.grad.abs().sum() > 0


def test_residual_bound_and_output_range_after_nonzero_head():
    model = TransformerUNet(tiny_config(max_residual=0.2)).eval()
    torch.nn.init.normal_(model.residual_head.weight, std=2.0)
    torch.nn.init.constant_(model.residual_head.bias, 1.0)
    source = torch.rand(1, 3, 11, 17)
    with torch.no_grad():
        output = model(source)
    assert torch.isfinite(output).all()
    assert output.min() >= 0
    assert output.max() <= 1
    assert (output - source).abs().max() <= 0.200001
    assert not torch.equal(output, source)


def test_checkpointing_is_disabled_without_grad():
    model = TransformerUNet(tiny_config(checkpoint_blocks=True)).train()
    with patch("gta2real.models.checkpoint", side_effect=AssertionError("unexpected checkpoint")):
        with torch.no_grad():
            model(torch.rand(1, 3, 9, 13))


def test_shift_mask_blocks_cyclic_wraparound():
    allowed = _make_window_mask(8, 8, 4, 2, torch.device("cpu"))
    assert allowed is not None
    assert allowed.shape == (4, 16, 16)
    assert allowed[0].all()
    assert allowed[-1].sum().item() == 64
    assert not allowed[-1, 0, -1]
    assert allowed.diagonal(dim1=-2, dim2=-1).all()


@pytest.mark.parametrize("shift", [0, 2])
def test_padding_mask_excludes_padded_keys_and_keeps_softmax_rows_valid(shift):
    height, width, window = 9, 11, 4
    allowed = _make_window_mask(height, width, window, shift, torch.device("cpu"))
    valid = torch.zeros(1, 12, 12, 1, dtype=torch.bool)
    valid[:, :height, :width] = True
    if shift:
        valid = torch.roll(valid, shifts=(-shift, -shift), dims=(1, 2))
    valid = _window_partition(valid, window).squeeze(-1)
    forbidden = valid.unsqueeze(-1) & ~valid.unsqueeze(1)
    assert allowed is not None
    assert not allowed[forbidden].any()
    assert allowed.any(dim=-1).all()


def test_shifted_attention_padding_has_finite_forward_and_backward():
    block = SwinBlock(dim=8, heads=2, window_size=4, shift_size=2, mlp_ratio=2.0)
    source = torch.randn(2, 8, 9, 11, requires_grad=True)
    output = block(source)
    output.square().mean().backward()
    assert output.shape == source.shape
    assert torch.isfinite(output).all()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    assert block.attention.relative_position_bias.grad is not None
    assert torch.isfinite(block.attention.relative_position_bias.grad).all()


def test_discriminator_consumes_external_features_and_backpropagates():
    discriminator = MultiScaleDiscriminator(in_channels=(4, 8, 16, 32), width=8, scales=3)
    features = [torch.randn(2, channels, size, size, requires_grad=True)
                for channels, size in [(4, 16), (8, 12), (16, 8), (32, 4)]]
    logits = discriminator(features)
    assert len(logits) == 3
    assert all(logit.shape[:2] == (2, 1) and torch.isfinite(logit).all() for logit in logits)
    sum(logit.mean() for logit in logits).backward()
    assert features[0].grad is None
    assert all(feature.grad is not None and torch.isfinite(feature.grad).all() for feature in features[1:])
    assert all(hasattr(head[0], "parametrizations") for head in discriminator.heads)


def test_discriminator_accepts_preselected_deepest_features():
    discriminator = MultiScaleDiscriminator(in_channels=(4, 8, 16, 32), width=8, scales=3).eval()
    with torch.no_grad():
        result = discriminator([torch.randn(1, channels, 4, 5) for channels in (8, 16, 32)])
    assert len(result) == 3


def test_configuration_rejects_incompatible_heads():
    with pytest.raises(ValueError, match="divisible"):
        tiny_config(heads=(3, 2, 4, 8))
