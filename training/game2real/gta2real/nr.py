"""Trainable continuous reconstruction of the repository's 310.8 NR topology.

All 71 numbered entries are retained: 70 complete Transformer blocks and the
projection/upsample/skip transition numbered 39. Widths, FFN branches, four
window phases, six pooling levels and eight global ViT blocks match nr_graph.cpp.
There are no convolutions, LayerNorms, diffusion steps, or reduced-width variants.

This is NOT a bit-exact replacement for the FP8 Vulkan deployment. Training uses
continuous floating-point matrices, the original cubic activation without half
publication, and conventional differentiable softmax instead of the native
half-bit exponential approximation. Autocast is controlled by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


WINDOW_PHASES = ((0, 0), (4, 4), (4, 0), (0, 4))  # positive left/top padding
_RESIDUAL_INITIALIZATION_SCALE = 1 / math.sqrt(2 * 70)


@dataclass(frozen=True)
class NRConfig:
    checkpoint_blocks: bool = True
    noise_mode: str = "zero"
    noise_std: float = 1.0
    local_tone: float = 1.0
    local_structure: float = 1.0
    skin_structure: float = -1.0
    style: float = 0.0
    auto_mask: bool = True

    def __post_init__(self) -> None:
        if self.noise_mode not in ("zero", "gaussian"):
            raise ValueError("NR noise_mode must be zero or gaussian")
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("NR noise_std must be finite and nonnegative")
        for name in ("local_tone", "local_structure", "skin_structure", "style"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"NR control {name} must be finite")


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _field_alignment(valid: int) -> int:
    reductions = 0
    size = valid
    for level in range(6):
        half = _align_up((size + 1) // 2, 4)
        if half < size:
            reductions += 1
        if level == 0 and half % 8:
            reductions += 1
        size = half
    return 1 << reductions


def _levels(width: int, height: int) -> tuple[tuple[int, int], ...]:
    result = []
    for _ in range(6):
        width = _align_up((width + 1) // 2, 4)
        height = _align_up((height + 1) // 2, 4)
        result.append((width, height))
    return tuple(result)


@dataclass(frozen=True)
class NRGeometry:
    valid_width: int
    valid_height: int
    full_width: int
    full_height: int
    levels: tuple[tuple[int, int], ...]  # (width, height), encoder levels 0..5

    @classmethod
    def from_valid(cls, width: int, height: int) -> "NRGeometry":
        if width < 1 or height < 1:
            raise ValueError("NR image dimensions must be positive")
        align_width, align_height = _field_alignment(width), _field_alignment(height)
        full_width = max(320, _align_up(width, align_width))
        full_height = max(320, _align_up(height, align_height))
        if full_width % (4 * align_width) == 0 and full_height % (4 * align_height) == 0:
            full_width += align_width
        return cls(width, height, full_width, full_height, _levels(full_width, full_height))


def mp_cubic_silu(value: Tensor) -> Tensor:
    """Continuous version of shaders/common.glsl::mpCubicSilu, not torch SiLU.

    Only the polynomial's argument is clamped. Its final multiplier is the
    original unclamped value. Half/FP8 rounding publications are omitted.
    """
    bounded = value.clamp(-4.0, 4.0)
    inner = -0.055908203125 * bounded.abs() + 0.447265625
    return value * (bounded * inner + 0.89453125)


def pool2x_align4(x: Tensor) -> Tensor:
    """Native 2x2 box pooling on BHWC, zero outside complete 2x2 cells.

    The target is ceil(input/2) aligned to four, not a resize. A partial source
    cell contributes zero in its entirety, matching shaders/ops.comp.
    """
    batch, height, width, channels = x.shape
    out_height = _align_up((height + 1) // 2, 4)
    out_width = _align_up((width + 1) // 2, 4)
    rows, columns = height // 2, width // 2
    if rows == 0 or columns == 0:
        # Keep a zero-gradient path through all inputs, including tiny test fields.
        return (x.sum((1, 2), keepdim=True) * 0).expand(batch, out_height, out_width, channels)
    a = x[:, :rows * 2:2, :columns * 2:2]
    b = x[:, :rows * 2:2, 1:columns * 2:2]
    c = x[:, 1:rows * 2:2, :columns * 2:2]
    d = x[:, 1:rows * 2:2, 1:columns * 2:2]
    pooled = ((a + b) + (c + d)) * 0.25
    return F.pad(pooled, (0, 0, 0, out_width - columns, 0, out_height - rows))


def _nearest2x_to(x: Tensor, height: int, width: int) -> Tensor:
    """Replicate each low-resolution pixel twice, then crop to the target."""
    if height > x.shape[1] * 2 or width > x.shape[2] * 2:
        raise ValueError("NR nearest transition cannot cover the requested skip dimensions")
    return x.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)[:, :height, :width]


class NRGroupedLinear(nn.Module):
    """Independent bias-free linear maps; weight layout is [groups, out, in]."""

    def __init__(self, groups: int, in_channels: int, out_channels: int):
        super().__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(groups, out_channels, in_channels))
        nn.init.uniform_(self.weight, -1 / math.sqrt(in_channels), 1 / math.sqrt(in_channels))

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("...gi,goi->...go", x, self.weight)


class NRFFN(nn.Module):
    """The four exact width-dependent FFN topologies, without their outer skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        if channels in (32, 1024):
            hidden = 128 if channels == 32 else 4096
            self.expand = nn.Linear(channels, hidden, bias=False)
            self.contract = nn.Linear(hidden, channels, bias=False)
            final = self.contract
        elif channels in (64, 128, 256):
            self.groups = channels // 32
            self.expand = nn.Linear(channels, self.groups * 128, bias=False)
            self.contract = NRGroupedLinear(self.groups, 128, 32)
            self.project = nn.Linear(channels, channels, bias=False)
            final = self.project
        elif channels == 512:
            self.groups = 8
            self.expand = nn.Linear(512, 512, bias=False)
            self.branch_expand = NRGroupedLinear(8, 64, 256)
            self.branch_contract = NRGroupedLinear(8, 256, 64)
            self.contract = nn.Linear(512, 512, bias=False)
            final = self.contract
        else:
            raise ValueError("NR FFN width must be 32, 64, 128, 256, 512, or 1024")
        # Nonzero residual branches preserve gradient paths while avoiding the
        # activation growth of stacking 70 unnormalized residual blocks at init.
        nn.init.normal_(final.weight, std=_RESIDUAL_INITIALIZATION_SCALE / math.sqrt(final.in_features))

    def forward(self, x: Tensor) -> Tensor:
        if self.channels in (32, 1024):
            return self.contract(mp_cubic_silu(self.expand(x)))
        if self.channels in (64, 128, 256):
            expanded = mp_cubic_silu(self.expand(x)).reshape(*x.shape[:-1], self.groups, 128)
            narrowed = self.contract(expanded).flatten(-2)
            return self.project(narrowed)
        # There is no activation after the initial 512 -> 512 mixing matrix.
        branches = self.expand(x).reshape(*x.shape[:-1], 8, 64)
        expanded = mp_cubic_silu(self.branch_expand(branches))
        return self.contract(self.branch_contract(expanded).flatten(-2))


def _cosine_qk(query: Tensor, key: Tensor, scale: Tensor, global_attention: bool = False):
    # Norms accumulate in fp32, while the caller's autocast dtype is retained for
    # SDPA. scale=1.0 below is necessary: Q already carries the learned scale.
    query = F.normalize(query.float(), dim=-1, eps=1e-6).to(query.dtype)
    key = F.normalize(key.float(), dim=-1, eps=1e-6).to(key.dtype)
    multiplier = scale.to(query.dtype).reshape(1, -1, 1, 1)
    if global_attention:
        multiplier = multiplier * math.sqrt(32)
    return query * multiplier, key


class NRWindowAttention(nn.Module):
    """8x8 cosine attention on zero-padded shifted windows, never cyclic roll.

    QKV matrices use the native head-major output layout: [head, Q32,K32,V32].
    All 64 keys remain in softmax, including zero keys outside the field.
    The learned prior is a full [heads,64,64] matrix in row-major token order.
    """

    def __init__(self, channels: int):
        super().__init__()
        if channels not in (32, 64, 128, 256, 512):
            raise ValueError("NR window attention width must be 32, 64, 128, 256, or 512")
        self.channels, self.heads = channels, channels // 32
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.projection = nn.Linear(channels, channels, bias=False)
        self.prior = nn.Parameter(torch.zeros(self.heads, 64, 64))
        self.attention_scale = nn.Parameter(torch.ones(self.heads))
        nn.init.normal_(self.projection.weight, std=_RESIDUAL_INITIALIZATION_SCALE / math.sqrt(channels))

    def forward(self, x: Tensor, phase: int = 0) -> Tensor:
        if phase not in range(4):
            raise ValueError("NR window phase must be 0, 1, 2, or 3")
        batch, height, width, channels = x.shape
        shift_x, shift_y = WINDOW_PHASES[phase]
        right, bottom = (-(width + shift_x)) % 8, (-(height + shift_y)) % 8
        qkv = F.pad(self.qkv(x), (0, 0, shift_x, right, shift_y, bottom))
        padded_height, padded_width = qkv.shape[1:3]
        windows = (qkv.reshape(batch, padded_height // 8, 8, padded_width // 8, 8, channels * 3)
                   .permute(0, 1, 3, 2, 4, 5).reshape(-1, 64, self.heads, 3, 32))
        query, key, value = windows.permute(3, 0, 2, 1, 4).unbind(0)
        query, key = _cosine_qk(query, key, self.attention_scale)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=self.prior.to(query.dtype).unsqueeze(0),
            dropout_p=0.0, scale=1.0)
        windows = attended.transpose(1, 2).reshape(-1, 8, 8, channels)
        image = (windows.reshape(batch, padded_height // 8, padded_width // 8, 8, 8, channels)
                 .permute(0, 1, 3, 2, 4, 5).reshape(batch, padded_height, padded_width, channels))
        return self.projection(image[:, shift_y:shift_y + height, shift_x:shift_x + width])


class NRGlobalAttention(nn.Module):
    """Global 1024-channel, 32-head ViT attention with no relative prior."""

    def __init__(self, channels: int = 1024):
        super().__init__()
        if channels != 1024:
            raise ValueError("the original NR global attention has exactly 1024 channels")
        self.channels, self.heads = channels, 32
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.projection = nn.Linear(channels, channels, bias=False)
        self.attention_scale = nn.Parameter(torch.ones(self.heads))
        nn.init.normal_(self.projection.weight, std=_RESIDUAL_INITIALIZATION_SCALE / math.sqrt(channels))

    def forward(self, x: Tensor, phase: int = 0) -> Tensor:
        batch, height, width, channels = x.shape
        tokens = height * width
        qkv = self.qkv(x).reshape(batch, tokens, self.heads, 3, 32)
        query, key, value = qkv.permute(3, 0, 2, 1, 4).unbind(0)
        query, key = _cosine_qk(query, key, self.attention_scale, global_attention=True)
        # Native token padding is exactly removed from its softmax denominator;
        # attending over valid tokens directly is the continuous equivalent.
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, scale=1.0)
        attended = attended.transpose(1, 2).reshape(batch, height, width, channels)
        return self.projection(attended)


class NRBlock(nn.Module):
    """FFN + scaled skip, followed by attention + scaled skip; no normalization."""

    def __init__(self, channels: int, global_attention: bool = False, phase: int = 0):
        super().__init__()
        self.channels = channels
        self.phase = phase
        self.global_attention = global_attention
        self.ffn = NRFFN(channels)
        self.ffn_scale = nn.Parameter(torch.ones(channels))
        self.attention = NRGlobalAttention(channels) if global_attention else NRWindowAttention(channels)
        self.attention_skip = nn.Parameter(torch.ones(channels))

    def forward(self, x: Tensor) -> Tensor:
        state = self.ffn(x) + x * self.ffn_scale.to(x.dtype)
        return self.attention(state, self.phase) + state * self.attention_skip.to(state.dtype)


class NRTransition(nn.Module):
    """Numbered block 39: projection, nearest 2x, and learned skip addition only."""

    def __init__(self):
        super().__init__()
        self.channels = 512
        self.projection = nn.Linear(1024, 512, bias=False)
        self.skip_scale = nn.Parameter(torch.ones(512))

    def forward(self, low_resolution: Tensor, skip: Tensor) -> Tensor:
        projected = self.projection(low_resolution)
        return _nearest2x_to(projected, skip.shape[1], skip.shape[2]) + skip * self.skip_scale.to(skip.dtype)


def _block_specifications() -> tuple[tuple[int, int, int | None], ...]:
    """(number, channels, phase); None marks global blocks or transition 39."""
    specifications: dict[int, tuple[int, int, int | None]] = {}
    specifications[0] = (0, 32, 0)
    specifications[70] = (70, 32, 1)
    for first, last, channels, offset in (
        (1, 4, 32, 0), (5, 8, 64, 0), (9, 14, 128, 0),
        (15, 22, 256, 0), (23, 30, 512, 0), (40, 47, 512, 8),
        (48, 55, 256, 8), (56, 61, 128, 6), (62, 65, 64, 4), (66, 69, 32, 4),
    ):
        for number in range(first, last + 1):
            specifications[number] = (number, channels, (offset + number - first) % 4)
    for number in range(31, 39):
        specifications[number] = (number, 1024, None)
    specifications[39] = (39, 512, None)
    return tuple(specifications[number] for number in range(71))


NR_TOPOLOGY = _block_specifications()


def _reflection_indices(length: int, padded: int, device: torch.device) -> Tensor:
    """Mirror once without edge repetition, then clamp to the source rectangle."""
    indices = torch.arange(padded, device=device)
    mirrored = torch.where(indices < length, indices, 2 * length - indices - 2)
    return mirrored.clamp(0, length - 1)


def _reflect_to(x: Tensor, height: int, width: int) -> Tensor:
    rows = _reflection_indices(x.shape[-2], height, x.device)
    columns = _reflection_indices(x.shape[-1], width, x.device)
    return x.index_select(-2, rows).index_select(-1, columns)


class NRGenerator(nn.Module):
    """Original 145,755,115-parameter NR graph, trained from scratch in PyTorch.

    RGB/previous/noise use BCHW. Internally blocks use BHWC, as the original
    graph's per-pixel channel matrices do. `previous` must already be reprojected.
    Missing or invalid history falls back to the current frame and blends at 0.
    """

    uses_history = True
    block_count = 71
    transformer_block_count = 70

    def __init__(self, config: NRConfig | None = None):
        super().__init__()
        self.config = config if config is not None else NRConfig()
        self.input_adapter = nn.Linear(16, 32, bias=False)
        self.blocks = nn.ModuleDict({
            str(number): (NRTransition() if number == 39 else
                          NRBlock(channels, global_attention=31 <= number <= 38, phase=phase or 0))
            for number, channels, phase in NR_TOPOLOGY
        })
        self.encoder_projections = nn.ModuleDict({
            str(last): nn.Linear(channels, channels * 2, bias=False)
            for last, channels in ((4, 32), (8, 64), (14, 128), (22, 256), (30, 512))
        })
        self.decoder_projections = nn.ModuleDict({
            str(first): nn.Linear(channels * 2, channels, bias=False)
            for first, channels in ((48, 256), (56, 128), (62, 64), (66, 32))
        })
        self.decoder_skip_scales = nn.ParameterDict({
            str(first): nn.Parameter(torch.ones(channels))
            for first, channels in ((48, 256), (56, 128), (62, 64), (66, 32))
        })
        self.input_scale = nn.Parameter(torch.ones(32))
        self.adapter_scale = nn.Parameter(torch.ones(32))
        self.head = nn.Linear(32, 4, bias=False)
        nn.init.zeros_(self.head.weight)
        self.blend_scale = nn.Parameter(torch.tensor(0.73974609375))

    def _run(self, number: int, *inputs: Tensor) -> Tensor:
        block = self.blocks[str(number)]
        if self.config.checkpoint_blocks and self.training and torch.is_grad_enabled():
            return checkpoint(block, *inputs, use_reentrant=False)
        return block(*inputs)

    @staticmethod
    def _validate_rgb(rgb: Tensor) -> None:
        if rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape[0], *rgb.shape[-2:]) < 1:
            raise ValueError("NR expects nonempty Bx3xHxW RGB input")
        if not rgb.is_floating_point():
            raise TypeError("NR RGB input must be floating point in [0, 1]")

    @staticmethod
    def _history(rgb: Tensor, previous: Tensor | None, history_mask: Tensor | None):
        if previous is None:
            return rgb, torch.zeros_like(rgb[:, :1])
        if previous.shape != rgb.shape:
            raise ValueError("NR previous must be reprojected to the current Bx3xHxW image")
        previous = previous.to(device=rgb.device, dtype=rgb.dtype)
        if history_mask is None:
            valid = torch.ones_like(rgb[:, :1])
        else:
            if history_mask.ndim == 3:
                history_mask = history_mask.unsqueeze(1)
            if history_mask.shape != rgb[:, :1].shape:
                raise ValueError("NR history_mask must be Bx1xHxW or BxHxW")
            valid = history_mask.to(device=rgb.device, dtype=rgb.dtype).clamp(0, 1)
        history = torch.where(valid.bool(), previous, rgb)
        return history, valid

    def prepare_features(self, rgb: Tensor, previous: Tensor | None = None,
                         noise: Tensor | None = None, *, controls: Mapping[str, Any] | None = None,
                         history_mask: Tensor | None = None) -> Tensor:
        """Create all sixteen input lanes on the exact native padded field.

        Explicit noise overrides noise_mode/std and may be B3HW or B3HpWp.
        Controls are scalars or one value per batch item. Zero noise is a
        deterministic training default, not a claim about NVIDIA's training.
        """
        self._validate_rgb(rgb)
        batch, _, height, width = rgb.shape
        geometry = NRGeometry.from_valid(width, height)
        padded_height, padded_width = geometry.full_height, geometry.full_width
        history, _ = self._history(rgb, previous, history_mask)
        proxy = _reflect_to(rgb, padded_height, padded_width)
        history = _reflect_to(history, padded_height, padded_width)
        if noise is None:
            shape = (batch, 3, padded_height, padded_width)
            noise = (rgb.new_zeros(shape) if self.config.noise_mode == "zero" else
                     torch.randn(shape, device=rgb.device, dtype=rgb.dtype) * self.config.noise_std)
        else:
            if noise.shape == rgb.shape:
                noise = _reflect_to(noise.to(rgb), padded_height, padded_width)
            elif noise.shape == (batch, 3, padded_height, padded_width):
                noise = noise.to(rgb)
            else:
                raise ValueError("NR noise must match B3HW or the B3HpWp padded field")
        controls = dict(controls or {})
        allowed = {"style", "local_tone", "local_structure", "skin_structure", "auto_mask"}
        unknown = controls.keys() - allowed
        if unknown:
            raise ValueError(f"unknown NR controls: {sorted(unknown)}")

        def lane(name: str) -> Tensor:
            value = torch.as_tensor(controls.get(name, getattr(self.config, name)), device=rgb.device, dtype=rgb.dtype)
            if value.numel() == 1:
                value = value.reshape(1, 1, 1, 1)
            elif value.shape == (batch,):
                value = value.reshape(batch, 1, 1, 1)
            else:
                raise ValueError(f"NR control {name} must be scalar or a batch-length vector")
            return value.expand(batch, 1, padded_height, padded_width)

        one, zero = torch.ones_like(proxy[:, :1]), torch.zeros_like(proxy[:, :1])
        structure, skin = lane("local_structure"), lane("skin_structure")
        automatic = lane("auto_mask") > 0
        structure_lane = torch.where(automatic, one, structure)
        skin_lane = torch.where(automatic, torch.where(skin < 0, structure, skin), -one)
        mask_lane = torch.where(automatic, structure, -one)
        return torch.cat((noise, one, (proxy - 0.5) * 0.125, (history - 0.5) * 0.125,
                          lane("style") / 128.0, lane("local_tone"),
                          structure_lane, skin_lane, mask_lane, zero), dim=1)

    def forward_features(self, features: Tensor) -> Tensor:
        """Run the complete graph on B16HpWp and return the raw B4HpWp head.

        Already-prepared fields are not repadded, which permits small-field
        mathematical tests without constructing a different-width network.
        """
        if features.ndim != 4 or features.shape[1] != 16 or min(features.shape[-2:]) < 1:
            raise ValueError("NR features must be Bx16xHpxWp")
        state = self.input_adapter(features.permute(0, 2, 3, 1))
        full_skip = self._run(0, state)
        state = pool2x_align4(full_skip)
        skips: dict[int, Tensor] = {}
        for first, last, channels in ((1, 4, 32), (5, 8, 64), (9, 14, 128), (15, 22, 256), (23, 30, 512)):
            for number in range(first, last + 1):
                state = self._run(number, state)
            skips[channels] = state
            state = self.encoder_projections[str(last)](pool2x_align4(state))
        for number in range(31, 39):
            state = self._run(number, state)
        state = self._run(39, state, skips[512])
        for number in range(40, 48):
            state = self._run(number, state)
        for first, last, channels in ((48, 55, 256), (56, 61, 128), (62, 65, 64), (66, 69, 32)):
            skip = skips[channels]
            projected = self.decoder_projections[str(first)](state)
            state = (_nearest2x_to(projected, skip.shape[1], skip.shape[2]) +
                     skip * self.decoder_skip_scales[str(first)].to(skip.dtype))
            for number in range(first, last + 1):
                state = self._run(number, state)
        state = (_nearest2x_to(state, full_skip.shape[1], full_skip.shape[2]) * self.input_scale.to(state.dtype) +
                 full_skip * self.adapter_scale.to(full_skip.dtype))
        state = self._run(70, state)
        return self.head(state).permute(0, 3, 1, 2).contiguous()

    def raw_head(self, rgb: Tensor, previous: Tensor | None = None, noise: Tensor | None = None,
                 *, controls: Mapping[str, Any] | None = None, history_mask: Tensor | None = None) -> Tensor:
        """Return the raw four-channel head cropped to the valid RGB rectangle."""
        features = self.prepare_features(rgb, previous, noise, controls=controls, history_mask=history_mask)
        return self.forward_features(features)[:, :, :rgb.shape[-2], :rgb.shape[-1]]

    def compose_head(self, rgb: Tensor, head: Tensor, previous: Tensor | None = None,
                     history_mask: Tensor | None = None) -> Tensor:
        """Native RGB residual and learned temporal mixture, without display style."""
        if head.shape != (rgb.shape[0], 4, *rgb.shape[-2:]):
            raise ValueError("NR head must match the current Bx4xHxW valid rectangle")
        history, valid = self._history(rgb, previous, history_mask)
        neural = (rgb + head[:, :3].to(rgb.dtype) / 4).clamp(0, 1)
        weight = (head[:, 3:4].float().sigmoid() * self.blend_scale).clamp(0, 1).to(rgb.dtype) * valid
        # Multiplication by an all-zero mask preserves the blend parameter's
        # autograd edge, without inventing history or a temporal blend.
        return neural + weight * (history - neural)

    def forward(self, rgb: Tensor, previous: Tensor | None = None, noise: Tensor | None = None,
                *, controls: Mapping[str, Any] | None = None, history_mask: Tensor | None = None) -> Tensor:
        head = self.raw_head(rgb, previous, noise, controls=controls, history_mask=history_mask)
        return self.compose_head(rgb, head, previous, history_mask)
