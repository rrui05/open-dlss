"""Trainable RGB restoration models; no diffusion or pretrained dependencies.

The generator has four feature scales, with the deepest scale serving as its
bottleneck. Each decoder stage concatenates its encoder skip before projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class GeneratorConfig:
    dim: int = 64
    depths: tuple[int, ...] = (2, 2, 6, 4)
    heads: tuple[int, ...] = (2, 4, 8, 16)
    window_size: int = 8
    mlp_ratio: float = 3.0
    max_residual: float = 0.35
    checkpoint_blocks: bool = True

    def __post_init__(self) -> None:
        if self.dim < 1 or self.window_size < 1:
            raise ValueError("dim and window_size must be positive")
        if len(self.depths) != 4 or len(self.heads) != 4:
            raise ValueError("depths and heads must each describe four scales")
        if any(depth < 1 for depth in self.depths):
            raise ValueError("each scale must contain at least one block")
        if any(head < 1 or (self.dim * 2**stage) % head
               for stage, head in enumerate(self.heads)):
            raise ValueError("each stage width must be divisible by its head count")
        if self.mlp_ratio <= 0 or not 0 < self.max_residual <= 1:
            raise ValueError("mlp_ratio must be positive and max_residual must be in (0, 1]")


def _window_partition(x: Tensor, window_size: int) -> Tensor:
    """BHWC -> (batch * windows, window_size**2, channels)."""
    batch, height, width, channels = x.shape
    return (x.reshape(batch, height // window_size, window_size,
                      width // window_size, window_size, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, window_size * window_size, channels))


def _window_reverse(windows: Tensor, window_size: int, batch: int,
                    height: int, width: int) -> Tensor:
    """Reverse _window_partition, retaining the padded spatial dimensions."""
    return (windows.reshape(batch, height // window_size, width // window_size,
                            window_size, window_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, height, width, -1))


def _make_window_mask(height: int, width: int, window_size: int, shift_size: int,
                      device: torch.device) -> Tensor | None:
    """Return allowed query/key pairs per window, including padding and wrap masks.

    True means attention is permitted, following PyTorch SDPA's convention.
    Padded queries attend only themselves, preventing all-masked softmax rows.
    Real queries never attend padded keys. The caller crops padded queries away.
    """
    padded_height = (height + window_size - 1) // window_size * window_size
    padded_width = (width + window_size - 1) // window_size * window_size
    has_padding = padded_height != height or padded_width != width
    if not shift_size and not has_padding:
        return None

    valid = torch.zeros((1, padded_height, padded_width, 1), device=device, dtype=torch.bool)
    valid[:, :height, :width, :] = True
    if shift_size:
        valid = torch.roll(valid, shifts=(-shift_size, -shift_size), dims=(1, 2))
    valid_windows = _window_partition(valid, window_size).squeeze(-1)
    tokens = window_size * window_size
    allowed = valid_windows.unsqueeze(1).expand(-1, tokens, -1)

    if shift_size:
        regions = torch.zeros((1, padded_height, padded_width, 1), device=device, dtype=torch.int64)
        height_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        width_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        label = 0
        for rows in height_slices:
            for columns in width_slices:
                regions[:, rows, columns, :] = label
                label += 1
        region_windows = _window_partition(regions, window_size).squeeze(-1)
        same_region = region_windows.unsqueeze(1) == region_windows.unsqueeze(2)
        allowed = allowed & same_region

    diagonal = torch.eye(tokens, device=device, dtype=torch.bool).unsqueeze(0)
    return torch.where(valid_windows.unsqueeze(-1), allowed, diagonal)


class WindowAttention(nn.Module):
    """Multi-head window attention with trainable relative position bias."""

    def __init__(self, dim: int, heads: int, window_size: int) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.projection = nn.Linear(dim, dim)
        self.relative_position_bias = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, heads))
        coordinates = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coordinates = coordinates.flatten(1)
        relative = coordinates[:, :, None] - coordinates[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous() + window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative.sum(-1), persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

    def forward(self, windows: Tensor, allowed: Tensor | None = None) -> Tensor:
        batch_windows, tokens, channels = windows.shape
        qkv = self.qkv(windows).reshape(batch_windows, tokens, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        bias = self.relative_position_bias[self.relative_position_index.reshape(-1)]
        bias = bias.reshape(tokens, tokens, self.heads).permute(2, 0, 1).to(query.dtype)
        attention_bias = bias.unsqueeze(0)
        if allowed is not None:
            windows_per_image = allowed.shape[0]
            if batch_windows % windows_per_image:
                raise ValueError("window mask does not match the number of windows")
            # Relative position bias remains differentiable; geometric masks do not.
            geometric_bias = torch.zeros_like(allowed, dtype=query.dtype).masked_fill(~allowed, float("-inf"))
            attention_bias = attention_bias + geometric_bias.unsqueeze(1)
            attention_bias = attention_bias.repeat(batch_windows // windows_per_image, 1, 1, 1)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_bias, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.projection(attended)


class LocalFeedForward(nn.Module):
    """A pointwise MLP with a depthwise spatial convolution for local detail."""

    def __init__(self, dim: int, ratio: float) -> None:
        super().__init__()
        hidden = max(1, int(dim * ratio))
        self.expand = nn.Linear(dim, hidden)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.contract = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.expand(x))
        x = self.depthwise(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.contract(F.gelu(x))


class SwinBlock(nn.Module):
    """Pre-normalized residual attention and local MLP; input/output are BCHW."""

    def __init__(self, dim: int, heads: int, window_size: int,
                 shift_size: int, mlp_ratio: float) -> None:
        super().__init__()
        if not 0 <= shift_size < window_size:
            raise ValueError("shift_size must be in [0, window_size)")
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm_attention = nn.LayerNorm(dim)
        self.attention = WindowAttention(dim, heads, window_size)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = LocalFeedForward(dim, mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        shortcut = x.permute(0, 2, 3, 1)
        normalized = self.norm_attention(shortcut)
        pad_height = (-height) % self.window_size
        pad_width = (-width) % self.window_size
        normalized = F.pad(normalized, (0, 0, 0, pad_width, 0, pad_height))
        padded_height, padded_width = height + pad_height, width + pad_width
        # One-window axes do not benefit from a cyclic shift.
        shift = self.shift_size if min(height, width) > self.window_size else 0
        if shift:
            normalized = torch.roll(normalized, shifts=(-shift, -shift), dims=(1, 2))
        allowed = _make_window_mask(height, width, self.window_size, shift, x.device)
        windows = _window_partition(normalized, self.window_size)
        attended = self.attention(windows, allowed)
        attended = _window_reverse(attended, self.window_size, batch, padded_height, padded_width)
        if shift:
            attended = torch.roll(attended, shifts=(shift, shift), dims=(1, 2))
        attended = attended[:, :height, :width, :]
        features = shortcut + attended
        features = features + self.mlp(self.norm_mlp(features))
        return features.permute(0, 3, 1, 2).contiguous()


class TransformerStage(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int,
                 config: GeneratorConfig) -> None:
        super().__init__()
        self.checkpoint_blocks = config.checkpoint_blocks
        self.blocks = nn.ModuleList([
            SwinBlock(dim, heads, config.window_size,
                      0 if index % 2 == 0 else config.window_size // 2, config.mlp_ratio)
            for index in range(depth)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled() and x.requires_grad:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class DecoderStage(nn.Module):
    def __init__(self, low_dim: int, dim: int, depth: int, heads: int,
                 config: GeneratorConfig) -> None:
        super().__init__()
        self.upsample_projection = nn.Conv2d(low_dim, dim, 3, padding=1)
        self.skip_projection = nn.Conv2d(dim * 2, dim, 1)
        self.blocks = TransformerStage(dim, depth, heads, config)

    def forward(self, low_resolution: Tensor, skip: Tensor) -> Tensor:
        # Explicit target sizes preserve every odd-sized encoder/decoder match.
        upsampled = F.interpolate(low_resolution, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        upsampled = self.upsample_projection(upsampled)
        merged = self.skip_projection(torch.cat((upsampled, skip), dim=1))
        return self.blocks(merged)


class TransformerUNet(nn.Module):
    """Four-scale Swin U-Net that predicts a bounded RGB residual.

    Inputs and outputs have shape Bx3xHxW and range [0, 1]. Initialization is
    exactly the identity on that range. Each spatial dimension may be odd.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__()
        self.config = config
        widths = [config.dim * 2**stage for stage in range(4)]
        self.stem = nn.Conv2d(3, widths[0], 3, padding=1)
        self.encoders = nn.ModuleList([
            TransformerStage(widths[stage], config.depths[stage], config.heads[stage], config)
            for stage in range(4)
        ])
        self.downsamples = nn.ModuleList([
            nn.Conv2d(widths[stage], widths[stage + 1], 3, stride=2, padding=1)
            for stage in range(3)
        ])
        self.decoders = nn.ModuleList([
            DecoderStage(widths[stage + 1], widths[stage], config.depths[stage], config.heads[stage], config)
            for stage in (2, 1, 0)
        ])
        self.residual_head = nn.Conv2d(widths[0], 3, 3, padding=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError("expected nonempty Bx3xHxW RGB input")
        if not x.is_floating_point():
            raise TypeError("RGB input must be floating point in [0, 1]")
        features = self.stem(x)
        skips: list[Tensor] = []
        for stage, encoder in enumerate(self.encoders):
            features = encoder(features)
            if stage < 3:
                skips.append(features)
                features = self.downsamples[stage](features)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            features = decoder(features, skip)
        residual = self.config.max_residual * torch.tanh(self.residual_head(features))
        return (x + residual).clamp(0.0, 1.0)


class _DiscriminatorHead(nn.Sequential):
    def __init__(self, channels: int, width: int) -> None:
        super().__init__(
            spectral_norm(nn.Conv2d(channels, width, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=False),
            spectral_norm(nn.Conv2d(width, width, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=False),
            spectral_norm(nn.Conv2d(width, width * 2, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=False),
            spectral_norm(nn.Conv2d(width * 2, width * 2, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=False),
            spectral_norm(nn.Conv2d(width * 2, 1, 1)),
        )


class MultiScaleDiscriminator(nn.Module):
    """Independent spectral-normalized heads on an external feature pyramid.

    By default the deepest three of four supplied scales are used. A caller may
    pass either the full pyramid or those selected scales. No backbone is owned
    or frozen here; gradients can flow through features into a generator.
    Returned tensors are unconstrained logits, with one tensor per used scale.
    """

    def __init__(self, in_channels: Sequence[int] = (64, 128, 256, 512),
                 width: int = 128, scales: int = 3) -> None:
        super().__init__()
        self.in_channels = tuple(in_channels)
        if not self.in_channels or any(channels < 1 for channels in self.in_channels):
            raise ValueError("in_channels must contain positive feature widths")
        if not 1 <= scales <= len(self.in_channels) or width < 1:
            raise ValueError("scales must select available feature levels and width must be positive")
        self.scales = scales
        self.selected_channels = self.in_channels[-scales:]
        self.heads = nn.ModuleList([_DiscriminatorHead(channels, width) for channels in self.selected_channels])

    def forward(self, feature_pyramid: Sequence[Tensor]) -> list[Tensor]:
        if len(feature_pyramid) not in (len(self.in_channels), self.scales):
            raise ValueError("feature pyramid must contain all configured scales or the selected deepest scales")
        selected = feature_pyramid[-self.scales:]
        logits = []
        for feature, channels, head in zip(selected, self.selected_channels, self.heads):
            if feature.ndim != 4 or feature.shape[1] != channels or min(feature.shape[-2:]) < 1:
                raise ValueError(f"expected Bx{channels}xHxW feature with positive spatial dimensions")
            logits.append(head(feature))
        return logits
