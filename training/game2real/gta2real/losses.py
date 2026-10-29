"""Unpaired translation losses with explicit source and temporal conventions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def _tensors(value: Tensor | Sequence[Tensor]) -> list[Tensor]:
    result = [value] if isinstance(value, Tensor) else list(value)
    if not result:
        raise ValueError("at least one tensor is required")
    return result


def _paired_features(
    fake: Tensor | Sequence[Tensor], source: Tensor | Sequence[Tensor]
) -> list[tuple[Tensor, Tensor]]:
    first, second = _tensors(fake), _tensors(source)
    if len(first) != len(second):
        raise ValueError("fake and source must have the same number of feature levels")
    pairs = list(zip(first, second))
    for generated, target in pairs:
        if generated.shape != target.shape:
            raise ValueError("fake and source feature shapes must match")
    return pairs


def hinge_discriminator_loss(
    real: Tensor | Sequence[Tensor], fake: Tensor | Sequence[Tensor]
) -> Tensor:
    """Average the real/fake hinge objectives over discriminator scales.

    The caller detaches generated images before the discriminator update. This
    function does not detach discriminator predictions.
    """
    real_scores, fake_scores = _tensors(real), _tensors(fake)
    if len(real_scores) != len(fake_scores):
        raise ValueError("real and fake discriminator scales must match")
    return torch.stack([
        F.relu(1 - positive).mean() + F.relu(1 + negative).mean()
        for positive, negative in zip(real_scores, fake_scores)
    ]).mean()


def generator_adversarial_loss(fake: Tensor | Sequence[Tensor]) -> Tensor:
    """Non-saturating hinge-generator objective averaged over scales."""
    return torch.stack([-score.mean() for score in _tensors(fake)]).mean()


def perceptual_content_loss(
    fake_feats: Tensor | Sequence[Tensor], source_feats: Tensor | Sequence[Tensor]
) -> Tensor:
    """Mean L1 between channel-normalized VGG features at matching locations.

    This is content preservation against the source, not a claim that the source
    is a paired photorealistic target. Source features act as fixed targets.
    """
    values = []
    for generated, target in _paired_features(fake_feats, source_feats):
        if generated.ndim != 4:
            raise ValueError("perceptual features must be BCHW tensors")
        generated = F.normalize(generated, dim=1, eps=1e-6)
        target = F.normalize(target.detach(), dim=1, eps=1e-6)
        values.append(F.l1_loss(generated, target))
    return torch.stack(values).mean()


def dino_structure_loss(fake: Tensor, source: Tensor) -> Tensor:
    """Cosine distance between corresponding DINO patch descriptors."""
    if fake.shape != source.shape or fake.ndim not in (3, 4):
        raise ValueError("DINO features must have matching BCHW or BNC shapes")
    dimension = 1 if fake.ndim == 4 else -1
    return (1 - F.cosine_similarity(fake, source.detach(), dim=dimension, eps=1e-6)).clamp_min(0).mean()


def patch_nce_loss(
    fake_feats: Tensor | Sequence[Tensor],
    source_feats: Tensor | Sequence[Tensor],
    num_patches: int = 128,
    temperature: float = 0.07,
) -> Tensor:
    """Contrast corresponding spatial features against other positions per image.

    Both branches sample the same coordinates. Negatives come only from the same
    image, so the similarity matrices have shape BxNxN, not (B*N)x(B*N). There is
    no trainable projection head and the source descriptors are detached targets.
    """
    if num_patches < 1 or temperature <= 0:
        raise ValueError("num_patches and temperature must be positive")
    values = []
    for generated, target in _paired_features(fake_feats, source_feats):
        if generated.ndim != 4:
            raise ValueError("PatchNCE features must be BCHW tensors")
        batch, channels, height, width = generated.shape
        count = min(num_patches, height * width)
        if count < 1:
            raise ValueError("PatchNCE feature maps must have a spatial extent")
        indices = torch.randperm(height * width, device=generated.device)[:count]
        queries = generated.flatten(2)[:, :, indices].transpose(1, 2)
        keys = target.detach().flatten(2)[:, :, indices].transpose(1, 2)
        queries = F.normalize(queries, dim=-1, eps=1e-6)
        keys = F.normalize(keys, dim=-1, eps=1e-6)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / temperature
        labels = torch.arange(count, device=scores.device).expand(batch, count)
        values.append(F.cross_entropy(scores.reshape(batch * count, count), labels.reshape(-1)))
    return torch.stack(values).mean()


def semantic_consistency_loss(
    fake_logits: Tensor,
    source_logits: Tensor | None,
    labels: Tensor | None = None,
    confidence: float = 0.6,
    ignore_index: int = 255,
) -> Tensor:
    """Use supplied class labels, otherwise confident detached teacher KL targets.

    A class-label tensor is BHW (or B1HW); valid labels take precedence over the
    source prediction. Without labels, confidence gates source pixels. An empty
    valid set returns a zero that remains connected to generated-image autograd.
    """
    if fake_logits.ndim != 4:
        raise ValueError("semantic logits must be BCHW")
    if labels is not None:
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 3 or labels.shape[0] != fake_logits.shape[0]:
            raise ValueError("semantic labels must be BHW or B1HW")
        labels = labels.to(device=fake_logits.device, dtype=torch.long)
        invalid = (labels != ignore_index) & ((labels < 0) | (labels >= fake_logits.shape[1]))
        if bool(invalid.any()):
            raise ValueError("semantic labels must use teacher train IDs 0..C-1 or ignore_index; configure data.label_map")
        if not bool((labels != ignore_index).any()):
            return fake_logits.sum() * 0
        logits = F.interpolate(fake_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        return F.cross_entropy(logits, labels, ignore_index=ignore_index)
    if source_logits is None or source_logits.ndim != 4:
        raise ValueError("source semantic logits are required when labels are absent")
    if fake_logits.shape[:2] != source_logits.shape[:2]:
        raise ValueError("source and fake semantic batch/class dimensions must match")
    if not 0 <= confidence <= 1:
        raise ValueError("semantic confidence must be in [0, 1]")
    target = source_logits.detach().softmax(dim=1)
    valid = target.amax(dim=1) >= confidence
    if not bool(valid.any()):
        return fake_logits.sum() * 0
    logits = F.interpolate(fake_logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    divergence = F.kl_div(F.log_softmax(logits, dim=1), target, reduction="none").sum(dim=1)
    return divergence[valid].mean()


def _same_images(fake: Tensor, source: Tensor) -> None:
    if fake.ndim != 4 or fake.shape != source.shape:
        raise ValueError("images must have matching BCHW shapes")


def _sobel(image: Tensor) -> Tensor:
    if image.shape[1] == 3:
        weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        luminance = (image * weights).sum(dim=1, keepdim=True)
    elif image.shape[1] == 1:
        luminance = image
    else:
        raise ValueError("edge loss expects RGB or grayscale images")
    kernels = image.new_tensor([
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    ]).unsqueeze(1) / 8
    return F.conv2d(F.pad(luminance, (1, 1, 1, 1), mode="replicate"), kernels)


def edge_loss(fake: Tensor, src: Tensor) -> Tensor:
    """Preserve luminance Sobel gradients and source-edge orientations."""
    _same_images(fake, src)
    generated, target = _sobel(fake), _sobel(src.detach())
    gradients = F.smooth_l1_loss(generated, target)
    strength = target.square().sum(dim=1).sqrt()
    valid = strength > 1e-3
    if not bool(valid.any()):
        return gradients
    cosine = (F.normalize(generated, dim=1, eps=1e-6) * F.normalize(target, dim=1, eps=1e-6)).sum(dim=1)
    directions = (1 - cosine).clamp_min(0)
    return gradients + 0.1 * (directions[valid] * strength[valid]).sum() / strength[valid].sum().clamp_min(1e-6)


def low_frequency_loss(fake: Tensor, src: Tensor, kernel: int = 32) -> Tensor:
    """Smooth-L1 on block averages, preserving coarse scene layout and tone."""
    _same_images(fake, src)
    if kernel < 1:
        raise ValueError("the low-frequency averaging kernel must be positive")
    size = min(kernel, fake.shape[-2], fake.shape[-1])
    generated = F.avg_pool2d(fake, size, stride=size, ceil_mode=True, count_include_pad=False)
    target = F.avg_pool2d(src.detach(), size, stride=size, ceil_mode=True, count_include_pad=False)
    return F.smooth_l1_loss(generated, target)


def total_variation_loss(image: Tensor) -> Tensor:
    """Anisotropic mean absolute variation over the available image axes."""
    if image.ndim != 4:
        raise ValueError("total variation expects BCHW images")
    value = image.sum() * 0
    if image.shape[-2] > 1:
        value = value + (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean()
    if image.shape[-1] > 1:
        value = value + (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean()
    return value


def masked_temporal_loss(
    current: Tensor, previous: Tensor, flow: Tensor, mask: Tensor | None = None
) -> Tensor:
    """Compare current to previous sampled with current-to-previous pixel flow.

    ``flow[:, 0, y, x]`` and ``flow[:, 1, y, x]`` give the x/y displacement
    from current pixel (x, y) to its previous-frame correspondence. Positive x
    samples farther right in ``previous``. This is backward sampling, not a
    forward splat. ``grid_sample`` uses ``align_corners=True``. Out-of-bounds
    samples are excluded. An optional B1HW/BHW mask weights visibility or a
    caller-computed photometric occlusion test (1 means valid, 0 excluded).

    Previous-frame gradients are retained; detach that frame at the caller when
    training independent frame pairs instead of differentiating through time.
    """
    _same_images(current, previous)
    batch, channels, height, width = current.shape
    if flow.shape != (batch, 2, height, width):
        raise ValueError("flow must be B2HW current-to-previous displacement in pixels")
    displacement = flow.to(device=current.device, dtype=current.dtype)
    y, x = torch.meshgrid(
        torch.arange(height, device=current.device, dtype=current.dtype),
        torch.arange(width, device=current.device, dtype=current.dtype), indexing="ij",
    )
    sample_x = x.unsqueeze(0) + displacement[:, 0]
    sample_y = y.unsqueeze(0) + displacement[:, 1]
    valid = (sample_x >= 0) & (sample_x <= width - 1) & (sample_y >= 0) & (sample_y <= height - 1)
    grid_x = sample_x * (2 / (width - 1)) - 1 if width > 1 else sample_x * 0
    grid_y = sample_y * (2 / (height - 1)) - 1 if height > 1 else sample_y * 0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    warped = F.grid_sample(previous, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    weights = valid.unsqueeze(1).to(dtype=current.dtype)
    if mask is not None:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape != (batch, 1, height, width):
            raise ValueError("the temporal mask must be B1HW or BHW")
        weights = weights * mask.detach().to(device=current.device, dtype=current.dtype).clamp(0, 1)
    return ((current - warped).abs() * weights).sum() / (weights.sum() * channels).clamp_min(1)
