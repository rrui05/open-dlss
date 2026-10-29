"""Generator selection and current-to-previous history sampling.

The default is the fixed 310.8 NR graph. Historical checkpoints that contain
Swin dimensions but no architecture tag still select the earlier RGB model.
"""

from __future__ import annotations

from dataclasses import asdict
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models import GeneratorConfig, TransformerUNet


def architecture(settings: Mapping) -> str:
    name = settings.get("architecture")
    if name is None:
        name = "swin" if any(key in settings for key in ("dim", "depths", "heads")) else "nr"
    if name not in ("nr", "swin"):
        raise ValueError("model.architecture must be nr or swin")
    return name


def model_defaults(name: str = "nr") -> dict:
    if name == "nr":
        from .nr import NRConfig
        return {"architecture": name, **asdict(NRConfig())}
    if name == "swin":
        return {"architecture": name, **asdict(GeneratorConfig())}
    raise ValueError("model.architecture must be nr or swin")


def generator_config(settings: Mapping):
    name = architecture(settings)
    values = dict(settings)
    values.pop("architecture", None)
    allowed = set(model_defaults(name)) - {"architecture"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown options in model for {name}: {sorted(unknown)}")
    if name == "nr":
        from .nr import NRConfig
        return NRConfig(**values)
    for key in ("depths", "heads"):
        if key in values:
            values[key] = tuple(values[key])
    return GeneratorConfig(**values)


def build_generator(settings: Mapping) -> nn.Module:
    config = generator_config(settings)
    if architecture(settings) == "nr":
        from .nr import NRGenerator
        return NRGenerator(config)
    return TransformerUNet(config)


def reproject_history(previous: Tensor, current: Tensor, flow: Tensor,
                      mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Sample detached neural history with current-to-previous pixel flow.

    Returns history and a B1HW validity weight. Outside/occluded pixels use
    current RGB in the features and a zero final blend weight. Coordinates
    stay float32 under AMP, including at large resolutions.
    """
    if previous.shape != current.shape or current.ndim != 4 or current.shape[1] != 3:
        raise ValueError("previous and current must have matching B3HW shapes")
    batch, _, height, width = current.shape
    if flow.shape != (batch, 2, height, width):
        raise ValueError("flow must be B2HW current-to-previous pixel displacement")
    with torch.autocast(current.device.type, enabled=False):
        displacement = flow.to(device=current.device, dtype=torch.float32)
        y, x = torch.meshgrid(torch.arange(height, device=current.device, dtype=torch.float32),
                              torch.arange(width, device=current.device, dtype=torch.float32), indexing="ij")
        sx, sy = x[None] + displacement[:, 0], y[None] + displacement[:, 1]
        finite = torch.isfinite(sx) & torch.isfinite(sy)
        valid = finite & (sx >= 0) & (sx <= width - 1) & (sy >= 0) & (sy <= height - 1)
        sx, sy = sx.nan_to_num(), sy.nan_to_num()
        gx = sx * (2 / (width - 1)) - 1 if width > 1 else sx * 0
        gy = sy * (2 / (height - 1)) - 1 if height > 1 else sy * 0
        warped = F.grid_sample(previous.detach().float(), torch.stack((gx, gy), dim=-1),
                               mode="bilinear", padding_mode="zeros", align_corners=True)
        weights = valid[:, None].float()
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, None]
            if mask.shape != (batch, 1, height, width):
                raise ValueError("history mask must be B1HW or BHW")
            weights = weights * mask.detach().to(device=current.device, dtype=torch.float32).nan_to_num().clamp(0, 1)
        history = torch.where(weights > 0, warped, current.float())
    return history, weights


def forward_source(generator: nn.Module, source: Tensor, batch: Mapping | None = None,
                   previous_output: Tensor | None = None) -> Tensor:
    """Use true history only when both a previous frame and its flow exist.

    A previous output supplied by the trainer is detached. Otherwise a reset
    pass generates it without gradients; only the current frame is unrolled.
    The DDP wrapper remains on the differentiable current-frame call.
    """
    core = generator.module if hasattr(generator, "module") else generator
    from .nr import NRGenerator
    if not isinstance(core, NRGenerator) or batch is None or not all(key in batch for key in ("previous", "flow")):
        return generator(source)
    if previous_output is None:
        with torch.no_grad():
            previous_output = core(batch["previous"])
    history, mask = reproject_history(previous_output, source, batch["flow"], batch.get("mask"))
    return generator(source, previous=history, history_mask=mask)
