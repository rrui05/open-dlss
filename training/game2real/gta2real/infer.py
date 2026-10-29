"""Run RGB enhancement with checkpoints produced by this project's trainer.

Example: python -m gta2real.infer --checkpoint runs/best.pt --input images --output enhanced
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import nullcontext
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn

from .models import GeneratorConfig, TransformerUNet


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    final = length - tile
    if starts[-1] != final:
        starts.append(final)
    return starts


def _model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return fallback if buffer is None else buffer.device


def _predict(model: nn.Module, patch: Tensor, device: torch.device) -> Tensor:
    prediction = model(patch.to(device=device, dtype=torch.float32))
    if not isinstance(prediction, Tensor) or prediction.shape != patch.shape:
        raise ValueError("generator must return a tensor with the input Bx3xHxW shape")
    prediction = prediction.to(device=patch.device, dtype=torch.float32)
    if not torch.isfinite(prediction).all():
        raise RuntimeError("generator returned nonfinite RGB values")
    return prediction


@torch.inference_mode()
def infer_tensor(model: nn.Module, x: Tensor, tile: int = 1024, overlap: int = 128) -> Tensor:
    """Enhance Bx3xHxW RGB without resizing, using optional Hann-blended tiles.

    The caller sets model.eval() and owns any CUDA autocast context. Computation
    follows the model's device, while the float32 result and tile accumulator
    remain on x.device. A CPU input therefore needs only tile-sized GPU memory.
    tile <= 0 disables tiling. All Hann weights are strictly positive.
    """
    if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[0], *x.shape[-2:]) < 1:
        raise ValueError("expected nonempty Bx3xHxW RGB input")
    if not x.is_floating_point():
        raise TypeError("RGB input must be floating point in [0, 1]")
    if not isinstance(tile, int) or not isinstance(overlap, int):
        raise TypeError("tile and overlap must be integers")
    if overlap < 0 or (tile > 0 and overlap >= tile):
        raise ValueError("overlap must be nonnegative and smaller than a positive tile size")
    if not torch.isfinite(x).all():
        raise ValueError("input RGB contains nonfinite values")

    device = _model_device(model, x.device)
    height, width = x.shape[-2:]
    if tile <= 0 or max(height, width) <= tile:
        return _predict(model, x, device).clamp(0, 1)

    source = x.float()
    residual_sum = torch.zeros_like(source)
    weight_sum = torch.zeros((1, 1, height, width), device=x.device, dtype=torch.float32)
    windows: dict[tuple[int, int], Tensor] = {}
    for row in _tile_starts(height, tile, overlap):
        for column in _tile_starts(width, tile, overlap):
            patch = source[:, :, row:row + tile, column:column + tile]
            prediction = _predict(model, patch, device)
            patch_height, patch_width = patch.shape[-2:]
            key = (patch_height, patch_width)
            if key not in windows:
                vertical = torch.hann_window(patch_height, periodic=False, device=x.device, dtype=torch.float32)
                horizontal = torch.hann_window(patch_width, periodic=False, device=x.device, dtype=torch.float32)
                # Finite edge weights cover the outer image border and single-pixel tiles.
                windows[key] = (vertical[:, None] * horizontal[None, :]).clamp_min(1e-3)[None, None]
            weight = windows[key]
            # Averaging residuals is algebraically equivalent to averaging RGB.
            # It also preserves an identity generator exactly, without rounding drift.
            residual_sum[:, :, row:row + patch_height, column:column + patch_width].add_(
                (prediction - patch) * weight)
            weight_sum[:, :, row:row + patch_height, column:column + patch_width].add_(weight)
    if not (weight_sum > 0).all():
        raise RuntimeError("tile layout left uncovered pixels")
    return (source + residual_sum / weight_sum).clamp(0, 1)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    if value not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(value)


def load_generator(checkpoint: Path | str, device: torch.device | str,
                   use_ema: bool = True) -> tuple[TransformerUNet, dict]:
    """Strictly load the trainer's configuration and chosen generator state."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    # Project checkpoints also contain configuration and optimizer metadata.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain the project's training dictionary")
    configuration = payload.get("config")
    if not isinstance(configuration, Mapping) or not isinstance(configuration.get("model"), Mapping):
        raise ValueError("checkpoint is missing config.model (GeneratorConfig settings)")
    settings = dict(configuration["model"])
    for name in ("depths", "heads"):
        if name in settings:
            settings[name] = tuple(settings[name])
    model = TransformerUNet(GeneratorConfig(**settings))
    state_name = "ema" if use_ema else "generator"
    state = payload.get(state_name)
    if not isinstance(state, Mapping) or not state:
        hint = "; pass --no-ema to explicitly use generator weights" if use_ema else ""
        raise ValueError(f"checkpoint is missing a nonempty {state_name} state dictionary{hint}")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    return model, {"step": payload.get("step"), "weights": state_name}


def _input_outputs(source: Path, destination: Path) -> list[tuple[Path, Path]]:
    if not source.exists():
        raise FileNotFoundError(f"input path does not exist: {source}")
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported input image extension: {source}")
        if destination.suffix.lower() == ".png" and not destination.is_dir():
            pairs = [(source, destination)]
        else:
            if destination.exists() and not destination.is_dir():
                raise ValueError("output must be a directory or a .png file")
            pairs = [(source, destination / f"{source.stem}.png")]
    elif source.is_dir():
        if destination.exists() and not destination.is_dir():
            raise ValueError("directory input requires an output directory")
        inputs = sorted(path for path in source.rglob("*")
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not inputs:
            raise ValueError(f"no supported images found recursively under: {source}")
        pairs = [(path, destination / path.relative_to(source).with_suffix(".png")) for path in inputs]
    else:
        raise ValueError(f"input is not a file or directory: {source}")
    seen = set()
    for original, output in pairs:
        resolved = output.resolve()
        if resolved == original.resolve():
            raise ValueError(f"output would overwrite an input image: {original}")
        if resolved in seen:
            raise ValueError(f"multiple input names map to the same output PNG: {output}")
        seen.add(resolved)
    return pairs


def _read_rgb(path: Path) -> Tensor:
    try:
        with Image.open(path) as image:
            pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    except (OSError, ValueError) as error:
        raise ValueError(f"failed to decode image {path}: {error}") from error
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)


def _save_rgb(tensor: Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = tensor[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(pixels).save(path, format="PNG")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="local checkpoint created by this project's trainer")
    parser.add_argument("--input", type=Path, required=True, help="one image or an image directory searched recursively")
    parser.add_argument("--output", type=Path, required=True, help="output directory, or .png path for one input image")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tile", type=int, default=1024, help="tile side in pixels; <=0 runs the full image")
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--strength", type=float, default=1.0, help="original/output blend in [0,1]")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--no-ema", action="store_true", help="explicitly select generator instead of EMA weights")
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.strength) or not 0 <= args.strength <= 1:
            raise ValueError("strength must be finite and in [0, 1]")
        if args.overlap < 0 or (args.tile > 0 and args.overlap >= args.tile):
            raise ValueError("overlap must be nonnegative and smaller than a positive tile size")
        pairs = _input_outputs(args.input.resolve(), args.output.resolve())
        device = resolve_device(args.device)
        precision = args.precision if device.type == "cuda" else "fp32"
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("this CUDA device does not support bf16; select --precision fp16 or fp32")
        model, metadata = load_generator(args.checkpoint.resolve(), device, use_ema=not args.no_ema)
        print(f"checkpoint={args.checkpoint} weights={metadata['weights']} step={metadata['step']} "
              f"device={device} precision={precision} images={len(pairs)}", flush=True)
        for position, (source, destination) in enumerate(pairs, 1):
            original = _read_rgb(source)
            print(f"[{position}/{len(pairs)}] {source} ({original.shape[-1]}x{original.shape[-2]})", flush=True)
            context = (torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
                       if device.type == "cuda" and precision != "fp32" else nullcontext())
            with context:
                enhanced = infer_tensor(model, original, args.tile, args.overlap)
            blended = original + args.strength * (enhanced - original)
            _save_rgb(blended, destination)
            print(f"saved {destination}", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
