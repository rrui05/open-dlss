"""Evaluate matched source/output content and optional unpaired distribution metrics.

No real image is treated as the pixel-aligned target of a game frame. FID/KID
describe set-level appearance distributions and cannot establish content fidelity.
Disabled perceptual teachers are explicitly reported as not measured.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.nn import functional as F

from .data import IMAGE_SUFFIXES
from .losses import dino_structure_loss, edge_loss, low_frequency_loss
from .perceptual import DinoStructure, SemanticTeacher


DEFAULT_DINO = "facebook/dinov2-base"
DEFAULT_SEMANTIC = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"image directory not found: {directory}")
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _image(path: Path, device: torch.device) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device) / 255


def _resize(image: Tensor, size: int) -> Tensor:
    return F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False, antialias=True)


def _number(value: Tensor | float) -> float:
    result = float(value.detach().cpu()) if isinstance(value, Tensor) else float(value)
    if not math.isfinite(result):
        raise ValueError("a metric produced a nonfinite value")
    return result


def _disabled(name: str | None) -> bool:
    return name is None or name.lower() in ("none", "off", "disabled", "")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="source image directory, searched recursively")
    parser.add_argument("--generated", type=Path, required=True, help="inference output directory with matching relative PNG stems")
    parser.add_argument("--real", type=Path, default=None, help="unpaired real-image directory; required for --fid")
    parser.add_argument("--output", type=Path, default=Path("report.json"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--limit", type=int, default=0, help="maximum images per domain; zero evaluates all sorted images")
    parser.add_argument("--teachers-config", type=Path, default=None, help="optional teachers.local.yaml from prepare_teachers")
    parser.add_argument("--dino-model", default=None, help=f"model/path, or none to disable; default {DEFAULT_DINO}")
    parser.add_argument("--semantic-model", default=None, help="model/path, or none to disable; disabled by default")
    parser.add_argument("--semantic", action="store_true", help="enable the default Cityscapes SegFormer")
    parser.add_argument("--dino-revision", default=None)
    parser.add_argument("--semantic-revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dino-size", type=int, default=392)
    parser.add_argument("--semantic-size", type=int, default=512)
    parser.add_argument("--semantic-confidence", type=float, default=0.6)
    parser.add_argument("--low-frequency-kernel", type=int, default=32)
    parser.add_argument("--fid", action="store_true", help="also measure FID and KID; may download standard Inception weights")
    parser.add_argument("--kid-subsets", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0, help="seed for KID subset sampling")
    return parser


def _teacher_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if args.teachers_config is not None:
        import yaml

        document = yaml.safe_load(args.teachers_config.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not isinstance(document.get("teachers"), dict):
            raise ValueError("--teachers-config must contain a teachers mapping")
        options = dict(document["teachers"])
    options["dino_model"] = args.dino_model if args.dino_model is not None else options.get("dino_model", DEFAULT_DINO)
    default_semantic = DEFAULT_SEMANTIC if args.semantic else "none"
    options["semantic_model"] = (
        args.semantic_model if args.semantic_model is not None else options.get("semantic_model", default_semantic)
    )
    options["dino_revision"] = args.dino_revision or options.get("dino_revision")
    options["semantic_revision"] = args.semantic_revision or options.get("semantic_revision")
    options["local_files_only"] = bool(args.local_files_only or options.get("local_files_only", False))
    return options


def _pairs(source_root: Path, generated_root: Path, limit: int) -> list[tuple[Path, Path]]:
    sources = _images(source_root)
    if limit:
        sources = sources[:limit]
    if not sources:
        raise ValueError("no source images were found")
    if not generated_root.is_dir():
        raise NotADirectoryError(f"generated image directory not found: {generated_root}")
    result: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    missing: list[Path] = []
    for source in sources:
        relative = source.relative_to(source_root).with_suffix(".png")
        generated = generated_root / relative
        if generated in seen:
            raise ValueError(f"multiple source images map to the same generated PNG: {relative}")
        seen.add(generated)
        if not generated.is_file():
            missing.append(relative)
        result.append((source, generated))
    if missing:
        examples = ", ".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"{len(missing)} generated PNGs are missing: {examples}")
    return result


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit < 0 or min(args.dino_size, args.semantic_size, args.low_frequency_kernel, args.kid_subsets) < 1:
        raise ValueError("limit must be nonnegative and metric sizes/subset count must be positive")
    if not 0 <= args.semantic_confidence <= 1:
        raise ValueError("semantic confidence must be in [0, 1]")
    source_root = args.source.expanduser().resolve()
    generated_root = args.generated.expanduser().resolve()
    pairs = _pairs(source_root, generated_root, args.limit)
    real_root = args.real.expanduser().resolve() if args.real is not None else None
    reals = _images(real_root) if real_root is not None else []
    if args.limit:
        reals = reals[:args.limit]
    if args.fid and real_root is None:
        raise ValueError("--fid requires --real with an unpaired real-image directory")
    if args.fid and not reals:
        raise ValueError("the real-image directory contains no images")

    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.manual_seed(args.seed)
    options = _teacher_options(args)
    teacher_report: dict[str, Any] = {}
    dino = None
    if not _disabled(options["dino_model"]):
        dino = DinoStructure(options["dino_model"], local_files_only=options["local_files_only"],
                             revision=options["dino_revision"]).to(device)
        teacher_report["dino"] = {"status": "measured", "model": options["dino_model"],
                                  "revision": options["dino_revision"], "metric_image_size": [args.dino_size] * 2}
    else:
        teacher_report["dino"] = {"status": "not_measured", "reason": "explicitly disabled; no random features substituted"}
    semantic = None
    if not _disabled(options["semantic_model"]):
        semantic = SemanticTeacher(options["semantic_model"], local_files_only=options["local_files_only"],
                                   revision=options["semantic_revision"]).to(device)
        teacher_report["semantic"] = {"status": "measured", "model": options["semantic_model"],
                                      "revision": options["semantic_revision"],
                                      "confidence_threshold": args.semantic_confidence,
                                      "metric_image_size": [args.semantic_size] * 2}
    else:
        teacher_report["semantic"] = {"status": "not_measured", "reason": "not requested; no random semantic model substituted"}

    distribution: dict[str, Any] = {"status": "not_measured", "reason": "--fid was not requested"}
    fid = kid = None
    if args.fid:
        if min(len(reals), len(pairs)) < 2:
            distribution = {"status": "not_measured", "reason": "FID and unbiased KID require at least two images per domain",
                            "real_images": len(reals), "generated_images": len(pairs)}
        else:
            try:
                from torchmetrics.image.fid import FrechetInceptionDistance
                from torchmetrics.image.kid import KernelInceptionDistance
            except ImportError as error:
                raise RuntimeError("--fid requires torchmetrics[image] and torch-fidelity; metrics were not substituted") from error
            subset_size = min(50, len(reals), len(pairs))
            fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            kid = KernelInceptionDistance(subset_size=subset_size, subsets=args.kid_subsets, normalize=True).to(device)
            distribution = {"status": "pending", "real_images": len(reals), "generated_images": len(pairs),
                            "kid_subset_size": subset_size, "kid_subsets": args.kid_subsets,
                            "feature_dimension": 2048, "seed": args.seed,
                            "preprocessing": "one original-resolution RGB float [0,1] image per update; standard Inception preprocessing inside torchmetrics"}

    per_image: list[dict[str, Any]] = []
    semantic_pixels = semantic_matches = semantic_total_pixels = 0
    for index, (source_path, generated_path) in enumerate(pairs):
        source, generated = _image(source_path, device), _image(generated_path, device)
        if source.shape != generated.shape:
            raise ValueError(f"generated dimensions differ from source for {source_path.relative_to(source_root)}: "
                             f"{tuple(source.shape[-2:])} versus {tuple(generated.shape[-2:])}")
        clipped = _number(((generated <= 0) | (generated >= 1)).float().mean())
        row: dict[str, Any] = {
            "source": source_path.relative_to(source_root).as_posix(),
            "generated": generated_path.relative_to(generated_root).as_posix(),
            "dimensions_wh": [source.shape[-1], source.shape[-2]],
            "edge_loss": _number(edge_loss(generated, source)),
            "low_frequency_loss": _number(low_frequency_loss(generated, source, args.low_frequency_kernel)),
            "mean_absolute_rgb_change": _number((generated - source).abs().mean()),
            "generated_boundary_channel_fraction": clipped,
            "clip_fraction": clipped,
            "generated_rgb_std": _number(generated.std(unbiased=False)),
            "dino_spatial_cosine_distance": None,
            "dino_global_cosine_distance": None,
            "semantic_pixel_agreement": None,
        }
        if dino is not None:
            source_features = dino(_resize(source, args.dino_size))
            generated_features = dino(_resize(generated, args.dino_size))
            row["dino_spatial_cosine_distance"] = _number(dino_structure_loss(generated_features, source_features))
            source_global = source_features.mean(dim=(-2, -1))
            generated_global = generated_features.mean(dim=(-2, -1))
            row["dino_global_cosine_distance"] = _number(
                (1 - F.cosine_similarity(generated_global, source_global, dim=1)).clamp_min(0).mean()
            )
        if semantic is not None:
            size = (args.semantic_size, args.semantic_size)
            source_logits = F.interpolate(semantic(_resize(source, args.semantic_size)), size=size, mode="bilinear", align_corners=False)
            generated_logits = F.interpolate(semantic(_resize(generated, args.semantic_size)), size=size, mode="bilinear", align_corners=False)
            probability, source_labels = source_logits.softmax(dim=1).max(dim=1)
            valid = probability >= args.semantic_confidence
            valid_count = int(valid.sum())
            correct_count = int(((generated_logits.argmax(dim=1) == source_labels) & valid).sum())
            semantic_pixels += valid_count
            semantic_matches += correct_count
            semantic_total_pixels += valid.numel()
            row["semantic_valid_pixels"] = valid_count
            row["semantic_matching_pixels"] = correct_count
            row["semantic_confident_fraction"] = valid_count / valid.numel()
            row["semantic_pixel_agreement"] = correct_count / valid_count if valid_count else None
        if fid is not None and kid is not None:
            fid.update(generated, real=False)
            kid.update(generated, real=False)
        per_image.append(row)
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(pairs):
            print(f"Evaluated {index + 1}/{len(pairs)} matched source/output images")

    if fid is not None and kid is not None:
        for real_path in reals:
            real = _image(real_path, device)
            fid.update(real, real=True)
            kid.update(real, real=True)
        kid_mean, kid_std = kid.compute()
        distribution.update(status="measured", fid=_number(fid.compute()), kid_mean=_number(kid_mean), kid_std=_number(kid_std))

    metric_names = ("edge_loss", "low_frequency_loss", "mean_absolute_rgb_change",
                    "generated_boundary_channel_fraction", "clip_fraction", "generated_rgb_std",
                    "dino_spatial_cosine_distance", "dino_global_cosine_distance")
    metrics: dict[str, Any] = {}
    for name in metric_names:
        values = [row[name] for row in per_image if row[name] is not None]
        metrics[name] = sum(values) / len(values) if values else None
    metrics["semantic_pixel_agreement"] = semantic_matches / semantic_pixels if semantic_pixels else None
    metrics["semantic_confident_fraction"] = semantic_pixels / semantic_total_pixels if semantic_total_pixels else None
    metrics["semantic_valid_pixels"] = semantic_pixels if semantic is not None else None
    if semantic is not None and not semantic_pixels:
        teacher_report["semantic"]["status"] = "not_measured"
        teacher_report["semantic"]["reason"] = "no source pixel met the confidence threshold"

    limitations = [
        "DINO, semantic agreement, edges and coarse tone measure source/output consistency; they do not establish photorealism.",
        "FID/KID compare unpaired image distributions; changes in viewpoint, scene composition or dataset sampling can alter them without better rendering.",
        "No paired-real PSNR/SSIM is computed because real photographs are not pixel-aligned targets for these game frames.",
        "Boundary/clip fraction counts decoded channels exactly at 0 or 1; PNGs cannot reveal whether pre-save network values were clipped.",
        "DINO and semantic inputs are identically resized for both branches only for their metrics; saved images are unchanged.",
        "Semantic agreement uses teacher predictions rather than ground-truth labels and can share the teacher's domain errors.",
        "This report has no optical-flow temporal consistency measurement or human realism study.",
        "Finite-sample FID/KID can be unstable; small image sets, especially far fewer than 2048 images, need cautious interpretation.",
    ]
    if dino is None:
        limitations.append("DINO content preservation was not measured because the teacher was explicitly disabled.")
    return {
        "status": "completed", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source_root), "generated_directory": str(generated_root),
        "real_directory": str(real_root) if real_root is not None else None,
        "device": str(device), "counts": {"matched_source_generated": len(pairs), "real": len(reals)},
        "method": {
            "pairing": "source-relative path with its extension replaced by .png; missing outputs and dimension changes are errors",
            "selection": "sorted recursive paths; --limit applies independently to source/generated pairs and real images",
            "content": "frozen DINOv2 patch-wise channel cosine distance and cosine distance between spatially pooled descriptors",
            "edges": "luminance Sobel gradient and direction consistency on original-resolution source/output pairs",
            "low_frequency": f"Smooth-L1 of {args.low_frequency_kernel}-pixel block averages at original resolution",
            "semantic": "confidence-gated source-teacher argmax agreement at the identically resized semantic view",
            "aggregation": "image means except semantic agreement, which is weighted by confident pixel count",
        },
        "teachers": teacher_report, "metrics": metrics, "distribution": distribution,
        "limitations": limitations, "per_image": per_image,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = evaluate(args)
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved evaluation report: {output}")
    except Exception as error:
        parser.exit(1, f"Evaluation failed; unavailable teachers or metrics were not substituted: {error}\n")


if __name__ == "__main__":
    main()
