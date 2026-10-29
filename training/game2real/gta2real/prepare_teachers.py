"""Download explicit pretrained teachers and export a portable local configuration.

This command writes model weights and public model configuration only. It does
not copy Hugging Face credentials, environment variables, or cache metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import torch
import yaml

from .perceptual import DinoStructure, FrozenVGG16, SemanticTeacher


def _require_cached_vgg() -> None:
    """Prevent torchvision's implicit download when offline operation is requested."""
    try:
        from torchvision.models import VGG16_Weights
    except (ImportError, RuntimeError) as error:
        raise RuntimeError("preparing ImageNet VGG16 requires a working torchvision installation") from error
    name = Path(urlparse(VGG16_Weights.IMAGENET1K_V1.url).path).name
    cached = Path(torch.hub.get_dir()) / "checkpoints" / name
    if not cached.is_file():
        raise FileNotFoundError(
            f"--local-files-only forbids downloading the missing ImageNet VGG16 checkpoint: {cached}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("weights"), help="local teacher output directory")
    parser.add_argument("--dino-model", default="facebook/dinov2-base")
    parser.add_argument("--semantic-model", default="nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    parser.add_argument("--dino-revision", default=None, help="optional immutable Hugging Face revision")
    parser.add_argument("--semantic-revision", default=None, help="optional immutable Hugging Face revision")
    parser.add_argument("--local-files-only", action="store_true", help="require existing caches; do not download")
    return parser


def prepare(args: argparse.Namespace) -> Path:
    output = args.out.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.local_files_only:
        _require_cached_vgg()

    print("Preparing pretrained ImageNet VGG16 features on CPU...")
    vgg = FrozenVGG16(weights="imagenet")
    vgg_path = output / "vgg16_features.pt"
    temporary = vgg_path.with_suffix(".pt.tmp")
    torch.save(vgg.features.state_dict(), temporary)
    temporary.replace(vgg_path)
    del vgg

    print(f"Preparing DINOv2: {args.dino_model}")
    dino = DinoStructure(args.dino_model, local_files_only=args.local_files_only, revision=args.dino_revision)
    dino_path = output / "dinov2"
    dino.model.save_pretrained(dino_path, safe_serialization=True)
    del dino

    print(f"Preparing Cityscapes semantic teacher: {args.semantic_model}")
    semantic = SemanticTeacher(
        args.semantic_model, local_files_only=args.local_files_only, revision=args.semantic_revision
    )
    semantic_path = output / "segformer_cityscapes"
    semantic.model.save_pretrained(semantic_path, safe_serialization=True)
    del semantic

    # Publish the usable configuration only after every teacher was saved.
    config = {"teachers": {
        "vgg_weights": str(vgg_path),
        "dino_model": str(dino_path),
        "semantic_model": str(semantic_path),
        "local_files_only": True,
    }}
    config_path = output / "teachers.local.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    provenance = {
        "status": "completed",
        "vgg": {"source": "torchvision VGG16_Weights.IMAGENET1K_V1", "saved": str(vgg_path)},
        "dino": {"source": args.dino_model, "revision": args.dino_revision, "saved": str(dino_path)},
        "semantic": {"source": args.semantic_model, "revision": args.semantic_revision, "saved": str(semantic_path)},
        "configuration": str(config_path),
        "note": "Model weights and public architecture configuration only; no account credentials are exported.",
    }
    (output / "teachers.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Saved local teacher configuration: {config_path}")
    return config_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        prepare(args)
    except Exception as error:
        parser.exit(1, f"Teacher preparation failed; no random substitute was used: {error}\n")


if __name__ == "__main__":
    main()
