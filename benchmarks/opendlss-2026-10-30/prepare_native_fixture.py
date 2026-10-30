"""Prepare a PNG proxy, optionally a safe feature-based WebGPU output fixture.

Without --features-file, writes proxy.f32 and preparation metadata only; it does
not write a proxy+RGBA8 parity fixture (that original CLI combination is unsafe).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "inputs/street_896x512.png")
    parser.add_argument("--output", type=Path, default=ROOT.parents[1] / "build/benchmark_native/input")
    parser.add_argument("--features-file", type=Path)
    parser.add_argument("--reference", type=Path, help="Own WebGPU output PNG; requires --features-file")
    parser.add_argument("--full-width", type=int)
    parser.add_argument("--full-height", type=int)
    args = parser.parse_args()
    if bool(args.reference) != bool(args.features_file):
        parser.error("--reference and --features-file must be supplied together")
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with Image.open(args.input) as image:
        width, height = image.size
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / np.float32(255)
    rgba.astype("<f4").tofile(target / "proxy.f32")
    metadata = {"input": str(args.input.resolve()), "input_sha256": digest(args.input),
                "width": width, "height": height, "proxy": str(target / "proxy.f32"),
                "conditioning": {"localTone": 1, "localStructure": 1, "skinStructure": -1, "style": 0},
                "seed": 0, "autoMask": True}
    if args.features_file:
        full_width, full_height = args.full_width or width, args.full_height or height
        expected_bytes = full_width * full_height * 16 * 4
        if args.features_file.stat().st_size != expected_bytes:
            parser.error(f"features byte count must be {expected_bytes}; supply padded --full-width/--full-height")
        values = np.fromfile(args.features_file, dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError("input features contain non-finite values")
        feature_target = target / "features.f32"
        if args.features_file.resolve() != feature_target:
            shutil.copyfile(args.features_file, feature_target)
        with Image.open(args.reference) as image:
            if image.size != (width, height):
                raise ValueError("reference PNG dimensions differ from input")
            np.asarray(image.convert("RGBA"), dtype=np.uint8).tofile(target / "reference_webgpu.rgba8")
        manifest = {
            "sourceDimensions": [width, height], "fullDimensions": [full_width, full_height],
            "inputFeatures": {"file": "features.f32"}, "checks": ["output"],
            "nativeOutput": {"file": "reference_webgpu.rgba8", "dtype": "u8", "width": width, "height": height},
            "referenceProvenance": "Own WebGPU output; not an NVIDIA capture or proof of NVIDIA parity.",
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        metadata.update(reference=str(args.reference.resolve()), reference_sha256=digest(args.reference),
                        features_sha256=digest(feature_target), safe_fixture=str(target / "manifest.json"))
    (target / "input.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
