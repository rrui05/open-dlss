"""Prepare the same image and WebGPU output for the native OpenDLSS verifier.

The reference is this experiment's WebGPU output, not a capture of NVIDIA NGX.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
target = ROOT / "native_fixture"
target.mkdir(exist_ok=True)
source = ROOT / "inputs/street_896x512.png"
reference = ROOT / "results/opendlss_896x512/output.png"
with Image.open(source) as im:
    pixels = np.asarray(im.convert("RGBA"), dtype=np.float32) / np.float32(255)
    width, height = im.size
pixels.astype("<f4").tofile(target / "proxy.f32")
with Image.open(reference) as im:
    assert im.size == (width, height)
    np.asarray(im.convert("RGBA"), dtype=np.uint8).tofile(target / "reference_webgpu.rgba8")
manifest = {
    "sourceDimensions": [width, height], "fullDimensions": [width, height],
    "proxy": {"file": "proxy.f32", "width": width, "height": height},
    "conditioning": {"localTone": 1, "localStructure": 1, "skinStructure": -1, "style": 0},
    "seed": 0, "autoMask": True, "checks": ["output"],
    "nativeOutput": {"file": "reference_webgpu.rgba8", "dtype": "u8", "width": width, "height": height},
    "referenceProvenance": "Own previously measured original WebGPU port output; this is not an NVIDIA native output parity test.",
    "sourcePng": str(source), "sourcePngSha256": hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
    "referencePng": str(reference), "referencePngSha256": hashlib.file_digest(reference.open("rb"), "sha256").hexdigest(),
}
(target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(target)
