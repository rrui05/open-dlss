"""Build labeled views of the actual input/output without altering either file."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
source = Image.open(OUT / "input.png").convert("RGB")
result = Image.open(OUT / "output.png").convert("RGB")
runtime = json.loads((OUT / "inference.json").read_text())
browser_input = Image.open(OUT / f'{runtime.get("prefix", "single")}-input.png').convert("RGB")
assert np.array_equal(source, browser_input), "browser changed the input pixels"
assert runtime["status"] == "completed", runtime
assert runtime["learnedBlockCount"] == 71
assert runtime["head"]["nonfinite"] == 0
assert runtime["gpuErrors"] == []
assert runtime["determinism"]["verified"]
assert source.size == result.size, (source.size, result.size)
assert np.asarray(result).std() > 1, "output is effectively blank"
assert not np.array_equal(source, result), "output is unchanged"

font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = ImageFont.truetype(font_file, 23)
small = ImageFont.truetype(font_file, 17)


def pair(left, right, path, subtitle):
    assert left.size == right.size
    w, h = left.size
    canvas = Image.new("RGB", (2 * w + 12, h + 98), "#10141b")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), "INPUT: GTA San Andreas / CJ", font=font, fill="white")
    draw.text((w + 30, 12), "OUTPUT: OpenDLSS-NR", font=font, fill="white")
    canvas.paste(left, (0, 50))
    canvas.paste(right, (w + 12, 50))
    draw.text((18, h + 64), subtitle, font=small, fill="#b9c4d4")
    canvas.save(path)


pair(source, result, OUT / "comparison_full.png",
     "Original resolution 887 x 498 | original repository WebGPU network | single frame")
provenance = json.loads((OUT / "input_provenance.json").read_text())
box = tuple(provenance["face_crop_xyxy"])
# Both crops use the same exact rectangle and nearest-neighbor display enlargement.
face_input = source.crop(box).resize((690, 840), Image.Resampling.NEAREST)
face_output = result.crop(box).resize((690, 840), Image.Resampling.NEAREST)
pair(face_input, face_output, OUT / "comparison_face.png",
     "Identical face crop, enlarged 3x with nearest-neighbor | no post-enhancement")
source.crop(box).save(OUT / "face_input.png")
result.crop(box).save(OUT / "face_output.png")

a = np.asarray(source).astype(np.float64)
b = np.asarray(result).astype(np.float64)
diff = np.abs(a - b)
checks = {
    "browser_input_pixels_identical": True,
    "dimensions_match": True,
    "input_dimensions": list(source.size),
    "output_dimensions": list(result.size),
    "output_nonblank": True,
    "output_differs_from_input": True,
    "changed_pixel_fraction": float(np.mean(np.any(a != b, axis=2))),
    "mean_absolute_rgb_change_0_255": float(diff.mean()),
    "max_absolute_rgb_change_0_255": int(diff.max()),
    "ground_truth_available": False,
    "metric_note": "Changes relative to input only; these are not reconstruction-quality scores.",
    "face_crop_xyxy": list(box),
    "display_crop_enlargement": "nearest-neighbor 3x, both sides",
    "output_sha256": hashlib.sha256((OUT / "output.png").read_bytes()).hexdigest(),
    "input_sha256": hashlib.sha256((OUT / "input.png").read_bytes()).hexdigest(),
}
(OUT / "output_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps(checks, indent=2))
