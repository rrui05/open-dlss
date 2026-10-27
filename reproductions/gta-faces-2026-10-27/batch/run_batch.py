"""Run independent still images through the existing, unmodified OpenDLSS harness.

Every launch receives a private NR_ARTIFACTS directory. The launcher's global output
aliases therefore never overwrite the first CJ reproduction. Pillow is used only
to decode/check files and build labeled comparisons; saved model PNGs are copied.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = HERE.parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def font(size: int):
    candidate = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(candidate), size) if candidate.exists() else ImageFont.load_default()


def input_path(case: dict, project: Path) -> Path:
    value = case["input_path"]
    if value.startswith("/artifacts/"):
        return project / value.lstrip("/")
    path = Path(value)
    return path if path.is_absolute() else project / path


def face_box(case: dict, size: tuple[int, int]) -> tuple[int, int, int, int]:
    box = tuple(case["face_crop_xyxy"])
    if len(box) != 4 or any(not isinstance(v, int) for v in box):
        raise ValueError(f"{case['id']}: face_crop_xyxy must contain four integers")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        raise ValueError(f"{case['id']}: crop {box} is outside {size}")
    return box


def pair_image(left: Image.Image, right: Image.Image, title: str,
               max_width: int, max_height: int, note: str) -> Image.Image:
    tile = (max_width, max_height)
    a = ImageOps.contain(left, tile, Image.Resampling.NEAREST)
    b = ImageOps.contain(right, tile, Image.Resampling.NEAREST)
    w, h = max(a.width, b.width), max(a.height, b.height)
    canvas = Image.new("RGB", (w * 2 + 16, h + 112), "#10141b")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, font=font(21), fill="#ffffff")
    draw.text((12, 42), "INPUT", font=font(17), fill="#bfcbdc")
    draw.text((w + 28, 42), "OpenDLSS-NR", font=font(17), fill="#bfcbdc")
    canvas.paste(a, ((w - a.width) // 2, 70))
    canvas.paste(b, (w + 16 + (w - b.width) // 2, 70))
    draw.text((12, h + 84), note, font=font(13), fill="#bfcbdc")
    return canvas


def render_and_check(case: dict, destination: Path, stage: Path | None = None) -> dict:
    before = Image.open(destination / "input.png").convert("RGB")
    after = Image.open(destination / "output.png").convert("RGB")
    runtime = json.loads((destination / "inference.json").read_text())
    assert runtime["status"] == "completed", runtime
    assert runtime["learnedBlockCount"] == 71, "unexpected network block count"
    assert runtime["adapter"].get("isFallbackAdapter") is False, "hardware WebGPU adapter was not confirmed"
    assert runtime["gpuErrors"] == [], runtime["gpuErrors"]
    assert runtime["head"]["nonfinite"] == 0, runtime["head"]
    assert runtime["determinism"]["verified"], "same-input determinism was not checked"
    assert not runtime["config"]["resized"], "harness resized the input"
    assert before.size == after.size, (before.size, after.size)
    a, b = np.asarray(before), np.asarray(after)
    assert np.std(b) > 1, "output is effectively blank"
    assert not np.array_equal(a, b), "model returned the unchanged input"
    browser_equal = None
    if stage is not None:
        browser = Image.open(stage / f"more-{case['id']}-input.png").convert("RGB")
        browser_equal = bool(np.array_equal(a, browser))
        assert browser_equal, "browser-decoded input differs from original input pixels"
    previous = destination / "checks.json"
    if browser_equal is None and previous.exists():
        browser_equal = json.loads(previous.read_text()).get("browser_input_pixels_identical")
    box = face_box(case, before.size)
    left_face, right_face = before.crop(box), after.crop(box)
    left_face.save(destination / "face_input.png")
    right_face.save(destination / "face_output.png")
    pair_image(before, after, case["title"], 1000, 620,
               "Same source resolution; display fit uses nearest-neighbor sampling.").save(destination / "comparison_full.png")
    pair_image(left_face, right_face, case["title"], 600, 660,
               "Same crop on both images; nearest-neighbor display scaling only.").save(destination / "comparison_face.png")
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    checks = {
        "status": "verified",
        "dimensions": list(before.size),
        "input_sha256": digest(destination / "input.png"),
        "output_sha256": digest(destination / "output.png"),
        "browser_input_pixels_identical": browser_equal,
        "output_nonblank": True,
        "output_differs_from_input": True,
        "gpu_errors": runtime["gpuErrors"],
        "head_nonfinite": 0,
        "same_input_determinism": runtime["determinism"],
        "face_crop_xyxy": list(box),
        "changed_pixel_fraction": float(np.mean(np.any(delta != 0, axis=2))),
        "mean_absolute_rgb_change_0_255": float(delta.mean()),
        "change_metric_note": "Changes relative to the input only; not quality scores against ground truth.",
        "ground_truth_available": False,
        "native_nvidia_output_parity_verified": False,
        "scope": "Independent still frame; no temporal/video-quality claim.",
        "model_output_bytes_preserved": True,
        "display_transforms": "Matched crop and nearest-neighbor display resizing only; no enhancement.",
    }
    dump(destination / "checks.json", checks)
    dump(destination / "case.json", {k: v for k, v in case.items() if k != "input_path"})
    return checks


def contact_sheets(cases: list[dict], output: Path) -> None:
    cards = []
    for case in cases:
        directory = output / case["id"]
        before = Image.open(directory / "input.png").convert("RGB")
        after = Image.open(directory / "output.png").convert("RGB")
        box = face_box(case, before.size)
        cards.append(pair_image(before.crop(box), after.crop(box), case["title"], 220, 260,
                                "Matched face crop; no post-enhancement"))
    if not cards:
        return
    cols = min(2, len(cards))
    cell_w = max(card.width for card in cards)
    cell_h = max(card.height for card in cards)
    rows = (len(cards) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w + (cols - 1) * 20, rows * cell_h + (rows - 1) * 20), "#252c38")
    for i, card in enumerate(cards):
        sheet.paste(card, ((i % cols) * (cell_w + 20), (i // cols) * (cell_h + 20)))
    sheet.save(output / "contact_sheet_faces.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="JSON array of cases, or {cases: [...]}.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--output", type=Path, help="Default: <project>/artifacts/more")
    parser.add_argument("--weights", type=Path, help="Default: <project>/models/nr")
    parser.add_argument("--harness", type=Path, help="Default: <project>/harness/run.mjs")
    parser.add_argument("--node", default="node")
    parser.add_argument("--port", type=int, default=8110)
    parser.add_argument("--debug-port", type=int, default=9234)
    parser.add_argument("--timeout", type=int, default=600000, help="Per-case timeout in milliseconds.")
    parser.add_argument("--case", action="append", help="Run selected case id; can be repeated.")
    parser.add_argument("--render-only", action="store_true", help="Check existing results and rebuild comparisons without inference.")
    parser.add_argument("--keep-stage", action="store_true", help="Retain local launch logs and raw heads under batch/_work.")
    args = parser.parse_args()
    project = args.project.resolve()
    output = (args.output or project / "artifacts" / "more").resolve()
    harness = (args.harness or project / "harness" / "run.mjs").resolve()
    weights = (args.weights or project / "models" / "nr").resolve()
    source = json.loads(args.cases.read_text())
    cases = source["cases"] if isinstance(source, dict) else source
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["id"] in wanted]
        if {case["id"] for case in cases} != wanted:
            raise ValueError("one or more selected case ids do not exist")
    ids = [case["id"] for case in cases]
    if not ids or len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", value) for value in ids):
        raise ValueError("case ids must be unique, nonempty, and contain only lowercase letters, digits, - or _")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
               "cases": [], "model": "DLSS-NR 310.8.0.0 / original repository 71-block WebGPU network"}
    dump(output / "batch_summary.json", summary)
    for case in cases:
        directory = output / case["id"]
        directory.mkdir(parents=True, exist_ok=True)
        stage = None
        try:
            if not args.render_only:
                original = input_path(case, project).resolve()
                if original != (directory / "input.png").resolve():
                    shutil.copyfile(original, directory / "input.png")
                # A private artifacts root isolates the original launcher's aliases.
                stage = HERE / "_work" / case["id"]
                stage_input = stage / "more" / case["id"] / "input.png"
                stage_input.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(directory / "input.png", stage_input)
                env = {**os.environ, "NR_ARTIFACTS": str(stage), "NR_WEIGHTS": str(weights)}
                prefix = f"more-{case['id']}"
                command = [args.node, str(harness), "--image", f"/artifacts/more/{case['id']}/input.png",
                           "--prefix", prefix, "--repeat", "2", "--port", str(args.port),
                           "--debug-port", str(args.debug_port), "--timeout", str(args.timeout)]
                print(f"RUN {case['id']}: {case['title']}", flush=True)
                subprocess.run(command, cwd=project, env=env, check=True, timeout=args.timeout / 1000 + 60)
                shutil.copyfile(stage / f"{prefix}-enhanced.png", directory / "output.png")
                shutil.copyfile(stage / f"{prefix}-runtime.json", directory / "inference.json")
            checks = render_and_check(case, directory, stage)
            summary["cases"].append({"id": case["id"], "title": case["title"], "status": "verified",
                                     "output_sha256": checks["output_sha256"],
                                     "network_milliseconds": json.loads((directory / "inference.json").read_text())["timings"]["networkMilliseconds"]})
            if stage is not None and not args.keep_stage:
                shutil.rmtree(stage)
            dump(output / "batch_summary.json", summary)
        except Exception as error:
            summary["status"] = "failed"
            summary["cases"].append({"id": case["id"], "status": "failed", "error": str(error)})
            dump(output / "batch_summary.json", summary)
            raise
    contact_sheets(cases, output)
    summary["status"] = "completed"
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    dump(output / "batch_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
