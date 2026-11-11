"""Prepare reproducible inputs for the joint DLSS component experiment."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "joint-dlss-20261028"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cj = ROOT / "open-dlss/reproductions/gta-cj-2026-10-27/artifacts"
    base = ROOT / "open-dlss/reproductions/gta-faces-2026-10-27"
    cases = [{"id": "cj", "title": "CJ / San Andreas", "full": cj / "input.png",
              "face": cj / "face_input.png", "box": [360, 12, 590, 292]}]
    for entry in json.loads((base / "batch/cases.json").read_text(encoding="utf-8")):
        folder = base / "artifacts/more" / entry["id"]
        cases.append({"id": entry["id"], "title": entry["title"],
                      "full": folder / "input.png", "face": folder / "face_input.png",
                      "box": entry["face_crop_xyxy"]})
    records = []
    for case in cases:
        directory = OUT / "images" / case["id"]
        directory.mkdir(parents=True, exist_ok=True)
        image = Image.open(case["face"]).convert("RGB")
        original_size = list(image.size)
        # Bound the doubled NR field on the actual 8 GiB GPU. Keep all transforms explicit.
        if max(image.size) > 512:
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        target = directory / "input.png"
        if target.exists():
            prior = Image.open(target).convert("RGB")
            if prior.size != image.size or prior.tobytes() != image.tobytes():
                raise RuntimeError(f"Refusing to replace a different prepared input: {target}")
        else:
            image.save(target)
        records.append({
            "id": case["id"], "title": case["title"], "input": str(target),
            "source_full": str(case["full"]), "source_full_sha256": digest(case["full"]),
            "source_face": str(case["face"]), "source_face_sha256": digest(case["face"]),
            "source_face_box_xyxy": case["box"], "source_face_size": original_size,
            "input_size": list(image.size), "input_sha256": digest(target),
            "resized": original_size != list(image.size), "max_input_edge": 512,
            "scope": "Previously tested face crop; independently initialized models on this crop.",
            "variants": ["nr", "sr", "sr_nr", "rr", "rr_nr"],
        })
    video = ROOT / "outputs/racing-dlss-20261028/original.mp4"
    if digest(video) != "553cfb345b842bd90ef50bf9a69200335f0bfbd39078cff43fed1bbe0acd6365":
        raise RuntimeError("The one-minute source video differs from the previous test.")
    manifest = {
        "schema": 1, "prepared_at": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(OUT), "images": records,
        "video": {"input": str(video), "sha256": digest(video), "width": 832, "height": 480,
                  "fps": "16/1", "frames": 960, "duration_seconds": 60, "audio_streams": 0},
        "hardware": {"gpu": "NVIDIA GeForce RTX 5060", "vram_mib": 8151, "driver": "610.88"},
        "pipelines": ["SR -> NR -> FG", "RR (includes super resolution) -> NR -> FG"],
        "nr": {"backend": "Original OpenDLSS-NR WebGPU / Windows Edge", "blocks": 71,
               "seed": 0, "style": 0, "tone": 1, "structure": 1, "skin": -1,
               "auto_mask": True, "temporal_feedback": False},
        "limitations": [
            "Still-image FG is not applicable; frame generation is tested only on frame pairs and video.",
            "RR media inputs lack engine G-buffers and raw ray samples. Flat proxy guides are an experiment, not native ray-tracing validation.",
            "d4r native AMD backend requires wave64; current NVIDIA hardware reports subgroup size 32 only. SR/RR use an explicitly recorded official runtime fallback if necessary.",
            "Face crops change model context relative to the earlier full-frame NR runs. One crop is bounded to 512 pixels for GPU memory.",
            "No high-resolution or path-traced ground truth is available. Input/output differences are not quality scores.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(OUT / "manifest.json"), "images": len(records),
                      "sizes": {r["id"]: r["input_size"] for r in records}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
