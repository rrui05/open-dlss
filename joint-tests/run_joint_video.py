"""Run genuine SR/RR -> existing NR -> recovered FG over all source frames."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/joint-dlss-20261028"
MEDIA_PYTHON = Path("C:/Users/Win11/anaconda3/envs/bipadelRL/python.exe")
FG_PYTHON = ROOT / "MLX-DLSS/.venv/Scripts/python.exe"
FFMPEG = Path("C:/Users/Win11/anaconda3/envs/bipadelRL/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["sr", "rr"], required=True)
    parser.add_argument("--frames", type=int, default=960)
    parser.add_argument("--out", type=Path, default=OUT / "videos")
    args = parser.parse_args()
    directory = args.out.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source_record = read(OUT / "manifest.json")["video"]
    source = Path(source_record["input"])
    stage = directory / args.kind
    stage.mkdir(parents=True, exist_ok=True)
    scaled = stage / "output.mkv"
    nr = directory / f"{args.kind}_nr"
    final = directory / f"{args.kind}_nr_fg.mp4"
    status_path = directory / f"{args.kind}_pipeline.json"
    state = {"state": "running", "kind": args.kind, "started_at": datetime.now(timezone.utc).isoformat(),
             "frames": args.frames, "source": str(source), "stages": []}
    write(status_path, state)

    def execute(name: str, command: list[str]):
        entry = {"stage": name, "state": "running", "command": command,
                 "started_at": datetime.now(timezone.utc).isoformat()}
        state["stages"].append(entry)
        write(status_path, state)
        print(f"starting {args.kind} {name}", flush=True)
        start = time.perf_counter()
        with (directory / f"{args.kind}_{name}.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        entry.update(returncode=proc.returncode, elapsed_seconds=time.perf_counter() - start,
                     state="completed" if proc.returncode == 0 else "failed")
        write(status_path, state)
        if proc.returncode:
            raise RuntimeError(f"{name} failed; see {directory / f'{args.kind}_{name}.log'}")
        print(f"completed {args.kind} {name}: {entry['elapsed_seconds']:.2f}s", flush=True)

    try:
        upscale_metadata = stage / "inference.json"
        upscale_valid = False
        if scaled.exists() and upscale_metadata.exists():
            cached = read(upscale_metadata)
            upscale_valid = (cached.get("status") == "completed" and
                             cached.get("input_frames") == args.frames and
                             cached.get("input_size") == [832, 480] and
                             cached.get("output_size") == [1664, 960] and
                             cached.get("input_sha256") == source_record["sha256"] and
                             cached.get("mode") == args.kind)
        if scaled.exists() and not upscale_valid:
            raise RuntimeError("Existing upscale artifact has incompatible metadata; preserve it and use a fresh output directory")
        if not upscale_valid:
            execute("upscale", [str(MEDIA_PYTHON), str(ROOT / "d4r/local-windows/run_video.py"),
                                 "--kind", args.kind, "--input", str(source), "--output", str(scaled),
                                 "--width", "832", "--height", "480", "--frames", str(args.frames),
                                 "--fps", "16", "--scale", "2", "--ffmpeg", str(FFMPEG)])
        nr_metadata = nr / "inference.json"
        nr_valid = False
        if nr_metadata.exists():
            cached = read(nr_metadata)
            nr_valid = (cached.get("status") == "completed" and cached.get("processedFrames") == args.frames and
                        cached.get("width") == 1664 and cached.get("height") == 960 and
                        cached.get("totalNonfinite") == 0 and not cached.get("gpuErrors"))
        if not nr_valid:
            execute("nr", ["node", str(ROOT / "open-dlss/reproductions/racing-video-2026-10-28/harness/run_video.mjs"),
                           "--input", str(scaled), "--out", str(nr), "--frames", str(args.frames),
                           "--width", "1664", "--height", "960", "--fps", "16/1",
                           "--port", "8146", "--debug-port", "9246"])
        metadata = read(nr / "inference.json")
        if metadata.get("status") != "completed" or metadata.get("totalNonfinite") != 0 or metadata.get("gpuErrors"):
            raise RuntimeError("NR video model validation failed")
        raw_frames = sorted((nr / "frames").glob("*.rgba"))
        if len(raw_frames) != args.frames or any(p.stat().st_size != 1664 * 960 * 4 for p in raw_frames):
            raise RuntimeError("NR lossless frame count or frame size is incorrect")
        fg_metadata = final.with_suffix(".json")
        fg_valid = False
        if final.exists() and fg_metadata.exists():
            cached = read(fg_metadata)
            fg_valid = (cached.get("state") == "completed" and cached.get("input_frames") == args.frames and
                        cached.get("output_frames") == args.frames * 2 and cached.get("width") == 1664 and
                        cached.get("height") == 960 and cached.get("device") == "cuda" and
                        cached.get("precision") == "reference" and cached.get("input") == str(nr / "frames"))
        if final.exists() and not fg_valid:
            raise RuntimeError("Existing FG artifact has incompatible metadata; preserve it and use a fresh output directory")
        if not fg_valid:
            execute("fg", [str(FG_PYTHON), str(ROOT / "MLX-DLSS/local/fg/run_video.py"),
                           "--input", str(nr / "frames"), "--output", str(final),
                           "--width", "1664", "--height", "960", "--frames", str(args.frames),
                           "--fps", "16", "--factor", "2", "--device", "cuda", "--precision", "reference",
                           "--ffmpeg", str(FFMPEG), "--crf", "16"])
        state.update(state="completed", finished_at=datetime.now(timezone.utc).isoformat(), output=str(final))
    except BaseException as exc:
        state.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        write(status_path, state)
        raise
    write(status_path, state)


if __name__ == "__main__":
    main()
