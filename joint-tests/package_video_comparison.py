"""Package a comparison from actual completed component outputs."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/joint-dlss-20261028"
FFMPEG = Path("C:/Users/Win11/anaconda3/envs/bipadelRL/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe")


def checked(command: list[str], *, binary: bool = False):
    p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=not binary)
    if p.returncode:
        err = p.stderr.decode("utf-8", errors="replace") if binary else p.stderr
        raise RuntimeError(f"Command failed ({p.returncode}): {command!r}\n{err[-10000:]}")
    return p.stdout


def font(size: int):
    return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)


def frame(path: Path, index: int, width: int, height: int):
    raw = checked([str(FFMPEG), "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{index})",
                   "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"], binary=True)
    if len(raw) != width * height * 3:
        raise RuntimeError(f"Unexpected decoded frame size: {path}, {index}, {len(raw)}")
    return Image.frombytes("RGB", (width, height), raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.root.resolve()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    original = Path(manifest["video"]["input"])
    sr = out / "videos/sr_nr_fg.mp4"
    rr = out / "videos/rr_nr_fg.mp4"
    for path in (original, sr, rr):
        if not path.exists():
            raise FileNotFoundError(path)
    cell_w, cell_h, top = 832, 480, 44
    header = Image.new("RGB", (cell_w * 3, top), "#111a27")
    draw = ImageDraw.Draw(header)
    for col, title in enumerate(("ORIGINAL | 16 fps repeated to 32 for comparison",
                                 "SR -> NR -> FG | 2x size, 32 fps",
                                 "RR (proxy guides) -> NR -> FG | 2x, 32 fps")):
        draw.text((col * cell_w + 10, 12), title, font=font(20), fill="white")
    header_path = out / "videos/comparison-header.png"
    header.save(header_path)
    target = out / "videos/comparison.mp4"
    filters = ("[0:v]fps=32,scale=832:480:flags=lanczos,setsar=1[a];"
               "[1:v]scale=832:480:flags=lanczos,setsar=1[b];"
               "[2:v]scale=832:480:flags=lanczos,setsar=1[c];"
               "[a][b][c]hstack=inputs=3:shortest=1,pad=iw:ih+44:0:44[g];"
               "[g][3:v]overlay=0:0:shortest=1[v]")
    checked([str(FFMPEG), "-hide_banner", "-v", "warning", "-n", "-i", str(original),
             "-i", str(sr), "-i", str(rr), "-loop", "1", "-i", str(header_path),
             "-filter_complex", filters, "-map", "[v]", "-an", "-frames:v", "1920",
             "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(target)])
    samples = out / "videos/samples"
    samples.mkdir(parents=True, exist_ok=True)
    cards = []
    for index in (0, 160, 480, 800, 959):
        images = [frame(original, index, 832, 480), frame(sr, index * 2, 1664, 960),
                  frame(rr, index * 2, 1664, 960)]
        card = Image.new("RGB", (2496, 524), "#111a27")
        card.paste(header, (0, 0))
        for col, image in enumerate(images):
            card.paste(image.resize((832, 480), Image.Resampling.LANCZOS), (col * 832, 44))
        path = samples / f"comparison-{index:06d}.png"
        card.save(path)
        cards.append(card.resize((1248, 262), Image.Resampling.LANCZOS))
    sheet = Image.new("RGB", (1248, 262 * len(cards)), "#111a27")
    for row, card in enumerate(cards):
        sheet.paste(card, (0, row * 262))
    sheet.save(out / "video_contact_sheet.png")
    # Odd frames are the actual FG outputs. Show their adjacent real model frames,
    # without claiming that an unavailable half-timestep ground truth exists.
    for kind, path in (("sr", sr), ("rr", rr)):
        triplets = Image.new("RGB", (1248, 276 * 3), "#111a27")
        for row, source_index in enumerate((160, 480, 800)):
            draw = ImageDraw.Draw(triplets)
            for col, (offset, label) in enumerate(((0, "REAL"), (1, "FG GENERATED"), (2, "REAL"))):
                index = source_index * 2 + offset
                image = frame(path, index, 1664, 960).resize((416, 240), Image.Resampling.LANCZOS)
                triplets.paste(image, (col * 416, row * 276 + 36))
                draw.text((col * 416 + 8, row * 276 + 8),
                          f"{kind.upper()} chain | {index/32:.5f}s | {label}", font=font(16), fill="white")
        triplets.save(out / "videos" / f"fg_triplets_{kind}.png")
    (out / "videos/comparison.json").write_text(json.dumps({
        "inputs": [str(original), str(sr), str(rr)], "output": str(target),
        "fps": 32, "frames": 1920, "duration_seconds": 60,
        "dimensions": [2496, 524], "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "display_only": "Enhanced outputs are downscaled for comparison only; original output files retain 1664x960.",
        "original_frame_rate_conversion": "Nearest-frame repetition, not model-generated interpolation.",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"comparison": str(target), "contact_sheet": str(out / "video_contact_sheet.png")}))


if __name__ == "__main__":
    main()
