"""Package and verify actual OpenDLSS-NR video frames without changing them.

Run after run_video.mjs exits successfully. The source and model output remain
at their recorded resolution. Labels are added only to comparison artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont


DEFAULT_OUT = Path("D:/newDLSS/outputs/racing-dlss-20261028")
DEFAULT_INPUT = Path("C:/Users/Win11/Desktop/racing-final704526.mp4")
DEFAULT_FFMPEG = Path(
    "C:/Users/Win11/anaconda3/envs/bipadelRL/Lib/site-packages/"
    "imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(command: list[str], *, cwd: Path | None = None, binary: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=not binary)
    if result.returncode:
        error = result.stderr.decode("utf-8", errors="replace") if binary else result.stderr
        raise RuntimeError(f"command failed ({result.returncode}): {command!r}\n{error[-12000:]}")
    return result


def wsl_path(path: Path) -> str:
    value = str(path.resolve())
    if len(value) > 2 and value[1] == ":":
        return f"/mnt/{value[0].lower()}/{value[2:].replace(chr(92), '/').lstrip('/')}"
    return value


def probe(path: Path) -> dict:
    command = (["wsl.exe", "-e", "ffprobe"] if os.name == "nt" else ["ffprobe"])
    result = run(command + [
        "-v", "error", "-count_frames", "-show_entries",
        "stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,"
        "nb_frames,nb_read_frames,duration,pix_fmt:format=duration",
        "-of", "json", wsl_path(path) if os.name == "nt" else str(path.resolve()),
    ])
    data = json.loads(result.stdout)
    data["method"] = "ffprobe -count_frames (WSL on Windows)"
    return data


def verify_video(data: dict, width: int, height: int, count: int,
                 fps: Fraction, name: str) -> dict:
    videos = [stream for stream in data["streams"] if stream["codec_type"] == "video"]
    require(len(videos) == 1, f"{name}: expected exactly one video stream")
    require(not any(stream["codec_type"] == "audio" for stream in data["streams"]),
            f"{name}: unexpected audio stream")
    stream = videos[0]
    require((stream["width"], stream["height"]) == (width, height),
            f"{name}: unexpected dimensions {stream['width']}x{stream['height']}")
    decoded = int(stream.get("nb_read_frames", stream.get("nb_frames", -1)))
    require(decoded == count, f"{name}: decoded {decoded} frames, expected {count}")
    actual_fps = Fraction(stream["avg_frame_rate"])
    require(actual_fps == fps, f"{name}: frame rate {actual_fps}, expected {fps}")
    expected_duration = float(Fraction(count, 1) / fps)
    duration = float(stream.get("duration", data["format"]["duration"]))
    require(abs(duration - expected_duration) < 1 / float(fps),
            f"{name}: duration {duration}, expected {expected_duration}")
    require(stream["codec_name"] == "h264", f"{name}: expected H.264 encoding")
    require(stream.get("pix_fmt") == "yuv420p", f"{name}: expected yuv420p")
    return {"verified": True, "decodedFrames": decoded, "dimensions": [width, height],
            "frameRate": str(fps), "durationSeconds": duration,
            "codec": stream["codec_name"], "pixelFormat": stream["pix_fmt"],
            "audioStreams": 0, "method": data["method"]}


def font_path() -> Path | None:
    return next((path for path in [Path("C:/Windows/Fonts/arial.ttf"),
                                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
                 if path.exists()), None)


def image_font(size: int) -> ImageFont.ImageFont:
    path = font_path()
    return ImageFont.truetype(str(path), size) if path else ImageFont.load_default()


def raw_frame(path: Path, width: int, height: int) -> Image.Image:
    data = path.read_bytes()
    require(len(data) == width * height * 4, f"{path}: invalid RGBA byte count")
    return Image.frombytes("RGBA", (width, height), data).convert("RGB")


def input_sample(source: Path, out: Path, ffmpeg: Path, index: int,
                 width: int, height: int) -> Path:
    path = out / "samples" / f"input-{index:06d}.rgba"
    if not path.exists():
        decoded = run([
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-vf", f"select=eq(n\\,{index})", "-frames:v", "1", "-an",
            "-fps_mode", "passthrough", "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1",
        ], binary=True).stdout
        require(len(decoded) == width * height * 4, f"source frame {index}: decode size mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(decoded)
    return path


def comparisons(source: Path, out: Path, ffmpeg: Path, entries: list[dict],
                width: int, height: int, fps: Fraction) -> list[str]:
    by_index = {entry["index"]: entry for entry in entries}
    first, last = entries[0]["index"], entries[-1]["index"]
    selected = list(dict.fromkeys(index for index in [first, 160, 480, 800, last]
                                  if index in by_index))
    cards = []
    names = []
    for index in selected:
        original = input_sample(source, out, ffmpeg, index, width, height)
        require(sha256(original) == by_index[index]["inputSha256"],
                f"frame {index}: saved input does not match inference input hash")
        left = raw_frame(original, width, height)
        right = raw_frame(out / "frames" / f"{index:06d}.rgba", width, height)
        card = Image.new("RGB", (width * 2, height + 40), "#111820")
        card.paste(left, (0, 40))
        card.paste(right, (width, 40))
        draw = ImageDraw.Draw(card)
        stamp = f"{float(Fraction(index, 1) / fps):.2f}s | frame {index}"
        draw.text((12, 8), f"ORIGINAL | {stamp}", fill="white", font=image_font(20))
        draw.text((width + 12, 8), "OpenDLSS-NR | same resolution", fill="white", font=image_font(20))
        name = f"comparison-{index:06d}.png"
        card.save(out / name)
        names.append(name)
        cards.append(card)
    cell_width = 832
    cell_height = round((height + 40) * cell_width / (width * 2))
    columns = min(2, len(cards))
    rows = (len(cards) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#111820")
    for position, card in enumerate(cards):
        thumbnail = card.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
        sheet.paste(thumbnail, ((position % columns) * cell_width, (position // columns) * cell_height))
    sheet.save(out / "contact_sheet.png")
    return names + ["contact_sheet.png"]


def encode_comparison(source: Path, output: Path, destination: Path, ffmpeg: Path,
                      width: int, height: int, start: int, count: int, fps: Fraction) -> None:
    # Use the font directory as cwd, so fontfile has no Windows drive colon to
    # escape through both the filtergraph parser and drawtext's option parser.
    font = font_path()
    font_option = f"fontfile='{font.name}':" if font else ""
    label = font_option + "fontcolor=white:fontsize=22:x=12:y=8"
    filters = (
        f"[0:v]trim=start_frame={start}:end_frame={start + count},setpts=PTS-STARTPTS,"
        f"pad={width}:{height + 40}:0:40:color=0x111820,"
        f"drawtext={label}:text='ORIGINAL'[left];"
        f"[1:v]setpts=PTS-STARTPTS,pad={width}:{height + 40}:0:40:color=0x111820,"
        f"drawtext={label}:text='OpenDLSS-NR - same resolution'[right];"
        "[left][right]hstack=inputs=2:shortest=1[v]"
    )
    temporary = destination.with_name(destination.stem + ".partial.mp4")
    run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(source), "-i", str(output), "-filter_complex", filters,
         "-map", "[v]", "-an", "-frames:v", str(count), "-r", str(fps),
         "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(temporary)], cwd=font.parent if font else None)
    temporary.replace(destination)


def write_review(out: Path, width: int, height: int, count: int,
                 fps: Fraction, images: list[str]) -> None:
    duration = float(Fraction(count, 1) / fps)
    gallery = "\n".join(
        f'<a href="{html.escape(name)}"><img src="{html.escape(name)}" alt="原视频与 DLSS-NR 输出的同帧对比"></a>'
        for name in images if name != "contact_sheet.png")
    page = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>赛车视频 · OpenDLSS-NR 对比</title>
<style>body{{margin:0;background:#10151c;color:#e7edf7;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1664px;margin:auto;padding:24px}}h1{{font-size:28px}}video,img{{display:block;width:100%;height:auto;background:#000;border-radius:8px}}a{{color:#8fc7ff}}.links{{display:flex;gap:24px;flex-wrap:wrap}}.note{{color:#b7c6d8}}details{{margin:24px 0}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}}.gallery img{{border-radius:4px}}</style>
<main><h1>赛车视频 · OpenDLSS-NR 对比</h1>
<p>左侧：原视频。右侧：项目内 OpenDLSS-NR 输出。</p>
<video controls autoplay muted loop playsinline preload="metadata" poster="comparison-{0:06d}.png"><source src="racing-comparison.mp4" type="video/mp4"></video>
<p class="links"><a href="racing-comparison.mp4">打开对比视频</a><a href="racing-dlss-nr.mp4">打开输出视频</a><a href="original.mp4">打开原视频</a><a href="verification.json">查看验证记录</a></p>
<p class="note">输出规格：{width} × {height}；{float(fps):g} 帧/秒；{count} 帧；{duration:g} 秒。对比视频顶部添加 40 像素标题。</p>
<p class="note">模型逐帧独立重渲染。每帧使用固定种子 0。模型没有读取前一帧输出，也没有时序反馈。此模型保持原分辨率，不提供超分辨率。</p>
<p class="note">视频采用 H.264 有损编码。RGBA 原始输出帧保留在 frames 目录。输入与输出的像素差异只表示改变量，不表示画质提升。</p>
<details><summary>单独播放模型输出</summary><video controls playsinline preload="none"><source src="racing-dlss-nr.mp4" type="video/mp4"></video></details>
<details><summary>单独播放原视频</summary><video controls playsinline preload="none"><source src="original.mp4" type="video/mp4"></video></details>
<h2>同帧对比</h2><div class="gallery">{gallery}</div>
</main></html>
"""
    (out / "review.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    args = parser.parse_args()
    out, source, ffmpeg = args.out.resolve(), args.input.resolve(), args.ffmpeg.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "startedAt": datetime.now(timezone.utc).isoformat(),
              "checks": {}, "files": {}, "limits": [
                  "Independent per-frame inference; no temporal feedback or motion reprojection.",
                  "Native NVIDIA output parity was not tested for this video.",
                  "No ground-truth quality score or claim of improved quality is made.",
                  "Input/output pixel differences quantify change only, not quality.",
                  "MP4 uses lossy H.264 yuv420p encoding; raw RGBA outputs preserve model bytes.",
                  "Repeated same-input head equality is checked on one frame, not every frame.",
              ]}
    try:
        require(source.is_file(), f"input video missing: {source}")
        require(ffmpeg.is_file(), f"FFmpeg missing: {ffmpeg}")
        inference = json.loads((out / "inference.json").read_text(encoding="utf-8"))
        config = inference.get("config", inference)
        width, height = int(config["width"]), int(config["height"])
        count, start = int(config["frameCount"]), int(config.get("startFrame", 0))
        fps = Fraction(str(config["fps"]))
        require(count > 0 and width > 0 and height > 0 and fps > 0, "invalid inference dimensions/count/fps")
        require(inference["status"] == "completed", "inference did not complete")
        require(inference["learnedBlockCount"] == 71, "unexpected learned block count")
        require(inference.get("processedFrames") == count, "inference processed frame count mismatch")
        require(inference.get("firstFrameIndex") == start, "inference first frame mismatch")
        require(inference.get("lastFrameIndex") == start + count - 1, "inference last frame mismatch")
        require(inference.get("gpuErrors") == [], "inference contains GPU errors")
        require(inference.get("totalNonfinite") == 0, "inference contains nonfinite head elements")
        determinism = inference.get("determinism", {})
        require(determinism.get("verified") is True, "same-input determinism was not verified")
        require(determinism.get("bitwiseMismatches") == 0, "same-input head bit patterns differ")
        require(determinism.get("repeatedSameFeatures", 0) >= 2, "determinism requires two repeated executions")
        report["checks"]["inference"] = {"verified": True, "learnedBlockCount": 71,
                                          "gpuErrors": [], "totalNonfinite": 0,
                                          "determinism": determinism}
        statistics = [json.loads(line) for line in (out / "frame-stats.jsonl").read_text(encoding="utf-8").splitlines()
                      if line.strip()]
        require([entry["index"] for entry in statistics] == list(range(start, start + count)),
                "frame statistics are not ordered, complete, and unique")
        expected_bytes = width * height * 4
        sequence_digest = hashlib.sha256()
        for position, entry in enumerate(statistics, 1):
            index = entry["index"]
            require(entry.get("frameIndex") == index, f"frame {index}: inconsistent frame index")
            require(entry.get("processedFrames") == position, f"frame {index}: inconsistent processed count")
            require(entry["head"]["nonfinite"] == 0, f"frame {index}: nonfinite model head")
            path = out / "frames" / f"{index:06d}.rgba"
            require(path.stat().st_size == expected_bytes, f"frame {index}: RGBA dimensions mismatch")
            actual = sha256(path)
            require(actual == entry["outputSha256"], f"frame {index}: output hash mismatch")
            sequence_digest.update(f"{index:06d} {actual}\n".encode("ascii"))
        report["scope"] = {"width": width, "height": height, "frameCount": count,
                           "firstFrame": start, "lastFrame": start + count - 1, "frameRate": str(fps),
                           "durationSeconds": float(Fraction(count, 1) / fps),
                           "sameResolution": True, "temporalFeedback": False}
        report["checks"]["rawFrames"] = {"verified": True, "framesChecked": count,
                                          "indicesOrderedCompleteUnique": True,
                                          "bytesPerFrame": expected_bytes, "allOutputHashesMatch": True,
                                          "allHeadNonfiniteCountsZero": True,
                                          "orderedFrameHashesSha256": sequence_digest.hexdigest()}
        print(f"VERIFIED {count} raw output frames and inference evidence", flush=True)
        output = out / "racing-dlss-nr.mp4"
        output_probe = probe(output)
        report["checks"]["outputVideo"] = verify_video(output_probe, width, height, count, fps, "output")
        report["outputProbe"] = output_probe
        copied = out / "original.mp4"
        if source != copied:
            shutil.copy2(source, copied)
        require(sha256(source) == sha256(copied), "copied original video hash mismatch")
        report["checks"]["originalCopy"] = {"verified": True, "byteIdentical": True}
        destination = out / "racing-comparison.mp4"
        print("ENCODING labeled side-by-side comparison", flush=True)
        encode_comparison(source, output, destination, ffmpeg, width, height, start, count, fps)
        comparison_probe = probe(destination)
        report["checks"]["comparisonVideo"] = verify_video(
            comparison_probe, width * 2, height + 40, count, fps, "comparison")
        report["comparisonProbe"] = comparison_probe
        images = comparisons(source, out, ffmpeg, statistics, width, height, fps)
        report["checks"]["comparisonImages"] = {"verified": True, "files": images,
                                                 "sampleInputHashesMatch": True,
                                                 "fullComparisonsAtSourcePixelSize": True,
                                                 "contactSheetDisplayResampling": "LANCZOS; matching scale on both sides"}
        write_review(out, width, height, count, fps, images)
        for name in ["original.mp4", "racing-dlss-nr.mp4", "racing-comparison.mp4", "inference.json",
                     "frame-stats.jsonl", "review.html", *images]:
            path = out / name
            report["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        report["encoding"] = {"comparison": {"codec": "libx264", "crf": 16,
                                               "pixelFormat": "yuv420p", "faststart": True,
                                               "left": "original", "right": "OpenDLSS-NR",
                                               "titleBarPixels": 40}}
        report["status"] = "verified"
        print(f"VERIFIED video package: {out / 'review.html'}", flush=True)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        print(f"FAILED: {error}", file=sys.stderr, flush=True)
    report["finishedAt"] = datetime.now(timezone.utc).isoformat()
    write_json(out / "verification.json", report)
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
