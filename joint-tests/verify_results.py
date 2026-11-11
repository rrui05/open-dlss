"""Independently verify saved joint DLSS artifacts without running any model.

The default run counts and fully decodes all existing expected videos.  --quick
omits full video decoding and cannot certify a complete experiment.  Missing
artifacts are reported explicitly, never replaced or treated as passed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image


EXPECTED_CASES = ("cj", "tommy_vice_city", "niko_iv", "roman_iv", "trevor_outdoor",
                  "michael_daylight", "franklin_daylight", "amanda_sunset")
VARIANTS = ("nr", "sr", "sr_nr", "rr", "rr_nr")
FINAL_VIDEO = {"width": 1664, "height": 960, "fps": "32/1", "frames": 1920,
               "duration_seconds": 60, "audio_streams": 0}
NR_VIDEO = {"width": 1664, "height": 960, "fps": "16/1", "frames": 960,
            "duration_seconds": 60, "audio_streams": 0}
DIFFERENCE_NOTE = "Change relative to an input/display-resampled input, not a quality score against ground truth."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> tuple[object | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except (OSError, ValueError) as error:
        return None, str(error)


def rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def check(name: str, condition: bool | None, *, actual=None, expected=None,
          detail: str | None = None) -> dict:
    value = {"name": name, "status": "passed" if condition is True else
             "failed" if condition is False else "unconfirmed"}
    if actual is not None:
        value["actual"] = actual
    if expected is not None:
        value["expected"] = expected
    if detail:
        value["detail"] = detail
    return value


def passed(checks: list[dict]) -> bool:
    return bool(checks) and all(item["status"] == "passed" for item in checks)


def completed(metadata: object) -> bool | None:
    if not isinstance(metadata, dict):
        return None
    state = str(metadata.get("status", metadata.get("state", ""))).lower()
    if state in {"completed", "complete", "success", "succeeded", "ok", "passed", "verified"}:
        return True
    if state in {"error", "failed", "failure", "blocked"}:
        return False
    if metadata.get("success") is True or metadata.get("verified") is True:
        return True
    return None


def nr_checks(metadata: object, *, video: bool = False) -> list[dict]:
    if not isinstance(metadata, dict):
        return [check("NR metadata readable", None)]
    errors = metadata.get("gpuErrors")
    head_nonfinite = metadata.get("totalNonfinite") if video else metadata.get("head", {}).get("nonfinite")
    repetition = metadata.get("determinism", {})
    comparison = repetition.get("comparisons")
    if isinstance(comparison, list) and comparison:
        mismatch_ok = all(isinstance(item, dict) and item.get("bitwiseMismatches") == 0 and
                          isinstance(item.get("float32Elements"), int) and item["float32Elements"] > 0
                          for item in comparison)
    elif isinstance(repetition, dict) and "bitwiseMismatches" in repetition:
        mismatch_ok = (repetition.get("bitwiseMismatches") == 0 and
                       isinstance(repetition.get("float32Elements"), int) and repetition["float32Elements"] > 0)
    else:
        mismatch_ok = None
    repeat_count = repetition.get("repeatedSameFeatures")
    checks = [
        check("NR execution completed", completed(metadata)),
        check("NR learned block count", metadata.get("learnedBlockCount") == 71,
              actual=metadata.get("learnedBlockCount"), expected=71),
        check("NR GPU error list present and empty", isinstance(errors, list) and not errors, actual=errors),
        check("NR nonfinite count present and zero", head_nonfinite == 0, actual=head_nonfinite, expected=0),
        check("NR determinism explicitly verified", repetition.get("verified") is True),
        check("NR repeated feature execution count", isinstance(repeat_count, int) and repeat_count >= 2,
              actual=repeat_count, expected=">=2"),
        check("NR compared outputs have no bitwise mismatches", mismatch_ok),
    ]
    if video:
        checks.append(check("NR video processed every input frame", metadata.get("processedFrames") == 960,
                            actual=metadata.get("processedFrames"), expected=960))
    return checks


def get_manifest_cases(manifest: object) -> list[dict]:
    if not isinstance(manifest, dict):
        return []
    cases = manifest.get("images", manifest.get("cases", []))
    if isinstance(cases, dict):
        return [dict(value, id=key) if isinstance(value, dict) else {"id": key}
                for key, value in cases.items()]
    return [value for value in cases if isinstance(value, dict)] if isinstance(cases, list) else []


def safe_id(value: object) -> str:
    name = str(value)
    if name in {"", ".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Invalid image case ID: {name!r}")
    return name


def inspect_input(root: Path, case: dict) -> dict:
    case_id = safe_id(case.get("id", ""))
    path = root / "images" / case_id / "input.png"
    result = {"id": case_id, "path": str(path), "verified": False, "checks": []}
    if not path.is_file():
        result["missing"] = [str(path)]
        return result
    try:
        image = rgb(path)
        actual_hash = sha256(path)
        expected_hash = case.get("input_sha256")
        result.update({"size": list(image.size), "sha256": actual_hash})
        result["checks"] = [
            check("input file is nonempty", path.stat().st_size > 0),
            check("input SHA-256 matches manifest", actual_hash.lower() == str(expected_hash).lower()
                  if expected_hash else None, actual=actual_hash, expected=expected_hash),
            check("input dimensions match manifest", list(image.size) == list(case["input_size"])
                  if isinstance(case.get("input_size"), list) else None,
                  actual=list(image.size), expected=case.get("input_size")),
        ]
    except (OSError, ValueError) as error:
        result["checks"].append(check("input PNG decodes", False, detail=str(error)))
    result["verified"] = passed(result["checks"])
    return result


def backend_record(metadata: object) -> dict:
    if not isinstance(metadata, dict):
        return {}
    keys = ("backend", "runtime", "dll", "runtime_dll", "runtime_path", "runtime_sha256",
            "backend_is_d4r_native", "model", "preset", "fallback", "guides", "proxy_guides",
            "native_gbuffer", "scope", "limitation", "note")
    return {key: metadata[key] for key in keys if key in metadata}


def inspect_variant(root: Path, case: dict, variant: str, input_record: dict) -> dict:
    case_id = safe_id(case["id"])
    directory = root / "images" / case_id / variant
    output = directory / "output.png"
    inference = directory / "inference.json"
    result = {"case": case_id, "variant": variant, "path": str(output), "verified": False,
              "checks": [], "missing": [str(path) for path in (output, inference) if not path.is_file()]}
    if result["missing"]:
        return result
    metadata, metadata_error = read_json(inference)
    result["backend_evidence"] = backend_record(metadata)
    result["checks"].append(check("inference JSON readable", metadata_error is None, detail=metadata_error))
    base_size = input_record.get("size")
    if not base_size:
        result["checks"].append(check("case input available", None))
        return result
    expected = base_size if variant == "nr" else [2 * value for value in base_size]
    source = root / "images" / case_id / "input.png"
    if variant in {"sr_nr", "rr_nr"}:
        source = directory.parent / variant.split("_")[0] / "output.png"
    if not source.is_file():
        result["missing"].append(str(source))
        return result
    try:
        before, after = rgb(source), rgb(output)
        result.update({"dimensions": list(after.size), "bytes": output.stat().st_size,
                       "sha256": sha256(output), "source_sha256": sha256(source)})
        if isinstance(metadata, dict):
            for field, actual in (("input_sha256", result["source_sha256"]),
                                  ("output_sha256", result["sha256"])):
                if field in metadata:
                    result["checks"].append(check(f"recorded {field} matches artifact",
                                                           str(metadata[field]).lower() == actual,
                                                           actual=actual, expected=metadata[field]))
        prior, prior_error = read_json(directory / "verification.json")
        if prior_error is None and isinstance(prior, dict) and prior.get("output_sha256"):
            result["checks"].append(check("stage verification output hash matches artifact",
                                                   str(prior["output_sha256"]).lower() == result["sha256"],
                                                   actual=result["sha256"], expected=prior["output_sha256"]))
        array = np.asarray(after)
        result["checks"].extend([
            check("output file nonempty", output.stat().st_size > 0),
            check("output dimensions", list(after.size) == expected, actual=list(after.size), expected=expected),
            check("output is not a blank color", float(array.std()) > 1, actual=float(array.std())),
        ])
        staged_input = directory / "input.png"
        if staged_input.is_file():
            result["checks"].append(check("staged input matches intended upstream pixels",
                                                   np.array_equal(np.asarray(rgb(staged_input)), np.asarray(before))))
        displayed = after if after.size == before.size else after.resize(before.size, Image.Resampling.LANCZOS)
        delta = np.abs(np.asarray(displayed).astype(np.int16) - np.asarray(before).astype(np.int16))
        changes = {"changed_channel_fraction": float(np.mean(delta != 0)),
                   "mean_absolute_rgb_change_0_255": float(delta.mean()), "note": DIFFERENCE_NOTE}
        result["change_statistics"] = changes
        result["checks"].append(check("output contains changes", bool(np.any(delta)),
                                               detail=DIFFERENCE_NOTE))
    except (OSError, ValueError) as error:
        result["checks"].append(check("PNG artifacts decode", False, detail=str(error)))
    if variant in {"nr", "sr_nr", "rr_nr"}:
        result["checks"].extend(nr_checks(metadata))
    else:
        result["checks"].append(check("model execution explicitly completed", completed(metadata)))
        evidence = result["backend_evidence"]
        backend_text = json.dumps(evidence, ensure_ascii=False).lower()
        result["checks"].append(check("actual backend recorded", bool(evidence), actual=evidence))
        result["nvidia_runtime_fallback_recorded"] = any(token in backend_text for token in
                                                        ("nvidia", "nvngx", "ngx", "runtime fallback"))
        if variant == "rr":
            result["rr_scope"] = "Proxy guides; not native game G-buffers or raw ray-tracing samples."
    result["verified"] = not result["missing"] and passed(result["checks"])
    return result


def wsl_path(path: Path) -> str:
    value = str(path.resolve())
    if len(value) > 2 and value[1] == ":":
        return "/mnt/" + value[0].lower() + "/" + value[2:].replace("\\", "/").lstrip("/")
    return value


def video_command(tool: str, path: Path, suffix: list[str]) -> list[str]:
    if os.name == "nt":
        return ["wsl.exe", "-e", tool] + suffix
    return [tool] + suffix


def run(command: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)


def inspect_fg_evidence(path: Path) -> dict:
    metadata_path = path.with_suffix(".json")
    statistics_path = path.with_suffix(".pair-stats.jsonl")
    result = {"metadata": str(metadata_path), "pair_statistics": str(statistics_path),
              "checks": [], "missing": []}
    for required in (metadata_path, statistics_path):
        if not required.is_file():
            result["missing"].append(str(required))
    if result["missing"]:
        return result
    metadata, error = read_json(metadata_path)
    result["checks"].append(check("FG execution JSON readable", error is None, detail=error))
    if not isinstance(metadata, dict):
        return result
    for key, expected in (("input_frames", 960), ("generated_frames", 959),
                          ("tail_repeats", 1), ("output_frames", 1920),
                          ("width", 1664), ("height", 960)):
        result["checks"].append(check(f"FG {key}", metadata.get(key) == expected,
                                               actual=metadata.get(key), expected=expected))
    result["checks"].append(check("FG execution explicitly completed", completed(metadata)))
    method = str(metadata.get("method", ""))
    normalized_method = method.lower().replace("-", "").replace("_", "")
    result["checks"].append(check("FG uses MLX-DLSS FrameGenerator", "mlxdlss" in normalized_method and
                                           "framegenerator" in normalized_method, actual=method))
    weights_hash = str(metadata.get("weights_sha256", metadata.get("weights_hash", "")))
    result["checks"].append(check("FG model weight SHA-256 recorded", re.fullmatch(r"[0-9a-fA-F]{64}", weights_hash) is not None,
                                           actual=weights_hash))
    device = str(metadata.get("device", ""))
    result["checks"].append(check("FG CUDA execution device recorded", device.lower().startswith("cuda"), actual=device))
    allocated = metadata.get("cuda_peak_allocated_bytes")
    reserved = metadata.get("cuda_peak_reserved_bytes")
    result["checks"].append(check("FG CUDA allocation evidence present", isinstance(allocated, int) and allocated > 0 and
                                           isinstance(reserved, int) and reserved >= allocated,
                                           actual={"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved}))
    for key, expected in (("input_fps", Fraction(16)), ("output_fps", Fraction(32))):
        try:
            actual = Fraction(str(metadata[key]))
            condition = actual == expected
        except (KeyError, ValueError, ZeroDivisionError):
            condition = False
        result["checks"].append(check(f"FG {key}", condition, actual=metadata.get(key), expected=str(expected)))
    try:
        duration = float(metadata.get("duration_seconds", "nan"))
        duration_ok = math.isfinite(duration) and abs(duration - 60) <= 1 / 32
    except (TypeError, ValueError):
        duration_ok = False
    result["checks"].append(check("FG duration recorded as 60 seconds", duration_ok,
                                           actual=metadata.get("duration_seconds"), expected=60))
    weights_path_value = metadata.get("weights_path", metadata.get("weights"))
    if isinstance(weights_path_value, str):
        weights_path = Path(weights_path_value)
        if not weights_path.is_absolute():
            weights_path = metadata_path.parent / weights_path
        if weights_path.is_file():
            actual_hash = sha256(weights_path)
            result["checks"].append(check("FG actual weight file matches recorded hash", actual_hash == weights_hash.lower(),
                                                   actual=actual_hash, expected=weights_hash))
        else:
            result["missing"].append(str(weights_path))
    try:
        lines = [line for line in statistics_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        rows = [json.loads(line) for line in lines]
        result["checks"].append(check("FG has one statistics row per generated pair", len(rows) == 959,
                                               actual=len(rows), expected=959))
        indices = [row.get("pair") for row in rows if isinstance(row, dict)]
        result["checks"].append(check("FG pair indices are ordered, complete and unique", indices == list(range(959))))
        timings = [row.get("inference_seconds") for row in rows if isinstance(row, dict)]
        finite_timings = len(timings) == 959 and all(isinstance(value, (int, float)) and not isinstance(value, bool) and
                                                  math.isfinite(value) and value >= 0 for value in timings)
        result["checks"].append(check("FG pair inference times are finite and nonnegative", finite_timings))
        result["checks"].append(check("FG each pair generated one actual intermediate frame", len(rows) == 959 and
                                               all(isinstance(row, dict) and row.get("generated_frames") == 1 for row in rows)))
        if finite_timings:
            result["timing_seconds"] = {"sum": float(sum(timings)), "min": float(min(timings)),
                                         "median": float(np.median(timings)), "max": float(max(timings))}
    except (OSError, ValueError, TypeError) as error:
        result["checks"].append(check("FG pair statistics readable", False, detail=str(error)))
    result["execution_evidence"] = {key: metadata[key] for key in
                                    ("state", "status", "method", "device", "precision", "weights_sha256",
                                     "weights_hash", "numerics", "runtime_source", "cuda_peak_allocated_bytes",
                                     "cuda_peak_reserved_bytes", "input_sha256", "output_sha256") if key in metadata}
    if metadata.get("output_sha256") and path.is_file():
        result["checks"].append(check("FG recorded output hash matches video", str(metadata["output_sha256"]).lower() == sha256(path),
                                               actual=sha256(path), expected=metadata["output_sha256"]))
    return result


def inspect_video(root: Path, relative: str, expected: dict, quick: bool,
                  timeout: int, *, nr_metadata: bool = False) -> dict:
    path = root / relative
    result = {"path": str(path), "expected": expected, "verified": False, "checks": [], "missing": []}
    if not path.is_file():
        result["missing"] = [str(path)]
        return result
    if not path.stat().st_size:
        result["checks"].append(check("video nonempty", False))
        return result
    if not nr_metadata:
        fg = inspect_fg_evidence(path)
        result["FG"] = fg
        result["checks"].extend(fg["checks"])
        result["missing"].extend(fg["missing"])
    target = wsl_path(path) if os.name == "nt" else str(path)
    probe = video_command("ffprobe", path, ["-v", "error", "-count_frames", "-show_entries",
                          "stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,"
                          "nb_frames,nb_read_frames,duration,pix_fmt:format=duration", "-of", "json", target])
    result["probe_command"] = probe
    try:
        process = run(probe, timeout)
        result["checks"].append(check("ffprobe completed without errors", process.returncode == 0 and not process.stderr.strip(),
                                               actual=process.returncode, detail=process.stderr[-4000:] or None))
        if process.returncode:
            return result
        data = json.loads(process.stdout)
        result["probe"] = data
        streams = data.get("streams", [])
        videos = [item for item in streams if item.get("codec_type") == "video"]
        audio = [item for item in streams if item.get("codec_type") == "audio"]
        result["checks"].append(check("one video stream", len(videos) == 1, actual=len(videos), expected=1))
        result["checks"].append(check("audio stream count", len(audio) == expected["audio_streams"],
                                               actual=len(audio), expected=expected["audio_streams"]))
        if len(videos) == 1:
            stream = videos[0]
            result["checks"].extend([
                check("video dimensions", [stream.get("width"), stream.get("height")] == [expected["width"], expected["height"]],
                      actual=[stream.get("width"), stream.get("height")], expected=[expected["width"], expected["height"]]),
                check("counted decoded frames", int(stream.get("nb_read_frames", -1)) == expected["frames"],
                      actual=stream.get("nb_read_frames"), expected=expected["frames"]),
                check("average frame rate", Fraction(stream.get("avg_frame_rate", "0")) == Fraction(expected["fps"]),
                      actual=stream.get("avg_frame_rate"), expected=expected["fps"]),
            ])
            duration = float(stream.get("duration", data.get("format", {}).get("duration", "nan")))
            result["checks"].append(check("video duration", math.isfinite(duration) and
                                                   abs(duration - expected["duration_seconds"]) <= 1 / float(Fraction(expected["fps"])),
                                                   actual=duration if math.isfinite(duration) else "missing/nonfinite",
                                                   expected=expected["duration_seconds"]))
        decode = video_command("ffmpeg", path, ["-v", "error", "-xerror", "-i", target,
                                                 "-map", "0:v:0", "-an", "-f", "null", "-"])
        result["decode_command"] = decode
        if quick:
            result["checks"].append(check("full ffmpeg decode", None, detail="Skipped by --quick; not certified."))
        else:
            decoded = run(decode, timeout)
            result["checks"].append(check("full ffmpeg decode", decoded.returncode == 0 and not decoded.stderr.strip(),
                                                   actual=decoded.returncode, detail=decoded.stderr[-4000:] or None))
        if nr_metadata:
            metadata_path = path.parent / "inference.json"
            metadata, error = read_json(metadata_path)
            if error == "missing":
                result["missing"].append(str(metadata_path))
            result["checks"].append(check("NR video metadata readable", error is None, detail=error))
            result["checks"].extend(nr_checks(metadata, video=True))
        result.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
    except (OSError, ValueError, TypeError, ZeroDivisionError, subprocess.TimeoutExpired) as error:
        result["checks"].append(check("video verification execution", False, detail=str(error)))
    result["verified"] = not result["missing"] and passed(result["checks"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs" / "joint-dlss-20261028")
    parser.add_argument("--quick", action="store_true", help="Skip full video decode; never certify it.")
    parser.add_argument("--strict", action="store_true", help="Exit 1 unless all required checks pass.")
    parser.add_argument("--timeout", type=int, default=1800, help="Per probe/decode timeout in seconds.")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    manifest_path = root / "manifest.json"
    manifest, manifest_error = read_json(manifest_path)
    cases = get_manifest_cases(manifest)
    by_id = {safe_id(case.get("id", "")): case for case in cases}
    manifest_checks = [check("manifest readable", None if manifest_error == "missing" else manifest_error is None,
                             detail=manifest_error),
                       check("all eight fixed face cases declared", True if set(EXPECTED_CASES).issubset(by_id) else None,
                             actual=sorted(by_id), expected=list(EXPECTED_CASES))]
    missing = [str(manifest_path)] if manifest_error == "missing" else []
    inputs, images = [], []
    for case_id in EXPECTED_CASES:
        if case_id not in by_id:
            missing.append(f"manifest.json:images[{case_id}]")
            continue
        case = by_id[case_id]
        input_record = inspect_input(root, case)
        inputs.append(input_record)
        missing.extend(input_record.get("missing", []))
        for variant in VARIANTS:
            result = inspect_variant(root, case, variant, input_record)
            images.append(result)
            missing.extend(result.get("missing", []))
    video_source = manifest.get("video", {}) if isinstance(manifest, dict) else {}
    source_path = Path(video_source["input"]) if video_source.get("input") else None
    source_checks = []
    if source_path and source_path.is_file():
        actual = sha256(source_path)
        expected = video_source.get("sha256")
        source_checks.append(check("video input SHA-256 matches manifest", actual == str(expected).lower() if expected else None,
                                   actual=actual, expected=expected))
    else:
        missing.append(str(source_path) if source_path else "manifest.json:video.input")
    videos = []
    for chain in ("sr", "rr"):
        videos.append(inspect_video(root, f"videos/{chain}_nr/racing-dlss-nr.mp4", NR_VIDEO,
                                    args.quick, args.timeout, nr_metadata=True))
        videos.append(inspect_video(root, f"videos/{chain}_nr_fg.mp4", FINAL_VIDEO,
                                    args.quick, args.timeout))
    for result in videos:
        missing.extend(result.get("missing", []))
    all_checks = manifest_checks + source_checks
    all_checks.extend(item for result in inputs + images + videos for item in result["checks"])
    failures = [item for item in all_checks if item["status"] == "failed"]
    complete = (not missing and not failures and passed(all_checks) and len(images) == 40 and
                all(result["verified"] for result in inputs + images + videos))
    report = {"status": "verified" if complete else "failed" if failures else "partial",
              "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
              "method": "Independent saved-artifact checks; no model or GPU inference is executed.",
              "quick_mode": args.quick,
              "summary": {"inputs_verified": sum(item["verified"] for item in inputs), "inputs_expected": 8,
                          "image_variants_verified": sum(item["verified"] for item in images), "image_variants_expected": 40,
                          "videos_verified": sum(item["verified"] for item in videos), "videos_expected": 4,
                          "failed_checks": len(failures), "missing_artifacts": len(set(missing))},
              "manifest_checks": manifest_checks, "source_video_checks": source_checks,
              "manifest_evidence": {key: manifest[key] for key in
                                    ("hardware", "nr", "sr", "rr", "sr_rr", "fg", "backends",
                                     "pipelines", "limitations", "repositories")
                                    if isinstance(manifest, dict) and key in manifest},
              "inputs": inputs, "images": images, "videos": videos,
              "missing": sorted(set(missing)), "failed_checks": failures,
              "scope": {"still_image_FG": "not applicable", "ground_truth_available": False,
                        "difference_metrics": DIFFERENCE_NOTE,
                        "SR_RR_backend": "See manifest and each inference.json; native NVIDIA runtime fallback is not an open model implementation.",
                        "RR_guides": "Flat proxy guides, not native engine G-buffers or ray samples.",
                        "NR_temporal_feedback": False,
                        "native_NVIDIA_output_parity": "not established by these artifact checks"}}
    destination = root / "verification.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report["summary"],
                      "verification": str(destination)}, indent=2))
    return 0 if complete or not args.strict else 1


if __name__ == "__main__":
    sys.exit(main())
