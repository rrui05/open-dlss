"""Collect native timing samples and compose its saved head, using CPU only."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("D:/newDLSS/open-dlss")
OUT = ROOT / "results/native_ptx_896x512"
OUT.mkdir(parents=True, exist_ok=True)


def file_hash(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def read_text(path):
    raw = path.read_bytes()
    return raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")


def describe(values):
    x = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean_ms": float(x.mean()), "median_ms": float(np.median(x)),
            "p95_ms": float(np.percentile(x, 95)), "std_ms": float(x.std()), "min_ms": float(x.min()),
            "max_ms": float(x.max()), "fps_from_mean": 1000 / float(x.mean()), "samples_ms": list(values)}


log_path = ROOT / "cuda_inspect/native_timed.log"
log = read_text(log_path)
samples = [json.loads(line.split("SAMPLE: ", 1)[1]) for line in log.splitlines() if line.startswith("SAMPLE: ")]
assert len(samples) == 20 and len([line for line in log.splitlines() if line.startswith("WARMUP: ")]) == 5
head_path = ROOT / "native_dump/head.f32"
feature_path = ROOT / "native_dump/features.f32"
head = np.fromfile(head_path, dtype=np.float32).reshape(512, 896, 4)
features = np.fromfile(feature_path, dtype=np.float32).reshape(512, 896, 16)
assert np.isfinite(head).all() and np.isfinite(features).all()
# The native composition uses one float32 fused multiply-add for head/32 +
# the half-grid centered color. Float64 holds this sum before its float32 cast.
inner = (head[..., :3].astype(np.float64) * (1.0 / 32) + features[..., 4:7].astype(np.float64)).astype(np.float32)
values = np.clip(inner * np.float32(8) + np.float32(0.5), 0, 1)
# All values are nonnegative: if round-to-nearest half rounded up, the preceding
# half bit pattern gives the required native truncation toward zero.
halves = values.astype(np.float16)
half_bits = halves.view(np.uint16).copy()
half_bits -= (halves.astype(np.float32) > values).astype(np.uint16)
published = half_bits.view(np.float16).astype(np.float32)
rgb = np.floor(published * np.float32(255) + np.float32(0.5)).astype(np.uint8)
Image.fromarray(rgb, "RGB").save(OUT / "output.png")
reference_path = ROOT / "results/opendlss_896x512/output.png"
reference = np.asarray(Image.open(reference_path).convert("RGB"))
delta = rgb.astype(np.float64) - reference.astype(np.float64)
mse = float(np.mean(delta ** 2))
comparison = {"reference": str(reference_path), "channels": int(delta.size),
              "mean_abs_uint8": float(np.abs(delta).mean()), "max_abs_uint8": float(np.abs(delta).max()),
              "rmse_uint8": math.sqrt(mse), "psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
              "changed_channel_count": int(np.count_nonzero(delta)),
              "channels_exceeding_one_code": int(np.count_nonzero(np.abs(delta) > 1)),
              "fraction_exceeding_one_code": float(np.mean(np.abs(delta) > 1)),
              "within_one_code": bool((np.abs(delta) <= 1).all()),
              "scope": "Same prepared RGB PNG and seed; original independent preprocess paths. This is a comparison to our WebGPU output, not an NVIDIA native capture."}
parity_log_path = ROOT / "cuda_inspect/native_parity_features.log"
parity_log = read_text(parity_log_path)
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
manifest = json.loads((SOURCE / "models/nr/manifest.json").read_text())
report = {
    "status": "completed", "stage": "complete", "collected_at_utc": datetime.now(timezone.utc).isoformat(),
    "configuration": {"width": 896, "height": 512, "full_width": 896, "full_height": 512,
                      "batch_size": 1, "warmups": 5, "iterations": 20, "seed": 0,
                      "precision": "Original FP8 E4M3 and FP16 publications, tensor-core PTX"},
    "environment": {"gpu_name": gpu.split(",")[0], "gpu_driver_query": gpu},
    "method": {"backend": "Original Vulkan host with CUDA/PTX tensor-core kernels and SPIR-V utility kernels",
               "pure_cuda_host": False, "training": False,
               "timing": "Main synchronized wall includes graph.record, submission, and GPU wait; GPU query spans graph only. Excludes model loading, real input feature preparation/upload, readback, composition, and PNG save.",
               "source_changes": "Tracked network source unchanged. Untracked main copy adds file-based resident features, 5 warmups, 20 samples, synchronized wall timers and unbuffered diagnostics.",
               "weight_compatibility": "All 71 original stage hashes verified. Separate manifest only uppercases SHA256 hex text to match the original native hash formatter. Stage bytes unchanged via hardlinks.",
               "cross_backend_parity": "Output comparison to the original WebGPU path exceeds its one-code tolerance. Timing and finite/repeatable execution are verified; bit-exact WebGPU output parity is not established."},
    "input": {"rgb_file": str(ROOT / "inputs/street_896x512.png"), "rgb_file_sha256": file_hash(ROOT / "inputs/street_896x512.png"),
              "features_file": str(feature_path), "features_file_sha256": file_hash(feature_path),
              "features_shape_hwc": [512, 896, 16], "features_all_finite": True,
              "feature_source": "Original native preprocessing of the identical prepared RGB file, seed 0."},
    "model": {"name": "DLSS-NR 310.8.0.0", "learned_blocks": manifest["totals"]["blockCount"],
              "tensors": manifest["totals"]["tensorCount"], "graph_dispatches": samples[0]["dispatches"],
              "fused_blocks": True, "counter_chaining": True, "ptx_target": "sm_89, PTX 8.7, driver JIT on local RTX 5060",
              "original_manifest_sha256": file_hash(SOURCE / "models/nr/manifest.json"),
              "executable": str(SOURCE / "build/dlss5vk_timed.exe"), "executable_sha256": file_hash(SOURCE / "build/dlss5vk_timed.exe")},
    "benchmark": {"synchronized_wall": describe([s["synchronized_wall_ms"] for s in samples]),
                  "gpu_timestamp": describe([s["cuda_ptx_gpu_ms"] for s in samples]), "samples": samples},
    "validation": {"head_all_finite": True, "head_shape_hwc": list(head.shape),
                   "head_min": float(head.min()), "head_max": float(head.max()),
                   "three_identical_production_heads": "head over 3 production submissions: identical" in parity_log,
                   "chaining_equals_barriers": "head with barriers instead of chaining: identical" in parity_log,
                   "webgpu_output_tolerance_passed": comparison["within_one_code"],
                   "parity_cli_exit_code": 1,
                   "parity_reference_is_nvidia_capture": False,
                   "parity_log": str(parity_log_path)},
    "image_comparison_to_original_webgpu": comparison,
    "output_png": str(OUT / "output.png"), "output_png_sha256": file_hash(OUT / "output.png"),
    "head_file": str(head_path), "head_file_sha256": file_hash(head_path), "timing_log": str(log_path),
}
(OUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({"results": str(OUT / "results.json"), "wall": report["benchmark"]["synchronized_wall"],
                  "gpu": report["benchmark"]["gpu_timestamp"], "comparison": comparison}))
