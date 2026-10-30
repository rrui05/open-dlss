"""Build the offline OpenDLSS-only report from the archived measurements (CPU only)."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
CASES = [
    ("WebGPU / WGSL", "results/opendlss_896x512", "gpu_timestamp"),
    ("PyTorch CUDA reference", "cuda_pytorch_896x512", "cuda_event"),
    ("Native Vulkan + CUDA/PTX", "results/native_ptx_896x512", "gpu_timestamp"),
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build(root):
    rows = []
    for label, directory, gpu_key in CASES:
        result = read_json(root / directory / "results.json")
        if result["status"] != "completed":
            raise ValueError(f"Incomplete historical measurement: {directory}")
        timing = result["benchmark"]
        wall = timing["synchronized_wall"]
        samples = wall["samples_ms"]
        if len(samples) != 20 or not abs(statistics.mean(samples) - wall["mean_ms"]) < 1e-6:
            raise ValueError(f"Invalid historical sample mean/count: {directory}")
        if result["configuration"]["width"] != 896 or result["configuration"]["height"] != 512:
            raise ValueError(f"Unexpected historical geometry: {directory}")
        rows.append({"backend": label, "directory": directory, "synchronized_wall": wall,
                     "gpu_mean_ms": timing[gpu_key]["mean_ms"]})
    native = read_json(root / CASES[2][1] / "results.json")
    cross = read_json(root / "results/webgpu_same_native_features/results.json")
    pixels = native["validation"]["same_features_png_comparison"]
    preprocessing = cross["preprocessing_comparison"]["whole_tensor"]
    if preprocessing["bitwise_mismatches"] != 0:
        raise ValueError("Historical preprocessing equality check no longer matches the archived result")
    input_path = root / "inputs/street_896x512.png"
    provenance = read_json(root / "inputs/source.json")
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if digest != provenance["sha256"]:
        raise ValueError("Archived input hash mismatch")
    summary = {
        "scope": "OpenDLSS-only archived single-image inference measurements; report generation does not run a GPU",
        "date": "2026-10-30", "gpu": "NVIDIA GeForce RTX 5060", "driver": "610.88",
        "input_sha256": digest, "dimensions_wh": [896, 512], "warmups": 5, "iterations": 20,
        "primary_metric": "synchronized_wall", "results": rows,
        "native_speedup_vs_webgpu_wall": rows[0]["synchronized_wall"]["mean_ms"] / rows[2]["synchronized_wall"]["mean_ms"],
        "cross_backend_validation": {
            "preprocessing_bitwise_equal": True,
            "head_bitwise_equal": native["validation"]["same_features_head_bitwise_equal"],
            "webgpu_output_tolerance_passed": native["validation"]["webgpu_output_tolerance_passed"],
            "same_features_png_comparison": pixels,
        },
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    table = "".join(
        f'<tr><td>{html.escape(row["backend"])}</td><td>{row["synchronized_wall"]["mean_ms"]:.3f}</td>'
        f'<td>{row["synchronized_wall"]["p95_ms"]:.3f}</td><td>{row["synchronized_wall"]["fps_from_mean"]:.2f}</td>'
        f'<td>{row["gpu_mean_ms"]:.3f}</td><td><a href="{row["directory"]}/results.json">Raw JSON</a></td></tr>'
        for row in rows
    )
    images = '<figure><img src="inputs/street_896x512.png"><figcaption>Shared input</figcaption></figure>'
    images += "".join(
        f'<figure><a href="{row["directory"]}/output.png"><img src="{row["directory"]}/output.png"></a>'
        f'<figcaption>{html.escape(row["backend"])}</figcaption></figure>' for row in rows
    )
    document = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenDLSS inference benchmarks — 2026-10-30</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#18212f;background:#f7f9fc}}
table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:12px;border:1px solid #c9d2df;text-align:left}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}}figure{{margin:0}}img{{width:100%}}
.notice{{padding:16px;background:#fff0d0;border-left:5px solid #c47b00}}code{{overflow-wrap:anywhere}}</style>
<h1>OpenDLSS inference benchmarks</h1>
<p>Archived run: 2026-10-30. RTX 5060, driver 610.88. Input: 896 × 512, batch 1, seed 0.
Five warmups and twenty measured submissions per backend. No training.</p>
<p>The primary metric is CPU wall time including graph submission and the GPU wait.
Model loading, compilation, input feature preparation/upload, readback, RGB composition and image saving are excluded.
GPU-only timers are listed separately. The WebGPU timestamp series was collected in a separate set of twenty calls.</p>
<table><thead><tr><th>Backend</th><th>Mean wall ms</th><th>P95 wall ms</th><th>Wall FPS</th><th>Mean GPU ms</th><th>Evidence</th></tr></thead><tbody>{table}</tbody></table>
<p>The native implementation already existed in OpenDLSS. This experiment added a timing entry and input adapters;
it did not introduce a new network or new PTX optimizations. Vulkan submits CUDA/PTX kernels and SPIR-V utilities.
The PyTorch route is an independent reference with FP32 tensors emulating FP8 publications; it is not a native FP8 tensor-core implementation.</p>
<p class="notice"><strong>Cross-backend numerical tolerance: FAIL.</strong>
Native timing is valid as a runtime measurement, but the native output did not pass the WebGPU ±1-code comparison.
The independently prepared 16-lane input features were bit-identical. With identical features, the raw heads still differ.
RGB PSNR: {pixels["psnr_db"]:.4f} dB; mean absolute error: {pixels["mae_uint8"]:.4f}/255;
maximum error: {pixels["max_abs_uint8"]:g}/255. The responsible graph operator has not been isolated.
Each backend's repeated outputs were finite and repeatable in the recorded checks.</p>
<p><a href="results/webgpu_same_native_features/results.json">Same-feature cross-check</a> ·
<a href="cuda_inspect/native_parity_features.log">Native parity log</a> ·
<a href="cuda_inspect/native_timed.log">Native timing log</a> · <a href="summary.json">Summary JSON</a> ·
<a href="README.md">Reproduction instructions</a></p>
<div class="images">{images}</div>
<p>Input SHA-256: <code>{digest}</code>. <a href="inputs/source.json">Input provenance</a>.</p>
<p>Raw historical JSON files retain the original machine paths as evidence. Those paths are not portable invocation defaults.
Large intermediate tensors, executables, DLLs and model weights are not included.</p></html>'''
    (root / "report.html").write_text(document, encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Directory holding the archived results")
    args = parser.parse_args()
    report = build(args.root)
    print(json.dumps({"report": str(args.root / "report.html"), "backends": len(report["results"])}))
