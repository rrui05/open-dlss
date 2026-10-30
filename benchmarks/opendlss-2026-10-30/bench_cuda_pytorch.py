"""Run the original ht71 DLSS-NR reference arithmetic on CUDA.

Only the inference boundary is adapted: precomputed GPU features enter the
learned graph; a four-channel raw head leaves it. Image preparation and the
WebGPU-compatible CPU composition are excluded from primary timing.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
DEFAULT_REFERENCE = PROJECT.parent / "OpenDLSS-CUDA"


def load_dependencies(reference_root):
    # Parse --help without importing CUDA or requiring the optional reference.
    global np, Image, torch, RESOURCES, _find_weights_resource, iter_records
    global Controls, frame_to_tensor, load_image, swin_core, NRNet, Weights
    sys.path.insert(0, str(reference_root))
    import numpy as np
    from PIL import Image
    import torch
    from dlssnr.extract import RESOURCES, _find_weights_resource, iter_records
    from dlssnr.io import Controls, frame_to_tensor, load_image
    from dlssnr.layers.pre_post import swin_core
    from dlssnr.model import NRNet
    from dlssnr.weights import Weights

MIB = 1024 ** 2


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()), "max_ms": float(a.max()),
            "std_ms": float(a.std()), "fps_from_mean": 1000 / float(a.mean()), "samples_ms": values}


def verify_source(weights, manifest_dir, dll):
    manifest = json.loads((manifest_dir / "manifest.json").read_text())
    dll_hash = sha256(dll)
    assert weights.metadata["network"] == "ht"
    assert weights.metadata["dll_sha256"] == dll_hash == manifest["provenance"]["source_sha256"]
    assert manifest["totals"] == {"blockCount": 71, "tensorCount": 153}
    blocks = sorted({int(key.split(".")[0][5:]) for key in weights.tensors})
    assert blocks == list(range(71))
    stages = {stage["id"]: (manifest_dir / "model" / stage["file"]).read_bytes() for stage in manifest["stages"]}
    for stage in manifest["stages"]:
        assert hashlib.sha256(stages[stage["id"]]).hexdigest() == stage["sha256"]
    lookup = {item["name"]: item for item in manifest["tensors"]}
    with dll.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as image:
        offset, size = _find_weights_resource(image, RESOURCES["ht"])
        records = list(iter_records(image, offset, size))
        assert len(records) == len(lookup) == 153
        for name, offset, length in records:
            item = lookup[name]
            assert item["byteLength"] == length
            packed = stages[item["stage"]]
            assert image[offset:offset + length] == packed[item["stageOffset"]:item["stageOffset"] + length]
    return {"dll_sha256": dll_hash, "source_matches_webgpu_manifest": True,
            "packed_stage_hashes_verified": len(stages), "packed_records_byte_equal": len(records),
            "network": "ht", "blocks": blocks, "unpacked_tensors": len(weights.tensors),
            "weights_metadata": weights.metadata}


def compose_webgpu(head, frame):
    """parity.js composeImage: fma32, clamp, half toward zero, byte publication."""
    residual = head[0, :3, :frame.shape[0], :frame.shape[1]].permute(1, 2, 0).float().cpu().numpy()
    proxy = (frame.astype(np.float64) / 255.0).astype(np.float32)
    centred = (proxy.astype(np.float64) * 0.125 - 0.0625).astype(np.float32)
    inner = (residual.astype(np.float64) * 0.03125 + centred.astype(np.float64)).astype(np.float32)
    pixels = np.clip((inner * np.float32(8) + np.float32(0.5)).astype(np.float32), 0, 1)
    half = pixels.astype(np.float16)
    # NumPy casts round to nearest. Step down only where the rounded half is
    # above the positive source, yielding the publication's toward-zero result.
    half = np.where(half.astype(np.float32) > pixels, np.nextafter(half, np.float16(0)), half)
    values = half.astype(np.float32)
    codes = np.floor((values * np.float32(255) + np.float32(0.5)).astype(np.float32))
    return np.clip(codes, 0, 255).astype(np.uint8)


def metrics(a, b):
    delta = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(delta ** 2))
    return {"psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
            "mae_uint8": float(np.abs(delta).mean()), "max_abs_uint8": float(np.abs(delta).max()),
            "exact_channel_fraction": float(np.mean(delta == 0)), "perfect_match": mse == 0}


def run_timed(net, image, controls):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    began = time.perf_counter()
    start.record()
    outputs = net.run(image, controls, capture=False)
    end.record()
    torch.cuda.synchronize()
    return outputs[len(net.plan) - 1], start.elapsed_time(end), (time.perf_counter() - began) * 1000


def raw_post(self, x, skip, frame):
    wide = swin_core(self.params, self.blend(x, skip), self.spec.shift_x, self.spec.shift_y)
    return torch.einsum("rc,bchw->brhw", self.project, wide)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "inputs/street_896x512.png")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/cuda_pytorch_896x512")
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE,
                        help="Checkout of mauri870/dlssnr-pytorch (not included)")
    parser.add_argument("--weights", type=Path, help="Extracted safetensors; default: REFERENCE_ROOT/weights.safetensors")
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT / "models/nr")
    parser.add_argument("--dll", type=Path, default=PROJECT / "runtime/nvngx_dlssnr.dll")
    parser.add_argument("--reference-output", type=Path, default=ROOT / "results/opendlss_896x512/output.png")
    parser.add_argument("--eager", action="store_true", help="Disable elementwise torch.compile, as in the historical smoke run")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.warmups < 0 or args.iterations < 1:
        parser.error("--warmups must be nonnegative and --iterations must be positive")
    if not (args.reference_root / "dlssnr").is_dir():
        parser.error(f"Reference checkout missing: {args.reference_root}; use --reference-root")
    os.environ["DLSSNR_TRITON"] = "0"
    os.environ["DLSSNR_COMPILE"] = "0" if args.eager else "1"
    load_dependencies(args.reference_root)
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "results.json"
    report = {"status": "running", "stage": "initialization", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "command": sys.argv, "configuration": {"width": 896, "height": 512, "warmups": args.warmups,
                  "iterations": args.iterations, "batch_size": 1, "precision": "FP32 reference arithmetic with FP8 E4M3 publication and prescribed FP16 roundings", "seed": 0},
              "method": {"backend": "PyTorch CUDA reference arithmetic", "temporal": False, "training": False,
                  "scope": "Resident precomputed 16-channel GPU features through original ht71/152 learned layers to raw 4-channel head. Excludes input preparation, host readback, CPU composition and PNG save.",
                  "implementation_boundary_adaptation": "Cache PreBlock.features without changing its values; PostBlock returns four projected raw channels and leaves RGB composition outside timing.",
                  "limitations": "Independent reference implementation. No bit-exact claim. E4M3 is emulated with FP32 tensors; no native FP8 TensorCore GEMM or BF16 substitution. TF32 and reduced-precision reductions disabled."}}
    try:
        report["environment"] = {"gpu_name": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "python": sys.version, "executable": sys.executable,
            "triton_amd_kernels_enabled": False, "elementwise_compile_enabled": os.environ["DLSSNR_COMPILE"],
            "tf32": torch.backends.cuda.matmul.allow_tf32, "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / MIB}
        write_report(result_path, report)
        weights_file = args.weights or args.reference_root / "weights.safetensors"
        weights = Weights.load(str(weights_file))
        report["stage"] = "verifying_weights"
        write_report(result_path, report)
        report["weight_validation"] = verify_source(weights, args.manifest_dir, args.dll)
        frame = load_image(str(args.input))
        assert frame.shape == (512, 896, 3)
        report["input"] = {"path": str(args.input), "file_sha256": sha256(args.input), "dimensions_wh": [896, 512]}
        expected_input_hash = json.loads((ROOT / "inputs/source.json").read_text())["sha256"]
        assert report["input"]["file_sha256"] == expected_input_hash
        report["input"]["matches_shared_input_source"] = True
        report["stage"] = "loading_model"
        write_report(result_path, report)
        began = time.perf_counter()
        net = NRNet(weights, 896, 512, network="ht").cuda().eval()
        assert len(net.plan) == 152 and sorted({spec.block for spec in net.plan}) == list(range(71)) and not net.skip
        image = frame_to_tensor(frame, "cuda", "ht")
        controls = Controls(seed=0)
        pre, post = net.layers[0], net.layers[-1]
        pre.controls, post.controls = controls, controls
        report["model"] = {"name": "DLSS-NR 310.8.0 ht71 CUDA reference", "blocks": 71, "layers": len(net.plan),
            "weights_path": str(weights_file), "weights_sha256": sha256(weights_file), "source": str(args.reference_root),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.reference_root, text=True).strip(),
            "padding": dict(net.padding._asdict()), "controls": asdict(controls),
            "load_and_device_wall_ms": (time.perf_counter() - began) * 1000}
        # The 16-feature preparation is performed once, outside graph timing.
        report["stage"] = "feature_preparation_and_first_helper_compilation"
        write_report(result_path, report)
        torch.cuda.synchronize(); began = time.perf_counter()
        features = pre.features(image)
        torch.cuda.synchronize()
        report["feature_preparation_wall_ms"] = (time.perf_counter() - began) * 1000
        pre.register_buffer("prepared_features", features, persistent=False)
        pre.features = types.MethodType(lambda self, unused: self.prepared_features, pre)
        post.project = weights.f16(70, 0, "out_project", 16, 32)[:4].contiguous().cuda()
        post.forward = types.MethodType(raw_post, post)
        del weights
        report["stage"] = "first_graph_forward_including_compilation"
        write_report(result_path, report)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            head, gpu_ms, wall_ms = run_timed(net, image, controls)
            assert head.shape == (1, 4, 512, 896) and bool(head.isfinite().all())
            report["first_forward"] = {"cuda_event_ms": gpu_ms, "synchronized_wall_ms": wall_ms,
                "output_shape": list(head.shape), "all_finite": True,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB}
            report["cold_start"] = {
                "first_feature_preparation_including_helper_compile_wall_ms": report["feature_preparation_wall_ms"],
                "first_graph_forward_including_compile_wall_ms": wall_ms,
                "excluded_from_steady_state_benchmark": True,
            }
            np.save(args.output / "raw_head.npy", head.float().cpu().numpy())
            began = time.perf_counter()
            output = compose_webgpu(head, frame)
            report["composition_including_readback_wall_ms"] = (time.perf_counter() - began) * 1000
            Image.fromarray(output).save(args.output / "output.png")
            reference = args.reference_output
            report["output_validation"] = {"dimensions_wh": [896, 512], "all_finite": True,
                "webgpu_reference": str(reference), "difference_vs_webgpu": metrics(output, load_image(str(reference)))}
            eager_reference = ROOT / "cuda_pytorch_smoke_896x512/output.png"
            if eager_reference.is_file() and eager_reference.parent != args.output:
                report["output_validation"]["difference_vs_eager_cuda_smoke"] = metrics(output, load_image(str(eager_reference)))
            report["images"] = [{"input": str(args.input), "output": str(args.output / "output.png")}]
            print(json.dumps({"smoke": report["first_forward"], "difference_vs_webgpu": report["output_validation"]["difference_vs_webgpu"]}), flush=True)
            del head
            write_report(result_path, report)
            if args.smoke_only:
                report["status"] = "completed"; report["stage"] = "smoke_complete"
                write_report(result_path, report); return 0
            report["stage"] = "warmup"; write_report(result_path, report)
            for i in range(args.warmups):
                head, gpu_ms, wall_ms = run_timed(net, image, controls)
                del head
                print(f"warmup {i+1}/{args.warmups}: {wall_ms:.3f} ms", flush=True)
            report["stage"] = "benchmark"; write_report(result_path, report)
            cuda, wall = [], []
            torch.cuda.reset_peak_memory_stats()
            for i in range(args.iterations):
                head, gpu_ms, wall_ms = run_timed(net, image, controls)
                if i == args.iterations - 1:
                    last_head = head.float().cpu().numpy()
                    first_head = np.load(args.output / "raw_head.npy")
                    report["determinism"] = {"repeated_identical_prepared_features": True,
                        "head_elements": int(last_head.size),
                        "bitwise_mismatches": int(np.count_nonzero(last_head.view(np.uint32) != first_head.view(np.uint32))),
                        "all_finite": bool(np.isfinite(last_head).all())}
                del head
                cuda.append(gpu_ms); wall.append(wall_ms)
                print(f"benchmark {i+1}/{args.iterations}: CUDA {gpu_ms:.3f} ms; wall {wall_ms:.3f} ms", flush=True)
            report["benchmark"] = {"cuda_event": describe(cuda), "synchronized_wall": describe(wall),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
                "scope": report["method"]["scope"]}
            report["status"] = "completed"; report["stage"] = "complete"
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_report(result_path, report)
            print(json.dumps(report["benchmark"]), flush=True)
    except Exception as exc:
        report["status"] = "failed"; report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_report(result_path, report); raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
