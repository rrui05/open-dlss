# OpenDLSS inference benchmarks, 2026-10-30

Runnable single-image harnesses for this repository's WebGPU and native
Vulkan + CUDA/PTX backends, plus an optional independent PyTorch CUDA reference.
All use the original DLSS-NR 310.8.0, 71-block network. No training is involved.

Open [report.html](report.html) locally for the three-backend results and output
images. [summary.json](summary.json) contains the comparison data.

## Recorded measurements

RTX 5060, driver 610.88, 896 × 512, batch 1, seed 0, five warmups and twenty
measured submissions. These are archived measurements from the original local
experiment, not new measurements produced while packaging these files.

| Backend | Mean synchronized wall | P95 wall | Wall FPS | Mean GPU time |
| --- | ---: | ---: | ---: | ---: |
| WebGPU / WGSL | 158.850 ms | 161.850 ms | 6.30 | 155.014 ms |
| PyTorch CUDA reference | 203.081 ms | 211.377 ms | 4.92 | 201.200 ms |
| Native Vulkan + CUDA/PTX | 5.996 ms | 7.155 ms | 166.79 | 4.087 ms |

The primary interval starts with resident, prepared 16-lane input features and
ends after the raw four-channel head is ready. Synchronized wall time includes
graph recording/submission and the GPU wait. It excludes loading, first-use
compilation, feature preparation/upload, output readback, RGB composition and
PNG saving. WebGPU GPU timestamps came from a separate twenty-call series.
Do not compare a GPU-only time against another backend's wall time.

The fast native backend already existed in this repository. The harness adds
real-image inputs, warmups, timing and artifact collection; it does not add new
PTX kernel optimizations. Native uses Vulkan to submit CUDA/PTX tensor-core
kernels and SPIR-V utilities. The PyTorch reference emulates FP8 publications
using FP32 tensors, and is not a native FP8 tensor-core implementation.

**Native/WebGPU numerical tolerance failed in this experiment.** Independently
prepared input features matched bit for bit, but the graph heads differed even
when both backends received the same saved features. The resulting RGB images
had PSNR 50.3643 dB, MAE 0.4814/255 and maximum error 8/255. The differing graph
operator has not been isolated. Finite outputs, repeatability and native
counter-chaining versus barrier equality were checked separately. Runtime speed
does not establish cross-backend numerical equivalence.

## Files and dependencies

- `opendlss_harness/`: isolated headless-browser runner, HTTP server, benchmark
  page and identical-feature investigation page.
- `prepare_native_timed_main.py`: generates a timing entry from `src/main.cpp`
  into ignored `build/benchmark_native/`, preserving the production graph.
- `native_timed/`: build wrapper, native runner and CPU log/output collector.
- `prepare_native_model.py`: verifies every packed stage hash and creates a
  separate manifest with uppercase SHA-256 text for the native loader.
- `prepare_native_fixture.py`: prepares an RGBA float proxy and, when saved
  features and a reference are supplied, a safe features-only parity fixture.
- `bench_cuda_pytorch.py`: optional ht71 reference CUDA benchmark.
- `results/`, `cuda_pytorch_896x512/`: archived JSON and generated PNG images.
- `cuda_pytorch_smoke_896x512/`: archived eager cold smoke run, excluded from the
  steady-state comparison.
- `cuda_inspect/`: historical native timing/parity logs, toolchain evidence and
  offline crash-inspection helpers. `repair_compare_output.patch` is an
  **unapplied** diagnostic candidate, not a tested source fix.

Model weights, NVIDIA DLLs, executables, crash dumps and large intermediate
tensors are excluded. Supply the packed `models/nr/manifest.json` and its stage
files using the repository's documented model layout. Optional reference
weights and a DLL are required only for the PyTorch weight-equality check.
Other projects' inference code, models and benchmark results are not bundled.

The PNG input is the NVIDIA example identified in [inputs/source.json](inputs/source.json),
resized once with Pillow bilinear interpolation to 896 × 512. Every backend
received the exact same file bytes. Archived JSON paths retain their original
machine locations as provenance; executable defaults use this checkout.

## WebGPU

Requirements: Node.js 22 or newer, hardware WebGPU in Edge/Chromium, and the
packed model directory. The historical runner used Node 24.12.0 and Edge 156.
From the repository root:

```powershell
node benchmarks/opendlss-2026-10-30/opendlss_harness/run.mjs --help
node benchmarks/opendlss-2026-10-30/opendlss_harness/run.mjs --width 896 --height 512 --warmups 5 --iterations 20
```

Outputs go to ignored `benchmarks/opendlss-2026-10-30/runs/opendlss_896x512/`.
Use `--out`, `--weights`, `--input` or `--edge` to override defaults. The browser
uses its own profile and is closed when the run finishes. See the
[WebGPU harness README](opendlss_harness/README.md) for all timing and cleanup
details. Different input images require matching `--width` and `--height`.

## Native Vulkan + CUDA/PTX

Use the Windows/MSVC prerequisites in the root README. Python 3.11+, NumPy and
Pillow are needed for fixture preparation and collection. Build the normal
shaders/PTX first, then the generated timing executable:

```powershell
powershell -File scripts/fetch_tools.ps1
powershell -File scripts/build.ps1
powershell -File benchmarks/opendlss-2026-10-30/native_timed/build_timed.ps1
powershell -File benchmarks/opendlss-2026-10-30/run_native_cuda_bench.ps1 -Warmups 5 -Iterations 20
```

`build_timed.ps1` supports `-ToolchainRoot` and `-Python`. The run wrapper
supports `-Model`, `-Executable`, `-Shaders`, `-PtxDirectory`, `-InputPng`,
`-Output`, `-Width`, `-Height`, `-Warmups`, `-Iterations` and `-Python`.
Default output is ignored `build/benchmark_native/run/`; choose a new `-Output`
for each subsequent run, because the runner refuses to overwrite evidence.

The original proxy preprocessing runs once before the timed loop. The same
resident features are reused for every warmup and sample. Dumps, output
composition and JSON collection happen outside timing. The native run saves
`native.log`, `execution.json`, `results.json`, `output.png`, and
`dump/{features,head}.f32`. Missing validation evidence is recorded as unknown,
not as a passing check. `-Reference` optionally compares the composed RGB PNG
with a supplied reference; the numeric result is separate from runtime status.
The original fixed, untimed compilation forward is also retained. Thus
`-Warmups 0` still performs that initial call before the measured loop.

To rerun from already captured features, pass `-FeaturesFile` with a float32
HWC tensor of the model's full padded dimensions and 16 channels. The original
`parity` CLI's `proxy` + `u8` output comparison has a CPU indexing bug; do not
use that combination. Generate a features-only fixture instead:

```powershell
python benchmarks/opendlss-2026-10-30/prepare_native_fixture.py --features-file build/benchmark_native/run/dump/features.f32 --reference benchmarks/opendlss-2026-10-30/results/opendlss_896x512/output.png --output build/benchmark_native/parity-fixture
build/dlss5vk.exe parity --model build/benchmark_native/run/model --fixture build/benchmark_native/parity-fixture
```

That reference is our own WebGPU output, not an NVIDIA capture. A mismatch is
a valid failed numerical check and must not be relabeled as a pass.

For the same-feature WebGPU investigation, copy the exact input PNG beside
copies of the native dumps named `native_features.f32` and `native_head.f32`,
then run the browser harness with `--page cross.html --input <that PNG>`.
The three diagnostic graph calls are not the twenty-sample speed benchmark.

## Optional PyTorch CUDA reference

The historical run used Ubuntu 22.04 under WSL, Python 3.11.16 and
PyTorch 2.11.0+cu128. Use a CUDA-enabled PyTorch environment compatible with the
GPU; NumPy, Pillow and safetensors are also required. Keep the reference checkout
outside this repository. From the repository root:

```text
git clone https://github.com/mauri870/dlssnr-pytorch ../OpenDLSS-CUDA
git -C ../OpenDLSS-CUDA checkout 3beef7935a8ef976b47ab78b906e75600c52d2de
python -m pip install -e ../OpenDLSS-CUDA --no-deps
python -m dlssnr extract runtime/nvngx_dlssnr.dll -o ../OpenDLSS-CUDA/weights.safetensors
python benchmarks/opendlss-2026-10-30/bench_cuda_pytorch.py --reference-root ../OpenDLSS-CUDA --warmups 5 --iterations 20
```

`--weights`, `--dll`, `--manifest-dir`, `--reference-output` and `--output`
override artifact paths. Default output is under ignored `runs/`. This harness
intentionally gates the recorded 896 × 512 input hash. `--eager --smoke-only`
reproduces the uncompiled cold smoke mode, which is not steady-state speed.
The script verifies all 71 packed stage hashes and all 153 packed DLL records
against the same model manifest. It does not skip layers or switch to a smaller
network. `DLSSNR_TRITON=0` disables the reference's AMD-specific kernels;
elementwise `torch.compile` is enabled unless `--eager` is supplied.

## Rebuild the archived report

This reads only the committed JSON and PNG files and does not invoke a GPU:

```powershell
python benchmarks/opendlss-2026-10-30/build_report.py
```

It verifies the input SHA-256, twenty-sample means/counts, dimensions and the
recorded preprocessing-equality result before writing `report.html` and
`summary.json`. Packaging validation is recorded separately in
`publication_validation.json`; it does not replace the historical results.
