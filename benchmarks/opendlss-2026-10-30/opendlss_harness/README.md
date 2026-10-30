# Standalone OpenDLSS WebGPU harness

Uses the repository's original `Network` and `composeImage` modules. It does not
modify the 71-block graph, weights, shaders, or numerical publications.

Requires Node.js 22 or newer (global `fetch` and `WebSocket`), a Chromium browser
with hardware WebGPU, and a user-supplied packed model at `<repo>/models/nr`.
The launcher finds Microsoft Edge automatically on Windows. Set `--edge` or
`EDGE` for another browser executable.

From the repository root:

```powershell
node benchmarks/opendlss-2026-10-30/opendlss_harness/run.mjs --help
node benchmarks/opendlss-2026-10-30/opendlss_harness/run.mjs --width 896 --height 512 --warmups 5 --iterations 20
```

Default input is `../inputs/street_896x512.png`, relative to this directory.
Default output is `../runs/opendlss_896x512`, separate from the archived results.
Use `--input`, `--out`, `--upstream`, and `--weights` to override these paths.
Relative command-line paths resolve against the current working directory.
The input dimensions must equal `--width` and `--height`; the harness does not
resize the image. The network's own internal geometry padding remains active.

The primary metric is synchronized wall time around `Network.run()` with
resident, prepared features. Model loading, image decoding, feature preparation,
readback, composition, and PNG encoding are outside this interval. The first
forward and warmups are recorded separately. GPU timestamp samples are a separate
optional series; failure to acquire them is reported in their own result field.

The browser checks that the head is finite, repeated calls produce identical
head bits, and no uncaptured WebGPU error or device loss occurred. A failed run
produces `status: "failed"` and a nonzero launcher exit code. The isolated browser
profile and loopback server are closed after the run. `results.json`, `output.png`,
`browser_input.png`, `launch.json`, `server_state.json`, and `edge.log` are saved
in the output directory.

## Identical-feature comparison with the native backend

Prepare the following files in the same directory:

- The RGB PNG passed with `--input`.
- `native_features.f32`: little-endian, contiguous float32 HWC, 16 channels.
- `native_head.f32`: little-endian, contiguous float32 HWC, 4 channels.

The dumps must come from the native backend for that exact PNG, seed 0, and the
same conditioning (`localTone=1`, `localStructure=1`, `skinStructure=-1`,
`style=0`, automatic mask enabled). This page requires a size whose native and
WebGPU full dimensions equal the requested size, such as 896 x 512.

```powershell
node benchmarks/opendlss-2026-10-30/opendlss_harness/run.mjs --page cross.html --input path/to/street.png --out path/to/cross-result
```

The page compares independently prepared native and WebGPU features, uploads the
saved native features, and compares the graph heads. It saves the resulting
WebGPU head to `head.f32` and includes tensor differences in `results.json`.
This is a numerical investigation with one cold call and two further calls;
those three durations are not a steady-state speed benchmark.

`status: "completed"` means this investigation executed successfully. It does
not mean native/WebGPU parity passed. Inspect
`same_feature_native_head_comparison.bitwise_equal` and
`preprocessing_comparison.whole_tensor.bitwise_equal` separately. The archived
RTX 5060 investigation found identical preprocessing and differing graph heads.
