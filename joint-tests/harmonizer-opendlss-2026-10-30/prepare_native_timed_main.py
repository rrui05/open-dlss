"""Generate an untracked native timing entry without changing OpenDLSS sources.

Only the benchmark input source, warmup count and timing/logging are changed.
The original graph, model loader, shader sources, and PTX kernels are reused.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT.parents[1] / "open-dlss"
source = UPSTREAM / "src/main.cpp"
text = source.read_text(encoding="utf-8")
start = text.index("int runBench(int argc, char** argv) {")
end = text.index("\n}  // namespace", start)
bench = text[start:end]


def replace_once(old, new):
    global bench
    if bench.count(old) != 1:
        raise RuntimeError(f"Expected one benchmark anchor: {old[:100]!r}")
    bench = bench.replace(old, new)


replace_once('argValue(argc, argv, "--frames", "10")', 'argValue(argc, argv, "--frames", "20")')
replace_once('  vk::Context context;', '  const int warmups = std::max(0, atoi(argValue(argc, argv, "--warmups", "5").c_str()));\n  vk::Context context;')
replace_once(
    '  context.upload(features->buffer, synthetic.data(), synthetic.size() * 4);',
    '''  const std::string featureFile = argValue(argc, argv, "--features-file");
  if (!featureFile.empty()) {
    const std::vector<uint8_t> bytes = readFile(featureFile);
    if (bytes.size() != synthetic.size() * sizeof(float))
      throw std::runtime_error("features-file size does not match geometry");
    std::memcpy(synthetic.data(), bytes.data(), bytes.size());
  }
  context.upload(features->buffer, synthetic.data(), synthetic.size() * 4);
  printf("INPUT: %s\\n", featureFile.empty() ? "synthetic" : featureFile.c_str());''',
)
replace_once(
    '  for (int frame = 0; frame < frames; ++frame) {\n    context.resetDescriptorPool();',
    '  for (int frame = -warmups; frame < frames; ++frame) {\n    const auto wallStarted = std::chrono::steady_clock::now();\n    context.resetDescriptorPool();',
)
replace_once(
    '    context.endAndSubmit(commands, true);\n    checkChainTimeouts(kernels);\n    std::vector<double> stamps',
    '''    context.endAndSubmit(commands, true);
    const double wallMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wallStarted).count();
    checkChainTimeouts(kernels);
    std::vector<double> stamps''',
)
replace_once(
    '    samples.push_back(stamps[1] - stamps[0]);\n    printf("frame %d: %.3f ms GPU (%u dispatches)\\n", frame, samples.back(), kernels.dispatchCount());',
    '''    const double gpuMs = stamps[1] - stamps[0];
    if (frame < 0) {
      printf("WARMUP: %d GPU %.9f ms, wall %.9f ms\\n", frame + warmups + 1, gpuMs, wallMs);
    } else {
      samples.push_back(gpuMs);
      printf("SAMPLE: {\\"index\\":%d,\\"cuda_ptx_gpu_ms\\":%.9f,\\"synchronized_wall_ms\\":%.9f,\\"dispatches\\":%u}\\n",
             frame, gpuMs, wallMs, kernels.dispatchCount());
    }''',
)
generated = text[:start] + bench + text[end:]
target = ROOT / "native_timed/main_timed.cpp"
target.parent.mkdir(exist_ok=True)
target.write_text(generated, encoding="utf-8")
metadata = {
    "original": str(source), "original_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "generated": str(target), "generated_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    "changes": ["Optional resident input-features file", "Explicit warmup count", "Per-frame synchronized wall time and GPU timestamps"],
    "network_math_changed": False,
}
(target.parent / "source_provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(target)
