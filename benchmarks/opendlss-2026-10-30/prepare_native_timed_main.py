"""Generate a timing-only native entry point; reuse the repository's graph/kernels.

Generated C++ and provenance go under the ignored build/ directory. The source
network is not edited. Fails if an expected source anchor has changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]


def generate(repository: Path, output: Path) -> Path:
    source = repository / "src/main.cpp"
    text = source.read_text(encoding="utf-8")
    start = text.index("int runBench(int argc, char** argv) {")
    end = text.index("\n}  // namespace", start)
    bench = text[start:end]

    def replace_once(old: str, new: str) -> None:
        nonlocal bench
        if bench.count(old) != 1:
            raise RuntimeError(f"Expected one benchmark anchor: {old[:100]!r}")
        bench = bench.replace(old, new)

    replace_once('argValue(argc, argv, "--frames", "10")', 'argValue(argc, argv, "--frames", "20")')
    replace_once('  vk::Context context;', '''  const int warmups = atoi(argValue(argc, argv, "--warmups", "5").c_str());
  const uint32_t seed = (uint32_t)strtoul(argValue(argc, argv, "--seed", "0").c_str(), nullptr, 10);
  const std::string dumpDir = argValue(argc, argv, "--dump");
  const std::string featureFile = argValue(argc, argv, "--features-file");
  const std::string proxyFile = argValue(argc, argv, "--proxy-file");
  if (width < 33 || height < 33 || frames < 1 || warmups < 0)
    throw std::runtime_error("invalid width/height/frames/warmups");
  if (featureFile.empty() == proxyFile.empty())
    throw std::runtime_error("choose exactly one of --features-file and --proxy-file");
  if (!dumpDir.empty()) std::filesystem::create_directories(dumpDir);
  vk::Context context;''')
    replace_once('  nr::Model model(context, modelDir, false);', '  nr::Model model(context, modelDir, true);')
    replace_once('  context.upload(features->buffer, synthetic.data(), synthetic.size() * 4);', '''  vk::Buffer proxyBuffer;
  if (!featureFile.empty()) {
    const std::vector<uint8_t> bytes = readFile(featureFile);
    if (bytes.size() != synthetic.size() * sizeof(float))
      throw std::runtime_error("features-file size does not match padded geometry");
    std::memcpy(synthetic.data(), bytes.data(), bytes.size());
    context.upload(features->buffer, synthetic.data(), synthetic.size() * sizeof(float));
  } else {
    const std::vector<uint8_t> bytes = readFile(proxyFile);
    if (bytes.size() != (size_t)width * height * 4 * sizeof(float))
      throw std::runtime_error("proxy-file size must match valid RGBA f32 geometry");
    proxyBuffer = context.createBuffer(bytes.size(), false, "benchmark proxy");
    context.upload(proxyBuffer, bytes.data(), bytes.size());
    nr::Kernels::PreprocessArgs preprocess{geometry.fullWidth, geometry.fullHeight, width, height,
                                         width, height, seed, true, 1, 1, -1, 0};
    VkCommandBuffer commands = context.beginCommands();
    kernels.preprocessFromProxy(commands, proxyBuffer, *features, preprocess);
    context.endAndSubmit(commands, true);
  }
  printf("CONFIG: {\\"width\\":%u,\\"height\\":%u,\\"full_width\\":%u,\\"full_height\\":%u,\\"warmups\\":%d,\\"iterations\\":%d,\\"seed\\":%u}\\n",
         width, height, geometry.fullWidth, geometry.fullHeight, warmups, frames, seed);
  if (!dumpDir.empty()) {
    const auto bytes = context.download(features->buffer, features->validBytes());
    std::ofstream file(dumpDir + "/features.f32", std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    if (!file) throw std::runtime_error("cannot write features dump");
  }''')
    replace_once('  for (int frame = 0; frame < frames; ++frame) {\n    context.resetDescriptorPool();',
                 '  for (int frame = -warmups; frame < frames; ++frame) {\n    const auto wallStarted = std::chrono::steady_clock::now();\n    context.resetDescriptorPool();')
    replace_once('    context.endAndSubmit(commands, true);\n    checkChainTimeouts(kernels);\n    std::vector<double> stamps',
                 '''    context.endAndSubmit(commands, true);
    const double wallMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wallStarted).count();
    checkChainTimeouts(kernels);
    std::vector<double> stamps''')
    replace_once('    samples.push_back(stamps[1] - stamps[0]);\n    printf("frame %d: %.3f ms GPU (%u dispatches)\\n", frame, samples.back(), kernels.dispatchCount());',
                 '''    const double gpuMs = stamps[1] - stamps[0];
    if (frame < 0) {
      printf("WARMUP: %d GPU %.9f ms, wall %.9f ms\\n", frame + warmups + 1, gpuMs, wallMs);
    } else {
      samples.push_back(gpuMs);
      printf("SAMPLE: {\\"index\\":%d,\\"cuda_ptx_gpu_ms\\":%.9f,\\"synchronized_wall_ms\\":%.9f,\\"dispatches\\":%u}\\n",
             frame, gpuMs, wallMs, kernels.dispatchCount());
    }''')
    replace_once('  std::sort(samples.begin(), samples.end());', '''  const auto head = context.download(graph.head().buffer, graph.head().validBytes());
  size_t nonFinite = 0;
  for (size_t i = 0; i < head.size() / sizeof(float); ++i) {
    float value;
    std::memcpy(&value, head.data() + i * sizeof(float), sizeof(float));
    nonFinite += !std::isfinite(value);
  }
  printf("VALIDATION: {\\"head_elements\\":%zu,\\"head_non_finite\\":%zu}\\n", head.size() / sizeof(float), nonFinite);
  if (!dumpDir.empty()) {
    std::ofstream file(dumpDir + "/head.f32", std::ios::binary);
    file.write(reinterpret_cast<const char*>(head.data()), head.size());
    if (!file) throw std::runtime_error("cannot write head dump");
  }
  if (proxyBuffer.buffer) context.destroyBuffer(proxyBuffer);
  if (nonFinite) throw std::runtime_error("non-finite network head");
  std::sort(samples.begin(), samples.end());''')
    generated = text[:start] + bench + text[end:]
    main_anchor = "int main(int argc, char** argv) {"
    if generated.count(main_anchor) != 1:
        raise RuntimeError("Expected one main entry point")
    generated = generated.replace(main_anchor, main_anchor + "\n  setvbuf(stdout, nullptr, _IONBF, 0);\n  setvbuf(stderr, nullptr, _IONBF, 0);")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "main_timed.cpp"
    target.write_text(generated, encoding="utf-8")
    metadata = {
        "original": str(source), "original_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "generated": str(target), "generated_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "network_math_changed": False,
        "changes": ["Resident real features or original preprocessing outside timing", "Configurable warmups and measured frames",
                    "Synchronized wall and GPU timestamps", "Untimed feature/head dumps and finite check", "Unbuffered diagnostics", "Verify packed weights"],
    }
    (output / "source_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument("--output", type=Path, default=REPOSITORY / "build/benchmark_native")
    args = parser.parse_args()
    print(generate(args.repository_root.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
