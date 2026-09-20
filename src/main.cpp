// dlss5vk: standalone Vulkan DLSS-NR runner.
//   dlss5vk parity --model <nr model dir> --fixture <fixtures/nr512> [--shaders <dir>] [--dump <dir>]
//   dlss5vk bench  --model <nr model dir> --width W --height H [--frames N]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <array>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "json.h"
#include "kernels.h"
#include "nr_graph.h"
#include "nr_model.h"
#include "numeric.h"
#include "reference.h"
#include "vk_context.h"

namespace {

std::vector<uint8_t> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("cannot read " + path);
  std::streamsize size = file.tellg();
  file.seekg(0);
  std::vector<uint8_t> bytes((size_t)size);
  file.read(reinterpret_cast<char*>(bytes.data()), size);
  return bytes;
}

std::string readText(const std::string& path) {
  std::vector<uint8_t> bytes = readFile(path);
  return std::string(bytes.begin(), bytes.end());
}

std::string argValue(int argc, char** argv, const char* name, const std::string& fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return argv[i + 1];
  return fallback;
}

bool hasFlag(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], name)) return true;
  return false;
}

std::string executableDirectory(const char* argv0) {
  std::string path = argv0;
  size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? "." : path.substr(0, slash);
}

struct Comparison {
  size_t count = 0, mismatches = 0, signedZeros = 0;
  size_t firstIndex = SIZE_MAX;
  uint8_t firstActual = 0, firstExpected = 0;
  double maxAbs = 0, sumSquares = 0;
  std::map<int, size_t> codeDeltaHistogram;  // signed E4M3 code distance
};

Comparison compareE4(const std::vector<uint8_t>& actual, const std::vector<uint8_t>& expected) {
  Comparison c;
  c.count = expected.size();
  for (size_t i = 0; i < expected.size(); ++i) {
    uint8_t a = actual[i], e = expected[i];
    if (a == e) continue;
    // +0 and -0 decode to the same value and multiply identically, so a difference in the sign of a zero is
    // counted and reported but is not a mismatch.
    if ((a & 0x7f) == 0 && (e & 0x7f) == 0) { ++c.signedZeros; continue; }
    ++c.mismatches;
    if (c.firstIndex == SIZE_MAX) { c.firstIndex = i; c.firstActual = a; c.firstExpected = e; }
    double da = num::e4m3ToF32(a), de = num::e4m3ToF32(e);
    double diff = std::fabs(da - de);
    c.maxAbs = std::max(c.maxAbs, diff);
    c.sumSquares += diff * diff;
    int signedA = (a & 0x80) ? -(int)(a & 0x7f) : (int)(a & 0x7f);
    int signedE = (e & 0x80) ? -(int)(e & 0x7f) : (int)(e & 0x7f);
    int delta = signedA - signedE;
    delta = std::max(-9, std::min(9, delta));
    c.codeDeltaHistogram[delta]++;
  }
  return c;
}

// DLSS5VK_UNFUSED=1 runs the least fused form of the graph (the GLSL reference kernels).
static bool fusedBlocksEnabled() {
  const char* unfused = getenv("DLSS5VK_UNFUSED");
  return !(unfused && !strcmp(unfused, "1"));
}

int runParity(int argc, char** argv) {
  std::string modelDir = argValue(argc, argv, "--model");
  std::string fixtureDir = argValue(argc, argv, "--fixture");
  std::string shaderDir = argValue(argc, argv, "--shaders", executableDirectory(argv[0]) + "/shaders");
  std::string dumpDir = argValue(argc, argv, "--dump");
  if (modelDir.empty() || fixtureDir.empty()) {
    fprintf(stderr, "usage: dlss5vk parity --model <dir> --fixture <dir> [--shaders <dir>] [--dump <dir>]\n");
    return 2;
  }
  json::Value manifest = json::parse(readText(fixtureDir + "/manifest.json"));
  uint32_t validWidth = (uint32_t)manifest["sourceDimensions"][0].integer();
  uint32_t validHeight = (uint32_t)manifest["sourceDimensions"][1].integer();

  vk::Context context;
  printf("device: %s\n", context.deviceName().c_str());
  auto started = std::chrono::steady_clock::now();
  nr::Model model(context, modelDir, !hasFlag(argc, argv, "--no-verify"));
  auto loaded = std::chrono::steady_clock::now();
  printf("model loaded in %.2f s\n", std::chrono::duration<double>(loaded - started).count());
  nr::Kernels kernels(context, shaderDir);
  kernels.setSiluTable(ref::siluTable());
  nr::Geometry geometry = nr::Geometry::fromValid(validWidth, validHeight);
  printf("geometry %ux%u -> full %ux%u, levels", validWidth, validHeight, geometry.fullWidth, geometry.fullHeight);
  for (auto level : geometry.levels) printf(" %ux%u", level.width, level.height);
  printf("\n");
  if (geometry.fullWidth != (uint32_t)manifest["fullDimensions"][0].integer() ||
      geometry.fullHeight != (uint32_t)manifest["fullDimensions"][1].integer())
    throw std::runtime_error("fixture full dimensions disagree with the runtime profile");

  nr::Graph::Options options;
  options.captureBoundaries = true;
  options.fusedBlocks = fusedBlocksEnabled();
  nr::Graph graph(context, model, kernels, geometry, options);
  const uint32_t fullRows = geometry.fullWidth * geometry.fullHeight;
  nr::Activation* features = graph.allocate("input features", fullRows, 16, nr::Format::F32);
  vk::Buffer proxyBuffer;
  nr::Kernels::PreprocessArgs preprocess{};
  std::vector<uint8_t> proxyBytes;
  uint32_t sourceWidth = 0, sourceHeight = 0;
  if (manifest.has("proxy")) {
    const json::Value& proxy = manifest["proxy"];
    proxyBytes = readFile(fixtureDir + "/" + proxy["file"].str());
    sourceWidth = (uint32_t)proxy["width"].integer(); sourceHeight = (uint32_t)proxy["height"].integer();
    if (proxyBytes.size() != (size_t)sourceWidth * sourceHeight * 16) throw std::runtime_error("proxy size mismatch");
    proxyBuffer = context.createBuffer(proxyBytes.size(), false, "proxy");
    context.upload(proxyBuffer, proxyBytes.data(), proxyBytes.size());
    const json::Value& conditioning = manifest["conditioning"];
    preprocess = {geometry.fullWidth, geometry.fullHeight, validWidth, validHeight, sourceWidth, sourceHeight,
                  (uint32_t)manifest["seed"].integer(), manifest["autoMask"].boolean,
                  (float)conditioning["localTone"].number, (float)conditioning["localStructure"].number,
                  (float)conditioning["skinStructure"].number, (float)conditioning["style"].number};
  } else {
    std::vector<uint8_t> featureBytes = readFile(fixtureDir + "/" + manifest["inputFeatures"]["file"].str());
    if (featureBytes.size() != (size_t)fullRows * 16 * 4) throw std::runtime_error("input feature size mismatch");
    context.upload(features->buffer, featureBytes.data(), featureBytes.size());
  }

  auto recordStart = std::chrono::steady_clock::now();
  VkCommandBuffer commands = context.beginCommands();
  VkQueryPool queries = context.createTimestampPool(2);
  vkCmdResetQueryPool(commands, queries, 0, 2);
  vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
  if (proxyBuffer.buffer) kernels.preprocessFromProxy(commands, proxyBuffer, *features, preprocess);
  if (proxyBuffer.buffer && !dumpDir.empty()) {
    context.endAndSubmit(commands, true);
    std::vector<uint8_t> bytes = context.download(features->buffer, features->validBytes());
    std::ofstream out(dumpDir + "/features.f32", std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    commands = context.beginCommands();
  }
  graph.record(commands, *features);
  vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
  auto recorded = std::chrono::steady_clock::now();
  context.endAndSubmit(commands, true);
  auto finished = std::chrono::steady_clock::now();
  std::vector<double> stamps = context.readTimestampsMs(queries, 2);
  printf("recorded %u dispatches in %.1f ms (pipeline compilation included); GPU %.3f ms; wall %.1f ms\n",
         kernels.dispatchCount(), std::chrono::duration<double, std::milli>(recorded - recordStart).count(),
         stamps[1] - stamps[0], std::chrono::duration<double, std::milli>(finished - recorded).count());
  printf("NaN weight codes replaced: %zu\n", model.nanWeightsReplaced());

  size_t exactBoundaries = 0, totalBoundaries = 0, signedZeroTotal = 0;
  bool firstMismatchReported = false;
  auto compareEntry = [&](const std::string& name, const json::Value& entry) {
    auto it = graph.boundaries().find(name);
    if (it == graph.boundaries().end()) {
      printf("%-16s missing in graph\n", name.c_str());
      return;
    }
    const nr::Activation& activation = *it->second;
    std::vector<uint8_t> expected = readFile(fixtureDir + "/" + entry["file"].str());
    std::vector<uint8_t> actual = context.download(activation.buffer, activation.validBytes());
    if (actual.size() != expected.size()) {
      printf("%-16s size mismatch %zu vs %zu\n", name.c_str(), actual.size(), expected.size());
      return;
    }
    if (!dumpDir.empty()) {
      std::ofstream out(dumpDir + "/" + name + ".u8", std::ios::binary);
      out.write(reinterpret_cast<const char*>(actual.data()), actual.size());
    }
    Comparison c = compareE4(actual, expected);
    ++totalBoundaries;
    if (c.mismatches == 0) {
      ++exactBoundaries;
      printf("%-16s %ux%ux%u exact\n", name.c_str(), activation.rows / (uint32_t)entry["height"].integer(),
             (uint32_t)entry["height"].integer(), activation.channels);
    } else {
      uint32_t width = (uint32_t)entry["width"].integer();
      uint32_t channels = activation.channels;
      size_t pixel = c.firstIndex / channels;
      printf("%-16s MISMATCH %zu/%zu (%.4f%%) max|d| %.5f rmse %.6f first@ x%zu y%zu c%zu: got 0x%02x exp 0x%02x",
             name.c_str(), c.mismatches, c.count, 100.0 * c.mismatches / c.count, c.maxAbs,
             std::sqrt(c.sumSquares / c.count), pixel % width, pixel / width, c.firstIndex % channels,
             c.firstActual, c.firstExpected);
      if (!firstMismatchReported) {
        printf("  code-delta histogram:");
        for (auto& [delta, count] : c.codeDeltaHistogram) printf(" %+d:%zu", delta, count);
        firstMismatchReported = true;
      }
      printf("\n");
    }
  };
  // Report in graph order: block 0, transitions interleaved.
  std::vector<std::pair<double, std::pair<std::string, const json::Value*>>> ordered;
  for (const json::Value& entry : manifest["blocks"].array) {
    int block = (int)entry["block"].integer();
    ordered.push_back({(double)block, {"block-" + std::to_string(block), &entry}});
  }
  for (const json::Value& entry : manifest["transitions"].array) {
    std::string id = entry["id"].str();
    double from = atof(id.c_str());
    ordered.push_back({from + 0.5, {"transition-" + id, &entry}});
  }
  std::sort(ordered.begin(), ordered.end(), [](auto& a, auto& b) { return a.first < b.first; });
  for (auto& item : ordered) compareEntry(item.second.first, *item.second.second);
  printf("\n%zu / %zu boundaries exact\n", exactBoundaries, totalBoundaries);

  // Head: statistics, plus a bitwise comparison when the fixture carries a reference head.
  bool headExact = true;
  {
    const nr::Activation& head = graph.head();
    std::vector<uint8_t> bytes = context.download(head.buffer, head.validBytes());
    const float* values = reinterpret_cast<const float*>(bytes.data());
    size_t count = head.rows * head.channels;
    if (manifest.has("referenceHead")) {
      std::vector<uint8_t> reference = readFile(fixtureDir + "/" + manifest["referenceHead"]["file"].str());
      if (reference.size() != bytes.size()) throw std::runtime_error("reference head size mismatch");
      const float* expected = reinterpret_cast<const float*>(reference.data());
      size_t mismatches = 0, first = SIZE_MAX; double maxAbs = 0;
      for (size_t i = 0; i < count; ++i) {
        if (values[i] == expected[i]) continue;
        if (first == SIZE_MAX) first = i;
        ++mismatches;
        maxAbs = std::max(maxAbs, (double)std::fabs(values[i] - expected[i]));
      }
      headExact = mismatches == 0;
      if (headExact) printf("head vs reference: exact (%zu values)\n", count);
      else {
        size_t pixel = first / 4;
        printf("head vs reference: MISMATCH %zu/%zu (%.4f%%) max|d| %.6f first@ x%zu y%zu c%zu got %.8g exp %.8g\n",
               mismatches, count, 100.0 * mismatches / count, maxAbs, pixel % geometry.fullWidth,
               pixel / geometry.fullWidth, first % 4, values[first], expected[first]);
      }
    }
    if (manifest.has("nativeOutput") && !proxyBytes.empty()) {
      // The fixture's reference RGB output for the same proxy (one frame, no history): compose the head as the
      // demo composite does - neural = clamp((head / 32 + proxy / 8 - 1 / 16) * 8 + 0.5, 0, 1) in f32, published by
      // truncation to the half grid - and compare every RGB half.
      std::vector<uint8_t> nativeBytes = readFile(fixtureDir + "/" + manifest["nativeOutput"]["file"].str());
      if (nativeBytes.size() != (size_t)sourceWidth * sourceHeight * 16) throw std::runtime_error("native output size mismatch");
      const float* native = reinterpret_cast<const float*>(nativeBytes.data());
      const float* proxy = reinterpret_cast<const float*>(proxyBytes.data());
      auto truncateHalf = [](float value) {
        uint32_t bits; memcpy(&bits, &value, 4);
        uint32_t signBit = (bits >> 16) & 0x8000u, exponent = (bits >> 23) & 0xffu, mantissa = bits & 0x7fffffu, halfBits;
        if (exponent == 0xffu) halfBits = signBit | (mantissa ? 0x7e00u : 0x7c00u);
        else {
          int halfExponent = (int)exponent - 112;
          if (halfExponent >= 31) halfBits = signBit | 0x7c00u;
          else if (halfExponent <= 0) halfBits = halfExponent < -10 ? signBit : signBit | ((mantissa | 0x800000u) >> (14 - halfExponent));
          else halfBits = signBit | ((uint32_t)halfExponent << 10) | (mantissa >> 13);
        }
        return num::f16ToF32((uint16_t)halfBits);
      };
      size_t mismatches = 0, first = SIZE_MAX, fmaMismatches = 0; double maxAbs = 0;
      for (uint32_t y = 0; y < sourceHeight; ++y)
        for (uint32_t x = 0; x < sourceWidth; ++x)
          for (uint32_t c = 0; c < 3; ++c) {
            float h = values[((size_t)y * geometry.fullWidth + x) * 4 + c];
            float p = proxy[((size_t)y * sourceWidth + x) * 4 + c];
            float inner = std::fmaf(h, 0.03125f, std::fmaf(p, 0.125f, -0.0625f));
            float neural = std::fmin(std::fmax(inner * 8.0f + 0.5f, 0.0f), 1.0f);
            float published = truncateHalf(neural);
            float expected = native[((size_t)y * sourceWidth + x) * 4 + c];
            if (published != expected) {
              // the same with the last multiply-add contracted (both spellings are checked; the report says which)
              float alt = truncateHalf(std::fmin(std::fmax(std::fmaf(inner, 8.0f, 0.5f), 0.0f), 1.0f));
              if (alt == expected) { ++fmaMismatches; continue; }
              if (first == SIZE_MAX) first = ((size_t)y * sourceWidth + x) * 4 + c;
              ++mismatches;
              maxAbs = std::max(maxAbs, (double)std::fabs(published - expected));
            }
          }
      const size_t total = (size_t)sourceWidth * sourceHeight * 3;
      if (mismatches == 0) printf("composed RGB vs native output: exact (%zu halves%s)\n", total,
                                  fmaMismatches ? (", " + std::to_string(fmaMismatches) + " only with the contracted spelling").c_str() : "");
      else {
        size_t pixel = first / 4;
        printf("composed RGB vs native output: MISMATCH %zu/%zu (%.4f%%) max|d| %.6f first@ x%zu y%zu c%zu\n", mismatches, total,
               100.0 * mismatches / total, maxAbs, pixel % sourceWidth, pixel / sourceWidth, first % 4);
        headExact = false;
      }
    }
    double sum = 0, sumSquares = 0; size_t nonFinite = 0;
    for (size_t i = 0; i < count; ++i) {
      if (!std::isfinite(values[i])) { ++nonFinite; continue; }
      sum += values[i]; sumSquares += (double)values[i] * values[i];
    }
    printf("head: %zu values, mean %.6f, rms %.6f, non-finite %zu\n", count, sum / count, std::sqrt(sumSquares / count), nonFinite);
    if (!dumpDir.empty()) {
      std::ofstream out(dumpDir + "/head.f32", std::ios::binary);
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  vkDestroyQueryPool(context.device(), queries, nullptr);
  if (proxyBuffer.buffer) context.destroyBuffer(proxyBuffer);
  return exactBoundaries == totalBoundaries && headExact ? 0 : 1;
}

int runBench(int argc, char** argv) {
  std::string modelDir = argValue(argc, argv, "--model");
  std::string shaderDir = argValue(argc, argv, "--shaders", executableDirectory(argv[0]) + "/shaders");
  uint32_t width = (uint32_t)atoi(argValue(argc, argv, "--width", "768").c_str());
  uint32_t height = (uint32_t)atoi(argValue(argc, argv, "--height", "768").c_str());
  int frames = atoi(argValue(argc, argv, "--frames", "10").c_str());
  if (modelDir.empty()) { fprintf(stderr, "usage: dlss5vk bench --model <dir> [--width W --height H --frames N]\n"); return 2; }
  vk::Context context;
  printf("device: %s\n", context.deviceName().c_str());
  nr::Model model(context, modelDir, false);
  nr::Kernels kernels(context, shaderDir);
  kernels.setSiluTable(ref::siluTable());
  nr::Geometry geometry = nr::Geometry::fromValid(width, height);
  nr::Graph graph(context, model, kernels, geometry, {.fusedBlocks = fusedBlocksEnabled()});
  const uint32_t fullRows = geometry.fullWidth * geometry.fullHeight;
  nr::Activation* features = graph.allocate("input features", fullRows, 16, nr::Format::F32);
  std::vector<float> synthetic((size_t)fullRows * 16);
  for (size_t i = 0; i < synthetic.size(); ++i) synthetic[i] = num::roundF16(std::sin(i * 0.0017f) * 0.125f);
  context.upload(features->buffer, synthetic.data(), synthetic.size() * 4);
  // Warm compile.
  {
    VkCommandBuffer commands = context.beginCommands();
    graph.record(commands, *features);
    context.endAndSubmit(commands, true);
  }
  VkQueryPool queries = context.createTimestampPool(2);
  std::vector<double> samples;
  for (int frame = 0; frame < frames; ++frame) {
    context.resetDescriptorPool();
    VkCommandBuffer commands = context.beginCommands();
    vkCmdResetQueryPool(commands, queries, 0, 2);
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
    graph.record(commands, *features);
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
    context.endAndSubmit(commands, true);
    std::vector<double> stamps = context.readTimestampsMs(queries, 2);
    samples.push_back(stamps[1] - stamps[0]);
    printf("frame %d: %.3f ms GPU (%u dispatches)\n", frame, samples.back(), kernels.dispatchCount());
  }
  std::sort(samples.begin(), samples.end());
  printf("median %.3f ms, min %.3f ms over %d frames at %ux%u (full %ux%u)\n", samples[samples.size() / 2],
         samples.front(), frames, width, height, geometry.fullWidth, geometry.fullHeight);
  vkDestroyQueryPool(context.device(), queries, nullptr);
  return 0;
}

}  // namespace

int runVerify(int argc, char** argv);

namespace {
int runShaderInfo(int argc, char** argv) {
  std::string modelDir = argValue(argc, argv, "--model");
  std::string shaderDir = argValue(argc, argv, "--shaders", executableDirectory(argv[0]) + "/shaders");
  std::string filter = argValue(argc, argv, "--filter", "");
  bool sass = hasFlag(argc, argv, "--sass");
  if (modelDir.empty()) { fprintf(stderr, "usage: dlss5vk shaderinfo --model <dir> [--filter name] [--sass]\n"); return 2; }
  vk::Context context;
  context.setCaptureStatistics(true);
  nr::Model model(context, modelDir, false);
  nr::Kernels kernels(context, shaderDir);
  kernels.setSiluTable(ref::siluTable());
  nr::Geometry geometry = nr::Geometry::fromValid(768, 768);
  nr::Graph graph(context, model, kernels, geometry, {.fusedBlocks = fusedBlocksEnabled()});
  nr::Activation* features = graph.allocate("input features", geometry.fullWidth * geometry.fullHeight, 16, nr::Format::F32);
  VkCommandBuffer commands = context.beginCommands();
  graph.record(commands, *features);
  context.endAndSubmit(commands, true);
  for (const auto& [key, pipeline] : kernels.pipelines()) {
    if (!filter.empty() && key.find(filter) == std::string::npos) continue;
    printf("== %s\n%s\n", key.c_str(), context.pipelineStatistics(pipeline, sass).c_str());
  }
  return 0;
}
}  // namespace

namespace {
int runProfile(int argc, char** argv) {
  std::string modelDir = argValue(argc, argv, "--model");
  std::string shaderDir = argValue(argc, argv, "--shaders", executableDirectory(argv[0]) + "/shaders");
  uint32_t width = (uint32_t)atoi(argValue(argc, argv, "--width", "768").c_str());
  uint32_t height = (uint32_t)atoi(argValue(argc, argv, "--height", "768").c_str());
  if (modelDir.empty()) { fprintf(stderr, "usage: dlss5vk profile --model <dir> [--width W --height H]\n"); return 2; }
  if (!getenv("DLSS5VK_CHAIN")) nr::Kernels::setChainEnabled(false);   // per-dispatch timings need the barriers
  vk::Context context;
  printf("maxComputeSharedMemorySize %u bytes\n", context.maxComputeSharedMemory());
  nr::Model model(context, modelDir, false);
  nr::Kernels kernels(context, shaderDir);
  kernels.setSiluTable(ref::siluTable());
  // Barrier/dispatch overhead: 400 trivial dispatches each followed by a full compute barrier.
  {
    nr::Activation tiny; tiny.format = nr::Format::F16; tiny.rows = 64; tiny.channels = 16; tiny.allocRows = 64;
    tiny.buffer = context.createBuffer(64 * 16 * 2, false, "tiny");
    nr::Activation tinyOut = tiny; tinyOut.format = nr::Format::E4;
    tinyOut.buffer = context.createBuffer(64 * 16, false, "tiny out");
    VkQueryPool pool = context.createTimestampPool(2);
    VkCommandBuffer commands = context.beginCommands();
    vkCmdResetQueryPool(commands, pool, 0, 2);
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, 0);
    for (int i = 0; i < 400; ++i) kernels.quantize(commands, tiny, tinyOut);
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 1);
    context.endAndSubmit(commands, true);
    std::vector<double> stamps = context.readTimestampsMs(pool, 2);
    printf("400 trivial dispatches + barriers: %.3f ms (%.2f us each)\n", stamps[1] - stamps[0], (stamps[1] - stamps[0]) * 2.5);
    vkDestroyQueryPool(context.device(), pool, nullptr);
    context.destroyBuffer(tiny.buffer); context.destroyBuffer(tinyOut.buffer);
    context.resetDescriptorPool();
  }
  nr::Geometry geometry = nr::Geometry::fromValid(width, height);
  nr::Graph graph(context, model, kernels, geometry, {.fusedBlocks = fusedBlocksEnabled()});
  const uint32_t fullRows = geometry.fullWidth * geometry.fullHeight;
  nr::Activation* features = graph.allocate("input features", fullRows, 16, nr::Format::F32);
  std::vector<float> synthetic((size_t)fullRows * 16);
  for (size_t i = 0; i < synthetic.size(); ++i) synthetic[i] = num::roundF16(std::sin(i * 0.0017f) * 0.125f);
  context.upload(features->buffer, synthetic.data(), synthetic.size() * 4);
  {
    VkCommandBuffer commands = context.beginCommands();
    graph.record(commands, *features);
    context.endAndSubmit(commands, true);
  }
  std::map<std::string, double> byLabel;
  std::map<std::string, double> byStage;
  double total = 0;
  // Other programs share the GPU: take the per-dispatch minimum over several frames.
  const int frames = atoi(argValue(argc, argv, "--frames", "7").c_str());
  std::vector<nr::Kernels::ProfileEntry> best;
  for (int frame = 0; frame < frames; ++frame) {
    context.resetDescriptorPool();
    VkCommandBuffer commands = context.beginCommands();
    kernels.beginProfile(commands, 4096);
    graph.record(commands, *features);
    context.endAndSubmit(commands, true);
    std::vector<nr::Kernels::ProfileEntry> entries = kernels.endProfile();
    if (best.empty()) best = entries;
    for (size_t i = 0; i < entries.size() && i < best.size(); ++i)
      best[i].milliseconds = std::min(best[i].milliseconds, entries[i].milliseconds);
  }
  const std::string stageFilter = argValue(argc, argv, "--stage", "");
  for (size_t index = 0; index < best.size(); ++index) {
    const auto& entry = best[index];
    size_t bar = entry.label.find(" | ");
    std::string stage = entry.label.substr(0, bar), kernel = entry.label.substr(bar + 3);
    if (!stageFilter.empty() && stage.find(stageFilter) != std::string::npos)
      printf("  %8.2f us  [%3zu] %s | %s\n", entry.milliseconds * 1000.0, index, stage.c_str(), kernel.c_str());
    byLabel[kernel] += entry.milliseconds;
    byStage[stage] += entry.milliseconds;
    total += entry.milliseconds;
  }
  std::vector<std::pair<std::string, double>> sorted(byLabel.begin(), byLabel.end());
  std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });
  printf("total %.3f ms (sum of per-dispatch spans, %ux%u)\n\nby kernel:\n", total, width, height);
  for (size_t i = 0; i < sorted.size() && i < 40; ++i)
    printf("  %8.3f ms  %5.1f%%  %s\n", sorted[i].second, 100 * sorted[i].second / total, sorted[i].first.c_str());
  std::vector<std::pair<std::string, double>> stages(byStage.begin(), byStage.end());
  std::sort(stages.begin(), stages.end(), [](auto& a, auto& b) { return a.second > b.second; });
  printf("\nby stage:\n");
  for (size_t i = 0; i < stages.size(); ++i)
    printf("  %8.3f ms  %5.1f%%  %s\n", stages[i].second, 100 * stages[i].second / total, stages[i].first.c_str());
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && !strcmp(argv[1], "verify")) return runVerify(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "shaderinfo")) return runShaderInfo(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "profile")) return runProfile(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "parity")) return runParity(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "bench")) return runBench(argc, argv);
    fprintf(stderr, "usage: dlss5vk parity|verify|bench|profile|shaderinfo --model <dir> ...\n");
    return 2;
  } catch (const std::exception& error) {
    fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
