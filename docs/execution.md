# Execution: Vulkan resources, kernels, scheduling

## The Vulkan surface

`vk::Context` is deliberately small: one device, one compute queue, storage buffers only, no images, no render
passes. The demo hands it a device created by Filament instead of creating one (`vk::DeviceRequirements` is the
feature/extension chain both paths use, so the renderer's device carries what the kernels need).

* **Descriptors.** One layout, **12 storage buffers**, every kernel binding into the same slots; unbound
  slots get a shared dummy buffer. Two pools in rotation, so a frame's sets stay valid while the next is
  recorded.
* **Push constants.** 128 bytes, one struct per kernel.
* **Buffers.** Device-local, always with `SHADER_DEVICE_ADDRESS` (the PTX kernels take raw addresses), created
  through one 256 MiB host-visible staging window.
* **Barriers.** Compute to compute only (`computeBarrier`). Including the transfer stages made the driver
  flush caches between every dispatch, tens of microseconds each once L2 is dirty. Captures and uploads use
  `transferBarrier` explicitly.
* **Subgroups.** Every pipeline is created with `requiredSubgroupSize = 32` and full subgroups; the
  cooperative-matrix fragment layouts assume it.
* **PTX.** `VK_NV_cuda_kernel_launch`: the generated `.ptx` text becomes a `VkCudaModuleNV`, the driver JITs
  it, and launches go into the same command buffer as the dispatches.

Extensions required: `VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2`, `VK_EXT_shader_float8`,
`VK_NV_cuda_kernel_launch`, plus `VK_KHR_pipeline_executable_properties`, `VK_KHR_shader_clock` and
`VK_NV_shader_sm_builtins` for tooling.

## Activations

```cpp
struct Activation { vk::Buffer buffer; Format format; uint32_t rows, channels, allocRows; };
```

Row-major `[rows][channels]`, rows padded to a multiple of 64 (`allocRows`) so a 16-row cooperative-matrix load
near the end of a tensor never leaves the allocation, and zero-filled once at creation so those padding rows are
zeros rather than garbage. Formats: `E4` (1 byte), `F16`, `F32`.

Native stores activations in **4x4 pixel tiles** of 16 tokens instead. That is pure addressing: the arithmetic
is per-token, the attention's token order is handled explicitly, and the exported fixtures are de-tiled once.
The one place the tiling leaks into this port is the attention prior, whose key axis stays in tiled order
([weights.md](weights.md#the-attention-prior)).

`Graph::allocate` keys buffers by `label/rows x channels/format` and reuses them across re-records; recording
the graph again (every frame, or after a resize) allocates nothing. `usedThisRecord_` catches a label collision
inside one recording, which would silently alias two tensors.

## The kernel set

Two complete routes compute the same bytes. The GLSL route is the specification; the PTX route is what runs.

| GLSL (`shaders/`) | PTX (`scripts/ptx/`) | what it does |
| --- | --- | --- |
| `gemm_fp8.comp` | `gemm2_e4m3.py`, `gemmt_e4m3.py`, `gemmv_e4m3.py`, `reduce_e4m3.py` | the FP8 GEMM, with the residual prologue and the SiLU / quantize epilogue; split-K plus a reduce for the few-row ViT shapes |
| `gemm_mlp.comp` | `mlp_e4m3.py` | two GEMMs with the hidden layer on chip |
| `fused_block32.comp` | `block32_e4m3.py` | a whole 32-channel block, one window per workgroup slot |
| `qkv_attention.comp` | `qkv_e4m3.py` | QKV projection + normalize + window attention per (window, head) |
| `global_attention.comp`, `global_normalize.comp`, `global_attend.comp` | `global_attention_e4m3.py`, `global_attention_stream_e4m3.py` | the ViT's attention; the streamed variant handles any token count |
| `gemm_f16.comp` | - | the input adapter and the RGBA head |
| `ops.comp` | - | f32 to f16, quantize, 2x2 pool, the post blend, the upsample + scaled skip merge |
| `preprocess.comp` | - | input features from a proxy image (the tool; the demo has its own) |
| `gemm_reduce.comp` | - | split-K partial sums |

`DLSS5VK_UNFUSED=1` runs the least fused form (separate GEMM, normalize and attend per block); every
`DLSS5VK_PTX_*=0` switch sends one family back to GLSL. `parity` runs under each of them, and all of them
produce identical bytes; without that gate the off-by-default kernels rot, as two of them had.

The unfused route has a ceiling the fused one does not: it materializes every intermediate, so its activation
working set is several gigabytes at 4K, and `global_attend.comp` keeps the whole score matrix in shared memory
(`48 * paddedTokens` bytes, against a 48 KB limit). It is exercised up to 2560x1440; above that only the fused
routes run.

### Fusion levels

The same block can be spelled at four levels of fusion. From least to most:

1. separate GEMM / normalize / attend dispatches (`DLSS5VK_UNFUSED=1`), ~538 dispatches at 512x512;
2. fused QKV+attention and fused MLP, ~317;
3. the whole 32-channel block in one dispatch, the expert FFN+W3 in one, **241** dispatches, the default;
4. plus the optional input adapter (`F_PRE`), pool (`F_POOL`), upsample merge (`F_UPRES`), post blend
   (`F_POST`) and head (`F_HEAD`) folded into the neighbouring block, and the previous block's projection folded
   into the next block's FFN kernel.

The reference implementation fuses the same set - the pool into the block that produces it, the decoder's
upsample into the block that consumes it, the input adapter and the post blend into blocks 0 and 70 - and works
one 8x8 window per 32-thread group. The fusion boundaries are where the specification already puts a
publication.

### The PTX generators

`scripts/ptx/` is a Python PTX emitter (`ptxgen.py`, ~130 lines: virtual registers, an entry-point header, and
loops unrolled in Python so every shared-memory offset and fragment index is a literal) plus `swin.py`, which
holds the network's arithmetic once: `mma_e4` / `mma_f16`, `silu`, `normalize`, `exp_weight`, `softmax`,
`tiled_token_regs`, the accumulator-to-A-operand relayouts, and the chaining primitives. Each kernel file is a
few hundred lines of scheduling on top of that. The build emits one variant per shape the graph asks for
(`gemm2_e4m3_K256_f5`, `ffn_e4m3_C128_R4_proj`, ...) into `build/ptx`, and `Kernels::ptxKernel` JITs them on
first use.

They exist because GLSL cannot express what the hot path needs: `cp.async` rings, `mma.sync` with explicit
fragment placement, `.maxnreg` control, `nanosleep` backoff, and, most of all, release/acquire traffic between
launches. The target is `sm_89`; the arithmetic is the same instruction the native kernels use, so the two
routes agree bit for bit and the GLSL route stays the readable specification.

Every publication rule lives once, in `swin.py`, and the kernels call it: there is no second spelling of the
SiLU polynomial or of the softmax reduction anywhere in `scripts/ptx/`. The generated variant set is enumerated
in `scriptsuild_shaders.ps1`; a shape the graph asks for and the build did not generate is a hard error at
record time for `gemm2`, `block32`, `qkv`, `ffn` and `mlp`, and a fall-back to another route for `gemmv` and
`gemmt`, which the chaining guards then make loud as well.

## Barrier-free chaining

A compute barrier between two dispatches costs about a microsecond and, more importantly, drains the GPU. For
the long chains of small launches in this graph that is most of the frame. So consecutive **PTX** launches are
chained: the producer increments a device counter after its stores, the consumer spins on that counter, and no
barrier is recorded between them.

Counters live in one zeroed-per-frame buffer, `kSyncSlotBytes = 6144` per block slot, three regions:

| region | meaning | granularity |
| --- | --- | --- |
| `kSyncBands` | the FFN output | one per row band per workgroup |
| `kSyncRows` | the attention output | one per window row, per head |
| `kSyncState` | the projection output = the next block's state | one per (row group, column group) |

A consumer knows both the counter address and the count to wait for, which is why `Chain` carries
`waitExpected`, `waitShiftY` (the producer's window phase, needed to map its rows to the consumer's) and
`waitScale` (whether the producer ran at twice, the same, or half the resolution).

The primitives are in `swin.py`: a producer ends with `fence.acq_rel.gpu`, a `bar.sync`, then thread 0 doing
`red.release.gpu.global.add.u32` on each counter; a consumer polls with `ld.acquire.gpu.global.u32` plus
`vote.sync.all` and a `nanosleep` backoff from 128 ns to 1 us. Release/acquire at device scope is what makes the
producer's stores visible, not the counter itself.

Two properties make this safe:

* **no cycles**: every wait is on a launch recorded earlier in the same command buffer, and the chains are
  linear (FFN -> attention -> projection -> next FFN);
* **co-residency**: the chained kernels are persistent grids sized from `smCount()` (8 workgroups per SM for
  the 32-channel block, 12 for the QKV kernel, 2 for the GLSL fused block), so a consumer cannot fill the GPU
  ahead of the producer it is spinning on.

The third property is the one that is easy to get wrong: **a chained launch must actually be a PTX launch.** The
GLSL kernels neither wait nor signal, so arming a chain around one either hangs the GPU (a consumer spinning on
a counter nobody increments) or races (a producer with no barrier after it). `Graph::Routes::chain` therefore
decides once, from every route the chain links, and the three GLSL kernels that could be reached by a chained
call reject one with a `check()`.

One capacity bound comes out of this: a counter is indexed by pixel-row band (`row / 8`) or by window row, and a
region holds `kSyncCountersPerRegion = 512` of them, so chaining covers field heights up to 4080. Above that the
graph says so and falls back to barriers.

`DLSS5VK_CHAIN=0` replaces the whole mechanism with barriers; `DLSS5VK_CHAIN_MASK` selects which chains are
armed. Both keep the output identical. Chaining is a scheduling choice, not arithmetic.

## Initialization and teardown

In order, with the costs measured at 512x512 on the reference machine:

| step | what it does | cost |
| --- | --- | --- |
| `vk::Context` | instance, pick the first device with the kernels' extensions, device + queue, command pool, the descriptor layout and two pools, a 256 MiB staging buffer, a dummy buffer | ms |
| `nr::Model` | parse `manifest.json`, read the eleven stage files (141 MiB), optionally SHA-256 them, upload each tensor's raw bytes | 0.25 s, or 1.10 s with hash verification |
| `nr::Kernels` | load the thirteen SPIR-V modules | ms |
| `setSiluTable` | build the 65536-entry f16 SiLU table on the CPU and upload it | ms |
| `nr::Graph` | resolve the routes from the environment, check the model's block count | - |
| first `record()` | allocate and zero every activation, build every weight matrix the graph asks for, compile every pipeline specialization, JIT every PTX module | 0.6-0.9 s |
| later `record()` | reuses all of it | allocates nothing |

Everything after the first recording is steady state: the graph is re-recorded per frame (the demo re-records
only when the history parity or the NR on/off flag changes, into secondary command buffers) and no Vulkan
object is created.

Teardown runs in reverse and is explicit, because in the demo the device outlives all of it: `Graph` destroys
its activations; `Kernels` destroys its pipelines, its CUDA functions and modules, the sync counters, the
split-K scratch, the SiLU table and any profiling pool; `Model` destroys the re-laid-out matrices and the raw
tensor buffers; `vk::Context` waits for the device to be idle, destroys the shader modules, the staging and
dummy buffers, the pools and the layouts, and the device and instance only when it created them.

## Lifetimes and ownership

| object | owner | lifetime |
| --- | --- | --- |
| weight and prior buffers | `Model` | load to shutdown, immutable |
| activations, boundary captures | `Graph` | graph construction to destruction; reused across re-records |
| sync counters, split-K scratch, SiLU table | `Kernels` | first use to destruction |
| descriptor sets | `vk::Context` pools | until the pool is reset; the demo resets the pool of the frame parity it is about to record |
| PTX modules / functions | `Kernels::ptxKernels_` | first use to destruction, keyed by file |

Two sharp edges, both commented in the code: the split-K scratch buffer's **device address is baked into
already-recorded launches**, so it is sized once and never reallocated (the code checks that a later, larger
split still fits); and `Kernels::tileCounters` hands out per-frame counter slots from a cursor that
`resetDispatchCount()` rewinds at the top of every recording.

## Failure modes

| symptom | cause |
| --- | --- |
| `no device with VK_KHR_cooperative_matrix + VK_EXT_shader_float8` | wrong GPU or driver |
| `cannot read PTX kernel ...` | `scripts/build.ps1` did not run, or `DLSS5VK_PTX_DIR` points elsewhere |
| `chained GEMM has no PTX route` / `... does not implement counter chaining` | a chain was armed around a GLSL kernel; a route predicate is out of step with `Graph::Routes` |
| `VK_ERROR_DEVICE_LOST` in a chained run | a consumer is spinning on a counter that is never signalled, the same bug one step later |
| `split-K partials exceed the scratch buffer` | a larger split reached the fixed scratch after it was sized |
| `stage SHA-256 mismatch` | a corrupt or mismatched model directory |
| `fixture full dimensions disagree with the runtime profile` | the fixture was made with a different field rule |

## Performance

RTX 4070 SUPER (56 SMs), whole network per frame, **minimum over 40 frames**. The GPU alternates between two
clock states under sustained load, so the minimum is the number that reproduces; medians run 3-5% higher.

| resolution | field | time | GFLOP / frame | achieved | dispatches |
| --- | --- | --- | --- | --- | --- |
| 512x512 | 576x512 | 2.72 ms | 139 | 51 TFLOP/s | 241 |
| 768x768 | 832x768 | 2.83 ms | 297 | 105 TFLOP/s | 241 |
| 1920x1080 | 1920x1152 | 7.77 ms | 1021 | 131 TFLOP/s | 241 |
| 2560x1440 | 2560x1472 | 12.6 ms | 1730 | 137 TFLOP/s | 241 |
| 3840x2160 | 3840x2176 | 29.3 ms | 3907 | 133 TFLOP/s | 241 |

The FLOP column is the graph's multiply-accumulates counted from its shapes, doubled. Two rows of it say where
the limit is, and the answer is not the same at both ends:

* **512x512 to 768x768 is 2.1x the arithmetic for 1.04x the time.** At that size nothing is saturated: 51
  TFLOP/s is under a fifth of the tensor rate, and the whole weight set - 148 MB, every byte of it read once per
  frame - is only 54 GB/s against a 504 GB/s part. The limit is stages too small to fill the machine, plus the
  fixed cost of 241 launches.
* **1080p to 4K is 3.83x the arithmetic for 3.77x the time.** At that size it is compute-limited and scales
  linearly, saturating around 133-137 TFLOP/s - roughly half the part's dense FP8 tensor rate.

So the work that made this fast (staging operands in shared memory, keeping a block's intermediates in
registers, folding the pool and the upsample into neighbouring blocks, replacing barriers with counters) was
aimed at the left-hand regime, and it removed traffic and launches rather than arithmetic. What is left at 4K
is tensor utilization, which is a different problem.

The arithmetic is also almost flat across the pyramid - each level is 11-18% of the frame's FLOPs:

| stage | GFLOP share at 768x768 |
| --- | --- |
| 256-channel blocks (L3, 16 of them) | 17.6% |
| 128-channel blocks (L2, 12) | 14.6% |
| 32-channel blocks 0 and 70 (field) | 14.1% |
| 32-channel blocks (L0, 8) | 14.1% |
| 512-channel split blocks (L4, 16) | 13.8% |
| ViT (L5, 8) | 13.4% |
| 64-channel blocks (L1, 8) | 11.5% |
| stage transitions, the adapter and the head | 0.9% |

That flatness is the U-net's doing: each level quarters the token count and roughly quadruples the work per
token, so no level dominates. Time is *not* distributed that way, which is the useful part - the two
full-resolution blocks are 14% of the arithmetic and about 10% of the time because they are one long streaming
pass, while the ViT is 13% of the arithmetic and 17% of the time because 192 tokens cannot fill the machine,
which is why it is the only stage that needs split-K and 192-row tiles.

