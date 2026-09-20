# The frame: from a rendered image to pixels

The network takes 16 f32 lanes per padded pixel and returns 4. Everything around that (building the lanes,
turning the head into an image, and the temporal loop) is the *pipeline*, and it is as much a part of matching
NVIDIA's output as the network is. `demo/` implements it; `demo/README.md` covers building and driving the demo,
this file covers what it computes and why.

```
 Filament scene view          -> HDR rgba16f color
 Filament structure pass      -> rgba32ui velocity (object id, depth bits, motion x/y bits)
      |
      v  velocity_unpack.comp
 motion rg16f  (current -> previous, uv units, y down; zero where nothing was drawn or the
      |         surface was off screen last frame)
      v  nr_preprocess.comp
 features f32 [field][16]  ---->  the network (src/)  ---->  head f32 [field][4]
      |                                                          |
      +---------------------------- nr_composite.comp <----------+
                                          |
                          rgba8 output + the next history (rgba16f)
                                          |
                                Filament present view + ImGui
```

The NR work is recorded **into Filament's own command buffer**, between the scene view and the present view,
through a patched `Engine::queueVulkanCommand` hook. No extra submissions, no host synchronization, and the
renderer's own resource tracking stays valid because the pass leaves every image in
`SHADER_READ_ONLY_OPTIMAL`.

## The proxy

The network does not see HDR. It sees a *display proxy*: paper-white-relative scene radiance with a soft
shoulder above 0.75, sRGB-encoded, on the half grid.

```
v = scene / max(paperWhite, 0.05)
v = v > 0.75 ? 0.75 + 0.25 * (1 - exp(-5.770780 * (v - 0.75))) : v
proxy = f16(srgbEncode(v))                           # 0..1 code value
centred = f16((f16(proxy) - 0.5) * 0.125)            # feature lanes 4-6
```

Lanes 7-9 are the same transform applied to the reprojected previous **output**, not the previous scene. Non-
finite and negative scene values are clamped to zero first; the network's dynamic range is not the renderer's.

## Motion vectors

Filament has no per-object motion vectors, so the patch adds them: the picking/structure pass writes, per pixel,
`(object id, depth bits, motion x, motion y)` where motion is `previous NDC - current NDC`. The engine keeps the
previous world transform per renderable, the previous bones and morph weights for skinned and morphed meshes,
the previous per-instance transforms, and the previous un-jittered clip-from-world per view. Cost: one
depth-only pass.

`velocity_unpack.comp` turns that into `rg16f` motion in uv units with y down, and **zeroes it where nothing was
drawn or where the previous position falls off screen**. A reprojection that would sample outside the history
is better treated as "no history" than as a clamped sample.

Blended (translucent) renderables are not in the structure pass, so they carry only the camera's motion;
masked ones carry their own.

Verification (`--frames 220 --capture verify --orbit 0.4`): reprojecting frame 99's color with frame 100's
motion must reproduce frame 100. On Bistro at 1280x720 the mean error is 0.006, against 0.030 with no
reprojection and 0.037 with the sign flipped; with an animated asset and a static camera the skinned/morphed
path gives 0.007 against 0.014.

## History reconstruction

Both the preprocess (for lanes 7-9) and the composite sample the history at the reprojected position with a
**five-tap Catmull-Rom** filter (the four axis taps plus the centre, with the bilinear-weight trick), clamped to
the valid rectangle. A box or bilinear filter here visibly softens the result frame over frame, because the
history is fed back into the network's input.

## Composition

```
neural  = clamp(proxy + rgb / 4, 0, 1)                         # head channels 0-2, in proxy code space
weight  = clamp(sigmoid(head.a) * blendScale, 0, 1)            # blendScale: a learned f16 in the model
neural  = lerp(neural, history, weight)                        # only when the history is valid
history' = truncate_to_half(neural)                            # stored for the next frame
```

The fourth head channel is the network's own opinion about how much of the history to keep, per pixel;
`blendScale` (`block70.layer0.blend_scale`, 0.7397 in the shipped model) is a global learned cap on it. Note the
publication: the stored history is **truncated** toward zero to the half grid, not rounded. The next frame's
input lanes depend on it, so it is a publication point like any other.

After the blend the demo applies the optional style operator, the intensity blend back towards the proxy, and
then its own display path, which is not part of NR:

* **style**: the `natural` / `cinematic` presets are exposure, contrast and saturation offsets scaled by the
  local tone, applied in the same HSL operator the native runtime uses (the `exp2(log2(x))` round trips in
  `nr_composite.comp` are that operator's gamma = 1 path, kept literally). `custom` exposes its knobs.
* **tone upgrade**: the neural result is LDR; `upgradeToneMap` puts it back on the HDR scene by matching
  luminance ratios and transferring hue in Oklab, so highlights that the proxy clipped are not lost.
* **display transform**: ACES fit + sRGB into rgba8, matching the WebGI viewer this demo mirrors.

With NR off the composite takes the same display path on the scene itself and keeps the history primed with the
proxy, so toggling NR does not produce a one-frame flash.

## Frames in flight

Two history images and two parameter buffers alternate by `frame & 1`. The NR work is **pre-recorded once per
(history parity, NR on/off)** into four secondary command buffers, because nothing in it changes between frames
except the parameter block, which is written into the command stream with `vkCmdUpdateBuffer`. A resize (or the
renderer handing over different images) rebuilds the descriptor sets and those four command buffers while the
renderer is idle; the model, the kernels and the pipelines survive it.

Timestamps follow the same parity: a frame's six stamps are read two frames later, when the GPU is certainly
done with them, which is why the UI's timings lag by two frames and never stall the queue.

## Resolution

The network runs at the window's resolution, padded to the field. `NrPass::resize` re-fits everything in about
50 ms. There is no upscaling anywhere in the pipeline: DLSS-SR is a different network and is not part of this
repository.
