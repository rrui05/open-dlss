// The parity harness: run the port on a recorded frame and compare every block boundary, byte for byte,
// against what native produced for the same input. It is the browser twin of `dlss5vk parity`.
//
// A fixture is a directory holding a manifest and whatever the capture recorded; `loadFixture` below says
// what the two kinds carry and which half of the port each one reaches.
//
// A boundary fixture reads its input features rather than generating them, for a reason worth stating: the
// three noise lanes are Box-Muller over the GPU's *approximate* f32 transcendentals, so their last bits
// depend on which transcendental unit evaluated them. Everything downstream of the features is exactly
// reproducible; the features themselves are the one place a port cannot be, so that comparison starts after
// them. An end-to-end fixture records the proxy instead and generates the features from it, which puts those
// three lanes back inside the comparison - on the card this was measured on they reproduce, but that is a
// measurement rather than a guarantee.

const CODE_VALUES = (() => {
  const values = new Float32Array(256);
  for (let byte = 0; byte < 256; ++byte) {
    const sign = (byte & 0x80) ? -1 : 1;
    const exponent = (byte >>> 3) & 0xf;
    const mantissa = byte & 0x7;
    values[byte] = exponent === 0 ? sign * mantissa * 2 ** -9
                 : (exponent === 0xf && mantissa === 0x7) ? sign * 0
                 : sign * (1 + mantissa / 8) * 2 ** (exponent - 7);
  }
  return values;
})();

/** Byte-for-byte, plus the shape of whatever disagreement there is. */
export function compareCodes(actual, expected) {
  let mismatches = 0;
  let maxAbs = 0;
  let sumSquares = 0;
  let first = null;
  const deltas = new Map();
  for (let i = 0; i < expected.length; ++i) {
    const a = actual[i];
    const e = expected[i];
    if (a === e) continue;
    mismatches += 1;
    if (!first) first = { index: i, actual: a, expected: e };
    const difference = CODE_VALUES[a] - CODE_VALUES[e];
    maxAbs = Math.max(maxAbs, Math.abs(difference));
    sumSquares += difference * difference;
    // A signed code delta says whether the port rounded one step up or down, which is a different problem
    // from a structural error; a histogram distinguishes the two at a glance.
    const delta = (a & 0x80 ? -(a & 0x7f) : a & 0x7f) - (e & 0x80 ? -(e & 0x7f) : e & 0x7f);
    deltas.set(delta, (deltas.get(delta) ?? 0) + 1);
  }
  return { count: expected.length, mismatches, maxAbs, rmse: Math.sqrt(sumSquares / expected.length), first,
           deltas: [...deltas.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6) };
}

/**
 * A fixture is a directory holding a manifest and whatever the capture recorded. Two kinds exist and they
 * check different halves of the port:
 *
 *   * a boundary fixture carries the input features and one file per block output, which gates the network
 *     block by block but stops at block 69 - it has nothing to say about the post block, the head, or the
 *     composition;
 *   * an end-to-end fixture carries the display proxy the network was given, the head native produced from
 *     it, and the composed image, which gates exactly the part the other one cannot reach.
 *
 * Neither is sufficient alone. `runParity` does whichever comparisons the fixture can support.
 */
export async function loadFixture(directory) {
  const manifest = await (await fetch(`${directory}/manifest.json`)).json();
  const floats = async (entry) => (entry
    ? new Float32Array(await (await fetch(`${directory}/${entry.file}`)).arrayBuffer()) : null);
  const features = manifest.inputFeatures ? await floats(manifest.inputFeatures) : null;
  const proxy = await floats(manifest.proxy);
  const referenceHead = await floats(manifest.referenceHead);
  // The older captures published the composed image as bytes; the ones with a proxy publish it as floats.
  const nativeOutput = manifest.nativeOutput
    ? (manifest.nativeOutput.file.endsWith('.u8')
        ? new Uint8Array(await (await fetch(`${directory}/${manifest.nativeOutput.file}`)).arrayBuffer())
        : await floats(manifest.nativeOutput))
    : null;
  return { directory, manifest, features, proxy, referenceHead, nativeOutput };
}

/** The boundaries the fixture carries, in graph order: blocks, with the transitions interleaved. */
export function fixtureBoundaries(fixture) {
  const entries = [];
  for (const entry of fixture.manifest.blocks ?? []) {
    entries.push({ order: entry.block, name: `block-${entry.block}`, entry });
  }
  for (const entry of fixture.manifest.transitions ?? []) {
    entries.push({ order: parseFloat(entry.id) + 0.5, name: `transition-${entry.id}`, entry });
  }
  entries.sort((a, b) => a.order - b.order);
  return entries;
}

export async function runParity(network, fixture, report, { repeat = 1, profile = false } = {}) {
  const geometry = network.geometry;
  const [width, height] = fixture.manifest.sourceDimensions;
  if (width !== geometry.validWidth || height !== geometry.validHeight) {
    throw new Error(`fixture is ${width}x${height}, the network was built for ` +
                    `${geometry.validWidth}x${geometry.validHeight}`);
  }
  const expectedFeatures = geometry.fullRows * 16;
  if (fixture.features && fixture.features.length !== expectedFeatures) {
    throw new Error(`input features are ${fixture.features.length} floats, expected ${expectedFeatures}`);
  }

  report.status('running the graph');
  if (profile) network.recorder.enableProfiling();
  if (fixture.features) network.writeFeatures(fixture.features);
  else await network.featuresFromProxy(fixture.proxy, fixture.manifest);
  // The first run pays for shader compilation and buffer residency, so a timing is only meaningful once the
  // graph has been through at least once. `repeat` reports the best of several, which is the number to quote.
  let elapsed = Infinity;
  for (let pass = 0; pass < Math.max(1, repeat); ++pass) {
    const started = performance.now();
    await network.run();
    const took = performance.now() - started;
    if (pass > 0 || repeat === 1) elapsed = Math.min(elapsed, took);
    report.status(`run ${pass + 1} of ${repeat}: ${took.toFixed(0)} ms`);
  }
  report.timing(elapsed, network.recorder.dispatchCount);
  const breakdown = profile ? await network.recorder.readProfile() : null;
  for (const kernel of breakdown ?? []) {
    report.note(`  ${kernel.name.padEnd(18)} ${kernel.milliseconds.toFixed(1)} ms`);
  }

  let exact = 0;
  let total = 0;
  const available = new Set(network.boundaryNames);
  for (const { name, entry } of fixtureBoundaries(fixture)) {
    if (!available.has(name)) { report.row(name, null, 'not captured'); continue; }
    report.status(`comparing ${name}`);
    const expected = new Uint8Array(await (await fetch(`${fixture.directory}/${entry.file}`)).arrayBuffer());
    const actual = await network.readBoundary(name);
    total += 1;
    if (actual.length !== expected.length) {
      report.row(name, null, `size ${actual.length} vs ${expected.length}`);
      continue;
    }
    const result = compareCodes(actual, expected);
    if (result.mismatches === 0) exact += 1;
    report.row(name, result, `${entry.width}x${entry.height}x${entry.channels}`);
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }
  // The boundaries stop at block 69. Everything after it - the full-resolution post block, the head matrix,
  // and the composition - is only reached by these two.
  const head = fixture.referenceHead || fixture.nativeOutput ? await network.readHead() : null;

  if (fixture.referenceHead) {
    report.status('comparing the head');
    const result = compareFloats(head, fixture.referenceHead);
    total += 1;
    if (result.mismatches === 0) exact += 1;
    report.row('head', result, `${geometry.fullWidth}x${geometry.fullHeight}x4`);
  }

  if (fixture.nativeOutput) {
    report.status('comparing the composed image');
    const result = compareComposed(head, fixture, geometry, width, height);
    total += 1;
    if (result.mismatches === 0) exact += 1;
    report.row('composed rgb', result, `${width}x${height}x3`);
    if (result.deltas.length) report.note(`  composed rgb: ${result.deltas[0][1]} ${result.deltas[0][0]}`);
  }

  report.done(exact, total);
  return { exact, total };
}

/** Bitwise, because these are the same numbers or they are not. */
export function compareFloats(actual, expected) {
  let mismatches = 0;
  let maxAbs = 0;
  let sumSquares = 0;
  let first = null;
  for (let i = 0; i < expected.length; ++i) {
    if (actual[i] === expected[i]) continue;
    mismatches += 1;
    if (!first) first = { index: i, actual: actual[i], expected: expected[i] };
    const difference = actual[i] - expected[i];
    maxAbs = Math.max(maxAbs, Math.abs(difference));
    sumSquares += difference * difference;
  }
  return { count: expected.length, mismatches, maxAbs,
           rmse: Math.sqrt(sumSquares / expected.length), first, deltas: [] };
}

/** One f32 fused multiply-add: the product is exact in f64, so a single narrowing is the fma's own. */
const fma32 = (a, b, c) => Math.fround(a * b + c);

/** The centred proxy of one pixel, from whichever of the two the fixture carries. */
function centredAt(fixture, x, y, c, width, fullWidth) {
  // A proxy fixture stores the code value; a feature fixture stores the centring of it, which is the same
  // number the composition wants and is what native's own kernel computed.
  return fixture.proxy ? fma32(fixture.proxy[(y * width + x) * 4 + c], 0.125, -0.0625)
                       : fixture.features[(y * fullWidth + x) * 16 + 4 + c];
}

/**
 * The composition, exactly as the demo publishes it and as the native capture recorded it: the head's
 * residual on the centred proxy, back to a code value, truncated to the half grid. No history (the capture
 * is one frame), no style, no display transform - the fixture stops where the renderer would take over.
 *
 * Two kinds of reference exist. A float capture is compared as published; an older byte capture is the same
 * image quantized to eight bits, so the comparison is too.
 */
export function compareComposed(head, fixture, geometry, width, height) {
  const expected = fixture.nativeOutput;
  const bytes = expected instanceof Uint8Array;
  let mismatches = 0;
  let maxAbs = 0;
  let sumSquares = 0;
  let first = null;
  let contracted = 0;
  const count = width * height * 3;
  for (let y = 0; y < height; ++y) {
    for (let x = 0; x < width; ++x) {
      for (let c = 0; c < 3; ++c) {
        const h = head[(y * geometry.fullWidth + x) * 4 + c];
        const inner = fma32(h, 0.03125, centredAt(fixture, x, y, c, width, geometry.fullWidth));
        const value = truncateToHalf(Math.min(Math.max(Math.fround(inner * 8 + 0.5), 0), 1));
        const published = bytes ? quantizeByte(value) : value;
        const want = expected[(y * width + x) * 4 + c];
        if (published === want) continue;
        // The same expression with the last multiply-add contracted; both spellings are legal and a compiler
        // may pick either, so a value that matches the other one is counted apart rather than called wrong.
        const alt = truncateToHalf(Math.min(Math.max(fma32(inner, 8, 0.5), 0), 1));
        if ((bytes ? quantizeByte(alt) : alt) === want) { contracted += 1; continue; }
        // A byte capture cannot distinguish a one-code difference from where its own rounding fell, so a
        // single code is counted apart rather than called wrong. The float capture has no such slack, and
        // it is the one that says whether the head is exact.
        if (bytes && Math.abs(published - want) <= 1) { contracted += 1; continue; }
        mismatches += 1;
        if (!first) first = { index: (y * width + x) * 3 + c, actual: published, expected: want };
        const difference = published - want;
        maxAbs = Math.max(maxAbs, Math.abs(difference));
        sumSquares += difference * difference;
      }
    }
  }
  return { count, mismatches, maxAbs, rmse: Math.sqrt(sumSquares / count), first,
           deltas: contracted ? [[bytes ? 'within one code' : 'contracted', contracted]] : [] };
}

/** The eight-bit publication, evaluated in f32 the way the capture that produced the byte reference did. */
const quantizeByte = (value) => Math.min(255, Math.max(0, Math.floor(Math.fround(value * 255 + 0.5))));

/** Toward zero to the half grid: what the publication does, and not the same as rounding. */
export function truncateToHalf(value) {
  const scratch = truncateToHalf.scratch ?? (truncateToHalf.scratch = {
    floats: new Float32Array(1), bits: null, half: new Uint16Array(1), out: new Float32Array(1),
  });
  scratch.bits = scratch.bits ?? new Uint32Array(scratch.floats.buffer);
  scratch.floats[0] = value;
  const bits = scratch.bits[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  let half;
  if (exponent === 0xff) half = sign | (mantissa ? 0x7e00 : 0x7c00);
  else {
    const halfExponent = exponent - 112;
    if (halfExponent >= 31) half = sign | 0x7c00;
    else if (halfExponent <= 0) half = halfExponent < -10 ? sign : sign | ((mantissa | 0x800000) >>> (14 - halfExponent));
    else half = sign | (halfExponent << 10) | (mantissa >>> 13);
  }
  const s = (half & 0x8000) ? -1 : 1;
  const e = (half >>> 10) & 0x1f;
  const m = half & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 0x1f) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

/** RGBA8 against RGBA8, ignoring alpha: the composed image is opaque by construction. */
export function compareImages(actual, expected) {
  let mismatches = 0;
  let maxAbs = 0;
  let sumSquares = 0;
  let first = null;
  const deltas = new Map();
  for (let i = 0; i < expected.length; ++i) {
    if ((i & 3) === 3) continue;
    const a = actual[i];
    const e = expected[i];
    if (a === e) continue;
    mismatches += 1;
    if (!first) first = { index: i, actual: a, expected: e };
    maxAbs = Math.max(maxAbs, Math.abs(a - e));
    sumSquares += (a - e) * (a - e);
    deltas.set(a - e, (deltas.get(a - e) ?? 0) + 1);
  }
  const count = expected.length - (expected.length >> 2);
  return { count, mismatches, maxAbs, rmse: Math.sqrt(sumSquares / count), first,
           deltas: [...deltas.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6) };
}

/**
 * The same composition as `compareComposed`, as a picture. It is the network's own output for the frame with
 * no history to blend and no display transform: a recorded frame has neither a previous frame nor the scene
 * the renderer would tone map back into.
 */
export function composeImage(head, fixture, geometry) {
  const { validWidth, validHeight, fullWidth } = geometry;
  const pixels = new Uint8ClampedArray(validWidth * validHeight * 4);
  for (let y = 0; y < validHeight; ++y) {
    for (let x = 0; x < validWidth; ++x) {
      const out = (y * validWidth + x) * 4;
      for (let c = 0; c < 3; ++c) {
        const inner = fma32(head[(y * fullWidth + x) * 4 + c] , 0.03125,
                            centredAt(fixture, x, y, c, validWidth, fullWidth));
        const value = truncateToHalf(Math.min(Math.max(Math.fround(inner * 8 + 0.5), 0), 1));
        pixels[out + c] = quantizeByte(value);
      }
      pixels[out + 3] = 255;
    }
  }
  return new ImageData(pixels, validWidth, validHeight);
}
