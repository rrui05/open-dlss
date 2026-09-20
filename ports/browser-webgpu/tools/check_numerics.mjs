// Checks src/numerics.js against web/fixtures/numerics.bin, the answers the Vulkan implementation's own CPU
// reference gives. Run from this directory with `node tools/check_numerics.mjs`. web/selftest.html runs the
// same case list against the WGSL; this one needs no GPU, so it is the first thing to run after touching
// either side of the port's arithmetic.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { readFixture, numericsCases, compare, fixedPointBound } from '../src/numerics_cases.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

const sections = readFixture(new Uint8Array(readFileSync(join(root, 'web/fixtures/numerics.bin'))));
const cases = numericsCases(sections);
let failed = 0;

for (const entry of cases) {
  if (entry.oracle === 'js') continue;  // nothing to check: the JS is the oracle there
  const { mismatches, nanPairs, skipped, first } = compare(entry, entry.actual);
  const notes = [];
  if (skipped) notes.push(`${skipped} non-finite skipped`);
  if (nanPairs) notes.push(`${nanPairs} NaN pairs`);
  const note = notes.length ? ` (${notes.join(', ')})` : '';
  if (mismatches === 0) {
    console.log(`  ok    ${String(entry.count).padStart(6)}  ${entry.name}${note}`);
  } else {
    failed += 1;
    console.log(`  FAIL  ${mismatches} of ${entry.count}  ${entry.name}${note}`);
    console.log(`        first at ${first.i}: reference 0x${first.expected.toString(16)}, ` +
                `port 0x${first.actual.toString(16)}`);
  }
}

const bound = fixedPointBound(cases);
console.log(`  aligned sums reached ${bound} (2^${Math.log2(bound).toFixed(1)}); ` +
            `the i32 the WGSL accumulates in holds ${2 ** 31 - 1}`);

if (failed) { console.log(`${failed} case(s) disagree with the reference`); process.exit(1); }
console.log('numerics agree with the Vulkan reference');
