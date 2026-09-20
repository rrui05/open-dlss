// Every gate the port has, in one command:
//
//   node tools/check.mjs
//
// Three runs, because they check different things and none of them subsumes another:
//
//   self-test      every numeric primitive against the C++ reference, over exhaustive inputs
//   parity nr512   75 block boundaries against what native produced from recorded input features
//   parity nr768   the head and the composed image against what native produced from a recorded proxy
//
// The boundary fixture stops at block 69: it says nothing about the full-resolution post block, the head
// matrix, or the composition. The end-to-end fixture carries no boundaries: it says nothing about where
// inside the network a difference began. Run both or neither.
//
// NR_FIXTURES must point at the directory holding nr512 and nr768. NR_HEADED=1 opens real windows.

import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const runs = [
  ['self-test', 'selftest', ''],
  ['parity nr512', 'parity', 'repeat=1'],
  ['parity nr768', 'parity', 'repeat=1&fixture=/fixtures/nr768'],
];

let failed = 0;
for (const [label, page, query] of runs) {
  process.stdout.write(`\n=== ${label} ===\n`);
  const code = await new Promise((resolve) => {
    const child = spawn(process.execPath, [join(here, 'headless.mjs'), page, query],
                        { stdio: 'inherit', env: process.env });
    child.on('exit', (status) => resolve(status ?? 1));
  });
  if (code !== 0) { failed += 1; process.stdout.write(`${label}: FAILED\n`); }
}
process.stdout.write(failed ? `\n${failed} of ${runs.length} checks failed\n` : '\nall checks passed\n');
process.exit(failed ? 1 : 0);
