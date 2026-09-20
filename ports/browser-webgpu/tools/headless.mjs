// Runs one of the port's pages in headless Chrome and exits with its verdict:
//
//   node tools/headless.mjs selftest
//   node tools/headless.mjs parity
//   NR_HEADED=1 node tools/headless.mjs demo      (a real window, to watch rather than to gate on)
//
// A server is started on an unused port, Chrome is pointed at the page with ?report=1, and the page posts its
// result back before closing itself. Set CHROME to a browser path if the usual locations are wrong.
//
// WebGPU in headless Chrome needs --enable-unsafe-webgpu; without it navigator.gpu is undefined and the page
// says so rather than silently passing.

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { startServer } from '../src/server.js';

const pages = { selftest: 'web/selftest.html', parity: 'web/parity.html', demo: 'demo/index.html' };

const which = process.argv[2] ?? 'selftest';
const extraQuery = process.argv[3] ?? '';
if (!pages[which]) {
  console.error(`unknown page "${which}"; expected one of ${Object.keys(pages).join(', ')}`);
  process.exit(2);
}

const candidates = [
  process.env.CHROME,
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);
const chrome = candidates.find((path) => existsSync(path));
if (!chrome) {
  console.error(`no Chrome found; tried:\n  ${candidates.join('\n  ')}\nSet CHROME to its path.`);
  process.exit(2);
}

const server = await startServer({ port: 0, onReport: report });
const port = server.address().port;
const profile = await mkdtemp(join(tmpdir(), 'nr-headless-'));

let settled = false;
function report(body) {
  if (settled) return;
  settled = true;
  console.log(body);
  const failed = /FAIL|ERROR/.test(body);
  finish(failed ? 1 : 0);
}

async function finish(code) {
  try { browser.kill(); } catch { /* already gone */ }
  server.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
  process.exit(code);
}

// NR_HEADED=1 opens a real window instead, for watching a run rather than gating on it.
const headed = process.env.NR_HEADED === '1';

const browser = spawn(chrome, [
  ...(headed ? [] : ['--headless=new']),
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
  `--user-data-dir=${profile}`,
  '--no-first-run',
  '--no-default-browser-check',
  '--disable-extensions',
  '--window-size=1920,1080',
  '--enable-logging=stderr',
  '--v=0',
  `http://localhost:${port}/${pages[which]}?report=1${extraQuery ? `&${extraQuery}` : ''}`,
], { stdio: ['ignore', 'ignore', 'pipe'] });

// Everything Chrome says goes through, including the page's own console. A run that fails before it can
// report has nothing else to explain itself with, and filtering here has twice hidden the actual cause.
browser.stderr.setEncoding('utf8');
browser.stderr.on('data', (text) => {
  for (const line of text.split('\n')) {
    const trimmed = line.trimEnd();
    if (trimmed && !/gcm|DEPRECATED_ENDPOINT|update_service_dialer|named pipe|Registration response/.test(trimmed)) {
      console.error(trimmed);
    }
  }
});

browser.on('exit', (code) => {
  if (!settled) { console.error(`the page closed without reporting (chrome exit ${code})`); finish(1); }
});

setTimeout(() => {
  if (!settled) { console.error('timed out after 300 s'); finish(1); }
}, 300_000).unref();
