// Run from WSL with Windows Edge. No npm packages or compiler are required.
// node harness/run.mjs --image /artifacts/input.png --prefix single --repeat 2
import { spawn, execFileSync } from 'node:child_process';
import { existsSync, openSync, closeSync } from 'node:fs';
import { writeFile, mkdir, copyFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const harness = resolve(fileURLToPath(new URL('.', import.meta.url)));
const project = resolve(harness, '..');
const artifacts = resolve(process.env.NR_ARTIFACTS ?? `${project}/artifacts`);
const args = process.argv.slice(2);
function option(name, fallback) { const at = args.indexOf(name); return at < 0 ? fallback : args[at + 1]; }
const prefix = option('--prefix', 'single');
if (!/^[a-zA-Z0-9_-]{1,64}$/.test(prefix)) throw new Error('invalid --prefix');
const port = Number(option('--port', process.env.PORT ?? 8100));
const debugPort = Number(option('--debug-port', process.env.EDGE_DEBUG_PORT ?? 9224));
const timeoutMilliseconds = Number(option('--timeout', '600000'));
const edge = process.env.EDGE ?? [
  '/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe',
].find(existsSync);
if (!edge) throw new Error('Windows Edge not found; set EDGE to its WSL executable path');
const windowsTemp = execFileSync('powershell.exe', ['-NoProfile', '-Command', '[System.IO.Path]::GetTempPath()'], { encoding: 'utf8' }).trim();
const profile = `${windowsTemp.replace(/[\\/]$/, '')}\\dlss-webgpu-${Date.now()}`;
const query = new URLSearchParams({ image: option('--image', '/artifacts/input.png'), prefix,
  repeat: option('--repeat', '2'), seed: option('--seed', '0'), saveHead: '1',
  tone: option('--tone', '1'), structure: option('--structure', '1'), skin: option('--skin', '-1'),
  autoMask: option('--auto-mask', '1') });
const pageUrl = `http://localhost:${port}/single.html?${query}`;
const sleep = milliseconds => new Promise(done => setTimeout(done, milliseconds));
async function cdp(websocket, method, params = {}) {
  const socket = new WebSocket(websocket);
  await new Promise((done, fail) => { socket.onopen = done; socket.onerror = fail; });
  const reply = new Promise((done, fail) => {
    const timer = setTimeout(() => fail(new Error(`CDP timeout: ${method}`)), 10000);
    socket.onmessage = event => {
      const value = JSON.parse(event.data);
      if (value.id !== 1) return;
      clearTimeout(timer);
      if (value.error) fail(new Error(JSON.stringify(value.error))); else done(value.result);
    };
    socket.onclose = () => { if (method === 'Browser.close') { clearTimeout(timer); done({}); } };
  });
  socket.send(JSON.stringify({ id: 1, method, params }));
  try { return await reply; } finally { socket.close(); }
}

await mkdir(artifacts, { recursive: true });
try {
  await fetch(`http://localhost:${port}/status`);
  throw new Error(`port ${port} is already in use; select --port with an unused port`);
} catch (error) {
  if (!String(error).includes('fetch failed')) throw error;
}
const serverLog = openSync(`${artifacts}/${prefix}-server.log`, 'w');
const edgeLog = openSync(`${artifacts}/${prefix}-edge.log`, 'w');
const server = spawn(process.execPath, [resolve(harness, 'server.mjs'), String(port)], {
  env: { ...process.env, PORT: String(port) }, stdio: ['ignore', serverLog, serverLog],
});
let browser;
let browserWebsocket;
const evidence = { startedAt: new Date().toISOString(), project, serverPid: server.pid,
  serverPort: port, debugPort, edgeExecutable: edge, isolatedWindowsProfile: profile,
  url: pageUrl, weightsDirectory: resolve(process.env.NR_WEIGHTS ?? `${project}/models/nr`) };
try {
  let ready = false;
  for (let attempt = 0; attempt < 100; attempt++) {
    try { const response = await fetch(`http://localhost:${port}/status`); if (response.ok) { ready = true; break; } } catch {}
    if (server.exitCode !== null) throw new Error(`server exited with ${server.exitCode}`);
    await sleep(100);
  }
  if (!ready) throw new Error('server failed to become ready');
  const flags = ['--headless=new', '--enable-unsafe-webgpu', `--user-data-dir=${profile}`,
    '--no-first-run', '--no-default-browser-check', '--disable-extensions',
    `--remote-debugging-port=${debugPort}`, pageUrl];
  evidence.browserArguments = flags;
  browser = spawn(edge, flags, { stdio: ['ignore', edgeLog, edgeLog] });
  evidence.browserLauncherPid = browser.pid;
  const start = Date.now();
  let previousMessage = '';
  let completed = false;
  while (Date.now() - start < timeoutMilliseconds) {
    if (!browserWebsocket) {
      try {
        const version = await (await fetch(`http://localhost:${debugPort}/json/version`)).json();
        browserWebsocket = version.webSocketDebuggerUrl;
        const processInfo = await cdp(browserWebsocket, 'SystemInfo.getProcessInfo');
        evidence.windowsBrowserPid = processInfo.processInfo.find(process => process.type === 'browser')?.id;
      } catch {}
    }
    const state = await (await fetch(`http://localhost:${port}/status`)).json();
    const message = state.events?.at(-1)?.message ?? state.status;
    if (message !== previousMessage) { console.log(message); previousMessage = message; }
    if (state.result?.prefix === prefix) {
      if (state.result.status !== 'completed') throw new Error(JSON.stringify(state.result));
      evidence.status = 'completed'; evidence.result = state.result;
      await writeFile(`${artifacts}/${prefix}-server-state.json`, JSON.stringify(state, null, 2) + '\n');
      await copyFile(`${artifacts}/${prefix}-enhanced.png`, `${artifacts}/output.png`);
      await copyFile(`${artifacts}/${prefix}-runtime.json`, `${artifacts}/inference.json`);
      completed = true;
      console.log(`saved ${artifacts}/${prefix}-enhanced.png`);
      break;
    }
    await sleep(500);
  }
  if (!completed) throw new Error(`inference timed out after ${timeoutMilliseconds} ms`);
} catch (error) {
  evidence.status = 'failed'; evidence.error = String(error.stack ?? error); process.exitCode = 1;
  console.error(evidence.error);
} finally {
  if (browserWebsocket) {
    try { await cdp(browserWebsocket, 'Browser.close'); evidence.browserClosed = true; }
    catch (error) { evidence.browserCloseError = String(error); }
  }
  server.kill('SIGTERM');
  evidence.finishedAt = new Date().toISOString();
  await writeFile(`${artifacts}/${prefix}-launch.json`, JSON.stringify(evidence, null, 2) + '\n');
  closeSync(serverLog); closeSync(edgeLog);
}
