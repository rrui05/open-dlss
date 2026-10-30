// Launch the standalone benchmark with an isolated, hidden headless Edge profile.
import { spawn, execFileSync } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { createWriteStream, existsSync } from 'node:fs';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { createServer as createPortProbe } from 'node:net';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createBenchServer } from './server.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const project = resolve(here, '..');
const args = process.argv.slice(2);
function option(name, fallback) {
  const index = args.indexOf(name);
  if (index === -1) return fallback;
  if (index === args.length - 1 || args[index + 1].startsWith('--')) throw new Error(`${name} requires a value`);
  return args[index + 1];
}
function integer(name, fallback, minimum, maximum = 1000000) {
  const value = Number(option(name, fallback));
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) throw new Error(`${name} must be an integer in [${minimum},${maximum}]`);
  return value;
}
function portablePath(value) {
  if (process.platform !== 'win32' && /^[a-zA-Z]:[\\/]/.test(value)) {
    return `/mnt/${value[0].toLowerCase()}/${value.slice(3).replaceAll('\\', '/')}`;
  }
  return resolve(value);
}
function capture(command, parameters, options = {}) {
  try { return execFileSync(command, parameters, { encoding: 'utf8', windowsHide: true, timeout: 15000, stdio: ['ignore', 'pipe', 'pipe'], ...options }).trim(); }
  catch (error) { return `unavailable: ${error.message}`; }
}
async function fileRequired(path, label) {
  const info = await stat(path).catch(() => null);
  if (!info?.isFile()) throw new Error(`${label} does not exist: ${path}`);
}
async function portFree(port) {
  const probe = createPortProbe();
  await new Promise((done, fail) => {
    probe.once('error', error => fail(new Error(`port ${port} is not available: ${error.message}`)));
    probe.listen(port, '127.0.0.1', () => probe.close(done));
  });
}
async function cdp(websocket, method) {
  const socket = new WebSocket(websocket);
  await new Promise((done, fail) => {
    const timer = setTimeout(() => { socket.close(); fail(new Error(`CDP connection timeout: ${method}`)); }, 5000);
    socket.onopen = () => { clearTimeout(timer); done(); };
    socket.onerror = error => { clearTimeout(timer); fail(error); };
  });
  try {
    return await new Promise((done, fail) => {
      const timer = setTimeout(() => fail(new Error(`CDP method timeout: ${method}`)), 5000);
      socket.onmessage = event => {
        const result = JSON.parse(event.data);
        if (result.id !== 1) return;
        clearTimeout(timer);
        if (result.error) fail(new Error(JSON.stringify(result.error))); else done(result.result);
      };
      socket.onclose = () => {
        clearTimeout(timer);
        if (method === 'Browser.close') done({}); else fail(new Error(`CDP socket closed: ${method}`));
      };
      socket.onerror = error => { clearTimeout(timer); fail(error); };
      socket.send(JSON.stringify({ id: 1, method }));
    });
  } finally { socket.close(); }
}
const sleep = ms => new Promise(done => setTimeout(done, ms));

async function main() {
  if (args.includes('--help')) {
    console.log(`Usage: node run.mjs [options]
  --width 896 --height 512       Exact prepared PNG dimensions; no resize
  --warmups 5 --iterations 20    Excluded warmups and recorded graph calls
  --input PNG                   Default: ../inputs/street_896x512.png
  --out DIR                     Default: ../runs/opendlss_<width>x<height>
  --upstream DIR                OpenDLSS repository (default: this repository)
  --weights DIR                 Packed model (default: <upstream>/models/nr)
  --page bench.html|cross.html   Benchmark or identical-feature cross-check
  --port 8146 --debug-port 9246  Loopback server and isolated browser debug port
  --edge EXE                    Browser executable (also read from EDGE)
  --timeout 600000              Browser result timeout, in milliseconds
  --help                        Print this text without starting a browser/GPU

cross.html reads native_features.f32 and native_head.f32 beside --input.
Node.js 22+ with global fetch and WebSocket is required.`);
    return;
  }
  const width = integer('--width', 896, 33), height = integer('--height', 512, 33);
  const warmups = integer('--warmups', 5, 0), iterations = integer('--iterations', 20, 1);
  const port = integer('--port', 8146, 1, 65535), debugPort = integer('--debug-port', 9246, 1, 65535);
  if (port === debugPort) throw new Error('server and browser debug ports must differ');
  const timeout = integer('--timeout', 600000, 1000, 86400000);
  const page = option('--page', 'bench.html');
  if (!['bench.html', 'cross.html'].includes(page)) throw new Error('--page must be bench.html or cross.html');
  const upstream = portablePath(option('--upstream', resolve(here, '../../..')));
  const weights = portablePath(option('--weights', resolve(upstream, 'models/nr')));
  const input = portablePath(option('--input', resolve(project, 'inputs/street_896x512.png')));
  const output = portablePath(option('--out', resolve(project, `runs/opendlss_${width}x${height}`)));
  const candidates = process.platform === 'win32'
    ? [join(process.env['ProgramFiles(x86)'] ?? 'C:/Program Files (x86)', 'Microsoft/Edge/Application/msedge.exe'), join(process.env.ProgramFiles ?? 'C:/Program Files', 'Microsoft/Edge/Application/msedge.exe')]
    : ['/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', '/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe', '/usr/bin/microsoft-edge', '/usr/bin/chromium'];
  const edgeOption = option('--edge', process.env.EDGE ?? candidates.find(existsSync));
  if (!edgeOption) throw new Error('Edge executable was not found; specify --edge');
  const edge = portablePath(edgeOption);
  await fileRequired(input, 'input PNG');
  await fileRequired(resolve(weights, 'manifest.json'), 'packed model manifest');
  await fileRequired(resolve(here, page), 'benchmark page');
  if (page === 'cross.html') {
    await fileRequired(resolve(dirname(input), 'native_features.f32'), 'native feature dump');
    await fileRequired(resolve(dirname(input), 'native_head.f32'), 'native head dump');
  }
  await fileRequired(edge, 'browser executable');
  await portFree(port);
  await portFree(debugPort);
  await mkdir(output, { recursive: true });
  const manifest = await readFile(resolve(weights, 'manifest.json'));
  const hostEnvironment = {
    node: process.version, platform: process.platform,
    gpu_query: capture('nvidia-smi', ['--query-gpu=name,driver_version,memory.total,memory.used,memory.free', '--format=csv']),
    opendlss_commit: capture('git', ['-C', upstream, 'rev-parse', 'HEAD']),
    input_sha256: createHash('sha256').update(await readFile(input)).digest('hex'),
    manifest_sha256: createHash('sha256').update(manifest).digest('hex'),
  };
  const unique = `opendlss-joint-${Date.now()}-${randomUUID()}`;
  let profile = join(tmpdir(), unique);
  if (process.platform !== 'win32' && edge.toLowerCase().endsWith('.exe')) {
    const windowsTemp = capture('powershell.exe', ['-NoProfile', '-Command', '[System.IO.Path]::GetTempPath()']);
    if (windowsTemp.startsWith('unavailable:')) throw new Error(`could not obtain a Windows profile path: ${windowsTemp}`);
    profile = `${windowsTemp.replace(/[\\/]$/, '')}\\${unique}`;
  }
  const evidence = {
    status: 'running', stage: 'launching', started_at_utc: new Date().toISOString(),
    command: process.argv, host_environment: hostEnvironment,
    configuration: { width, height, warmups, iterations, input, output, upstream, weights, port, debugPort },
    browser_executable: edge, isolated_browser_profile: profile,
    process_cleanup_scope: 'Only the browser launched with this unique profile and debug port is closed; no global browser termination.',
  };
  const saveLaunch = () => writeFile(resolve(output, 'launch.json'), JSON.stringify(evidence, null, 2) + '\n');
  let bench, browser, browserSocket, browserPid, spawnError, interrupted = false;
  let log;
  const onInterrupt = () => { interrupted = true; };
  process.on('SIGINT', onInterrupt); process.on('SIGTERM', onInterrupt);
  try {
    bench = await createBenchServer({ harness: here, upstream, weights, input, output, width, height,
      warmups, iterations, port, hostEnvironment,
      onEvent: event => console.log(JSON.stringify({ stage: event.stage, message: event.message })),
    });
    const flags = [
      '--headless=new', '--enable-unsafe-webgpu', `--user-data-dir=${profile}`,
      '--no-first-run', '--no-default-browser-check', '--disable-extensions',
      '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
      '--disable-backgrounding-occluded-windows', `--remote-debugging-port=${debugPort}`,
      '--remote-debugging-address=127.0.0.1', new URL(page, bench.url).href,
    ];
    evidence.browser_arguments = flags;
    log = createWriteStream(resolve(output, 'edge.log'));
    browser = spawn(edge, flags, { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
    browser.stdout.pipe(log, { end: false }); browser.stderr.pipe(log, { end: false });
    browser.on('error', error => { spawnError = error; });
    evidence.browser_launcher_pid = browser.pid;
    await saveLaunch();
    const deadline = Date.now() + timeout;
    while (!bench.state.result && Date.now() < deadline) {
      if (interrupted) throw new Error('benchmark launcher was interrupted');
      if (spawnError) throw spawnError;
      if (!browserSocket) {
        try {
          const response = await fetch(`http://127.0.0.1:${debugPort}/json/version`, { signal: AbortSignal.timeout(1500) });
          const version = await response.json();
          browserSocket = version.webSocketDebuggerUrl;
          evidence.browser_version = version;
          const processes = await cdp(browserSocket, 'SystemInfo.getProcessInfo');
          browserPid = processes.processInfo?.find(item => item.type === 'browser')?.id;
          evidence.browser_pid = browserPid;
          await saveLaunch();
        } catch { /* Edge may not have opened its debug endpoint yet. */ }
      }
      await sleep(400);
    }
    if (!bench.state.result) throw new Error(`benchmark timed out after ${timeout} ms; last stage: ${bench.state.stage}`);
    evidence.result_status = bench.state.result.status;
    evidence.stage = bench.state.result.stage ?? 'complete';
    if (bench.state.result.status !== 'completed') throw new Error(`browser benchmark failed: ${JSON.stringify(bench.state.result.error ?? bench.state.result.errors ?? bench.state.result)}`);
    evidence.status = 'completed';
    const timing = bench.state.result.benchmark?.synchronized_wall;
    console.log(JSON.stringify({ status: 'completed', output, synchronized_wall: timing }));
  } catch (error) {
    evidence.status = 'failed'; evidence.error = String(error.stack ?? error); process.exitCode = 1;
    if (!bench?.state.result) {
      await writeFile(resolve(output, 'results.json'), JSON.stringify({
        status: 'failed', stage: bench?.state.stage ?? evidence.stage, error: { type: error.name, message: error.message, traceback: String(error.stack ?? error) },
        configuration: evidence.configuration, environment: { host: hostEnvironment },
      }, null, 2) + '\n');
    }
    console.error(evidence.error);
  } finally {
    process.off('SIGINT', onInterrupt); process.off('SIGTERM', onInterrupt);
    if (browserSocket) {
      try { await cdp(browserSocket, 'Browser.close'); evidence.browser_closed_via_cdp = true; }
      catch (error) { evidence.browser_close_error = String(error); }
    }
    if (!evidence.browser_closed_via_cdp && browser?.pid) {
      if (process.platform === 'win32') {
        evidence.browser_cleanup_fallback = capture('taskkill', ['/PID', String(browserPid ?? browser.pid), '/T', '/F']);
      } else {
        evidence.browser_cleanup_fallback = browser.kill('SIGTERM');
      }
    }
    if (bench) {
      await writeFile(resolve(output, 'server_state.json'), JSON.stringify(bench.state, null, 2) + '\n');
      await bench.close();
    }
    // CDP close and child-process stream closure need not occur in the same tick.
    // Stop piping before ending the log to avoid a late "write after end" error.
    browser?.stdout?.unpipe(log);
    browser?.stderr?.unpipe(log);
    log?.end();
    evidence.finished_at_utc = new Date().toISOString();
    await saveLaunch();
  }
}

main().catch(error => { console.error(String(error.stack ?? error)); process.exitCode = 1; });
