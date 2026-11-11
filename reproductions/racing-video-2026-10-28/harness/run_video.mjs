// Persistent WebGPU frame runner. The network and its numerical kernels are unmodified.
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { readFile, writeFile, mkdir, stat, appendFile } from 'node:fs/promises';
import { createWriteStream } from 'node:fs';
import { resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { once } from 'node:events';

const here = fileURLToPath(new URL('.', import.meta.url));
const root = resolve(here, '../../..');
const args = process.argv.slice(2);
function option(name, fallback) { const at = args.indexOf(name); return at < 0 ? fallback : args[at + 1]; }
const source = resolve(option('--input', 'C:/Users/Win11/Desktop/racing-final704526.mp4'));
const out = resolve(option('--out', 'D:/newDLSS/outputs/racing-dlss-20261028'));
const ffmpeg = option('--ffmpeg', 'C:/Users/Win11/anaconda3/envs/bipadelRL/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe');
const edge = option('--edge', 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe');
const port = Number(option('--port', '8136'));
const debugPort = Number(option('--debug-port', '9236'));
const frameCount = Number(option('--frames', '960'));
const width = Number(option('--width', '832'));
const height = Number(option('--height', '480'));
const fps = option('--fps', '16/1');
const frameBytes = width * height * 4;
const config = { width, height, fps, frameCount, startFrame: 0, seed: 0, autoMask: true,
  conditioning: { localTone: 1, localStructure: 1, skinStructure: -1, style: 0 } };
const state = { status: 'initializing', startedAt: new Date().toISOString(), input: source,
  outputDirectory: out, config, decodedFrames: 0, encodedFrames: 0, events: [] };
await mkdir(resolve(out, 'frames'), { recursive: true });
await mkdir(resolve(out, 'samples'), { recursive: true });
await writeFile(resolve(out, 'frame-stats.jsonl'), '');
const outputVideo = resolve(out, 'racing-dlss-nr.mp4');
const sleep = ms => new Promise(done => setTimeout(done, ms));
const logs = [];
let decoder, encoder, browser, browserWebSocket;
let frameIterator, carry = Buffer.alloc(0), carryOffset = 0;
let nextInput = 0, nextOutput = 0, complete;
let failure;
let processFramesStarted = false;
let lastProgress = '';
async function saveState() { await writeFile(resolve(out, 'status.json'), JSON.stringify(state, null, 2)); }
function child(command, childArgs, logName, stdio) {
  const log = createWriteStream(resolve(out, logName)); logs.push(log);
  const process = spawn(command, childArgs, { stdio, windowsHide: true });
  process.stderr?.pipe(log);
  process.on('error', error => { failure = error; });
  return process;
}
function startPipes() {
  if (processFramesStarted) return;
  processFramesStarted = true;
  decoder = child(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-i', source, '-map', '0:v:0',
    '-an', '-frames:v', String(frameCount), '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1'],
    'decode.log', ['ignore', 'pipe', 'pipe']);
  encoder = child(ffmpeg, ['-hide_banner', '-loglevel', 'warning', '-y', '-f', 'rawvideo', '-pixel_format', 'rgba',
    '-video_size', `${width}x${height}`, '-framerate', fps, '-i', 'pipe:0', '-an', '-c:v', 'libx264',
    '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', outputVideo],
    'encode.log', ['pipe', 'ignore', 'pipe']);
  encoder.stdin.on('error', error => { failure = error; });
  frameIterator = decoder.stdout[Symbol.asyncIterator]();
}
async function frame() {
  const result = Buffer.allocUnsafe(frameBytes);
  let offset = 0;
  while (offset < result.length) {
    if (carryOffset === carry.length) {
      const next = await frameIterator.next();
      if (next.done) throw new Error(`decoder ended inside frame ${nextInput}: ${offset}/${frameBytes}`);
      carry = next.value; carryOffset = 0;
    }
    const count = Math.min(result.length - offset, carry.length - carryOffset);
    carry.copy(result, offset, carryOffset, carryOffset + count);
    offset += count; carryOffset += count;
  }
  return result;
}
async function body(request, limit = frameBytes + 65536) {
  const chunks = []; let length = 0;
  for await (const chunk of request) {
    length += chunk.length;
    if (length > limit) throw new Error('request body too large');
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}
function json(response, value, code = 200) {
  response.writeHead(code, { 'content-type': 'application/json', 'cache-control': 'no-store' });
  response.end(JSON.stringify(value));
}
const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.wgsl': 'text/plain', '.bin': 'application/octet-stream' };
const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://localhost');
    if (url.pathname === '/favicon.ico') { response.writeHead(204); return response.end(); }
    if (request.method === 'GET' && url.pathname === '/config') return json(response, config);
    if (request.method === 'GET' && url.pathname === '/status') return json(response, state);
    if (request.method === 'GET' && /^\/frame\/\d+$/.test(url.pathname)) {
      const index = Number(url.pathname.split('/').at(-1));
      if (index !== nextInput || index >= frameCount) throw new Error(`unexpected input frame ${index}; expected ${nextInput}`);
      startPipes();
      const data = await frame();
      if (index === 0 || index === frameCount - 1 || index % 160 === 0) {
        await writeFile(resolve(out, 'samples', `input-${String(index).padStart(6, '0')}.rgba`), data);
      }
      nextInput++; state.decodedFrames = nextInput;
      response.writeHead(200, { 'content-type': 'application/octet-stream', 'content-length': data.length });
      return response.end(data);
    }
    if (request.method === 'POST' && /^\/output\/\d+$/.test(url.pathname)) {
      const index = Number(url.pathname.split('/').at(-1));
      if (index !== nextOutput || index >= nextInput) throw new Error(`unexpected output frame ${index}; expected ${nextOutput}`);
      const data = await body(request);
      if (data.length !== frameBytes) throw new Error(`output frame ${index} has ${data.length} bytes, expected ${frameBytes}`);
      await writeFile(resolve(out, 'frames', `${String(index).padStart(6, '0')}.rgba`), data);
      await new Promise((done, fail) => encoder.stdin.write(data, error => error ? fail(error) : done()));
      nextOutput++; state.encodedFrames = nextOutput;
      return json(response, { ok: true, index, sha256: createHash('sha256').update(data).digest('hex') });
    }
    if (request.method === 'POST' && url.pathname === '/frame-stats') {
      const entry = JSON.parse((await body(request)).toString());
      await appendFile(resolve(out, 'frame-stats.jsonl'), JSON.stringify(entry) + '\n');
      return json(response, { ok: true });
    }
    if (request.method === 'POST' && url.pathname === '/progress') {
      const entry = JSON.parse((await body(request)).toString());
      state.status = 'running'; state.progress = entry;
      if (entry.message !== lastProgress) { console.log(entry.message); lastProgress = entry.message; }
      state.events.push({ at: new Date().toISOString(), ...entry });
      if (state.events.length > 30) state.events.shift();
      await saveState();
      return json(response, { ok: true });
    }
    if (request.method === 'POST' && url.pathname === '/complete') {
      complete = JSON.parse((await body(request)).toString());
      await writeFile(resolve(out, 'inference.json'), JSON.stringify(complete, null, 2));
      return json(response, { ok: true });
    }
    let directory = here, relative = url.pathname === '/' ? 'video.html' : url.pathname.slice(1);
    if (url.pathname.startsWith('/open-dlss/')) { directory = root; relative = url.pathname.slice(11); }
    if (url.pathname.startsWith('/weights/')) { directory = resolve(root, 'models/nr'); relative = url.pathname.slice(9); }
    const target = resolve(directory, decodeURIComponent(relative));
    if (!target.startsWith(resolve(directory) + sep)) return json(response, { error: 'outside root' }, 403);
    const info = await stat(target);
    if (!info.isFile()) return json(response, { error: 'not a file' }, 404);
    response.writeHead(200, { 'content-type': mime[extname(target)] ?? 'application/octet-stream', 'content-length': info.size });
    response.end(await readFile(target));
  } catch (error) {
    console.error(error);
    if (request.url?.startsWith('/frame/') || request.url?.startsWith('/output/')) failure = error;
    json(response, { error: String(error.stack ?? error) }, error.code === 'ENOENT' ? 404 : 500);
  }
});
async function cdp(websocket, method) {
  const socket = new WebSocket(websocket);
  await new Promise((done, fail) => { socket.onopen = done; socket.onerror = fail; });
  const result = new Promise((done, fail) => {
    const timer = setTimeout(() => fail(new Error(`CDP timeout ${method}`)), 10000);
    socket.onmessage = event => { const data = JSON.parse(event.data); if (data.id === 1) { clearTimeout(timer); done(data.result); } };
    socket.onclose = () => { clearTimeout(timer); if (method === 'Browser.close') done({}); };
  });
  socket.send(JSON.stringify({ id: 1, method }));
  try { return await result; } finally { socket.close(); }
}
try {
  await new Promise((done, fail) => { server.once('error', fail); server.listen(port, '127.0.0.1', done); });
  const flags = ['--headless=new', '--enable-unsafe-webgpu', `--user-data-dir=${resolve(out, '.edge-profile')}`,
    '--no-first-run', '--no-default-browser-check', '--disable-extensions', '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows', `--remote-debugging-port=${debugPort}`,
    `http://127.0.0.1:${port}/video.html`];
  state.browserArguments = flags;
  browser = child(edge, flags, 'edge.log', ['ignore', 'ignore', 'pipe']);
  await saveState();
  const deadline = Date.now() + Number(option('--timeout', '1800000'));
  while (!complete && !failure && Date.now() < deadline) {
    if (!browserWebSocket) {
      try { browserWebSocket = (await (await fetch(`http://127.0.0.1:${debugPort}/json/version`)).json()).webSocketDebuggerUrl; } catch {}
    }
    await sleep(500);
  }
  if (failure) throw failure;
  if (!complete) throw new Error('video inference timed out');
  if (complete.status !== 'completed') throw new Error(JSON.stringify(complete));
  if (nextInput !== frameCount || nextOutput !== frameCount) throw new Error(`incomplete frame counts ${nextInput}/${nextOutput}/${frameCount}`);
  const encoderDone = encoder.exitCode === null ? once(encoder, 'exit') : Promise.resolve([encoder.exitCode]);
  encoder.stdin.end();
  const [encodeExit] = await encoderDone;
  if (encodeExit !== 0) throw new Error(`FFmpeg encoder exited ${encodeExit}`);
  const decodeExit = decoder.exitCode ?? (await once(decoder, 'exit'))[0];
  if (decodeExit !== 0) throw new Error(`FFmpeg decoder exited ${decodeExit}`);
  state.status = 'completed'; state.video = outputVideo; state.result = complete;
  console.log(`COMPLETED ${nextOutput} frames: ${outputVideo}`);
} catch (error) {
  state.status = 'failed'; state.error = String(error.stack ?? error); process.exitCode = 1;
  console.error(state.error);
} finally {
  if (browserWebSocket) { try { await cdp(browserWebSocket, 'Browser.close'); state.browserClosed = true; } catch (error) { state.browserCloseError = String(error); } }
  if (!state.browserClosed) browser?.kill();
  decoder?.kill(); encoder?.kill();
  server.closeAllConnections(); await new Promise(done => server.close(done));
  for (const log of logs) log.end();
  state.finishedAt = new Date().toISOString(); await saveState();
}
