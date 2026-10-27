import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, stat } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const harness = resolve(fileURLToPath(new URL('.', import.meta.url)));
const project = resolve(harness, '..');
const upstream = resolve(process.env.NR_UPSTREAM ?? `${project}/../..`);
const artifacts = resolve(process.env.NR_ARTIFACTS ?? `${project}/artifacts`);
const weights = resolve(process.env.NR_WEIGHTS ?? `${project}/models/nr`);
const port = Number(process.env.PORT ?? process.argv[2] ?? 8099);
const state = { status: 'ready', startedAt: new Date().toISOString(), events: [] };
const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.json': 'application/json', '.wgsl': 'text/plain',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.bin': 'application/octet-stream' };

await mkdir(artifacts, { recursive: true });
function event(value) {
  state.events.push({ at: new Date().toISOString(), ...value });
  if (state.events.length > 300) state.events.shift();
  console.log(JSON.stringify(state.events.at(-1)));
}
async function jsonBody(request) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > 128 * 1024 * 1024) throw new Error('request body exceeds 128 MiB');
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}
function json(response, value, code = 200) {
  response.writeHead(code, { 'content-type': 'application/json', 'cache-control': 'no-store' });
  response.end(JSON.stringify(value));
}
const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://localhost');
    if (request.method === 'GET' && url.pathname === '/status') return json(response, state);
    if (request.method === 'POST' && url.pathname === '/progress') {
      const entry = await jsonBody(request);
      state.status = entry.status ?? 'running';
      event(entry);
      return json(response, { ok: true });
    }
    if (request.method === 'POST' && url.pathname === '/result') {
      const result = await jsonBody(request);
      const prefix = String(result.prefix ?? 'single');
      if (!/^[a-zA-Z0-9_-]{1,64}$/.test(prefix)) throw new Error('invalid output prefix');
      const { pngBase64, inputPngBase64, headBase64, ...metadata } = result;
      if (pngBase64) await writeFile(`${artifacts}/${prefix}-enhanced.png`, Buffer.from(pngBase64, 'base64'));
      if (inputPngBase64) await writeFile(`${artifacts}/${prefix}-input.png`, Buffer.from(inputPngBase64, 'base64'));
      if (headBase64) await writeFile(`${artifacts}/${prefix}-head.f32`, Buffer.from(headBase64, 'base64'));
      await writeFile(`${artifacts}/${prefix}-runtime.json`, JSON.stringify(metadata, null, 2) + '\n');
      state.status = result.status;
      state.result = metadata;
      event({ status: result.status, message: 'result saved', prefix });
      return json(response, { ok: true, prefix });
    }
    let directory;
    let relative;
    if (url.pathname.startsWith('/open-dlss/')) {
      directory = upstream; relative = url.pathname.slice('/open-dlss/'.length);
    } else if (url.pathname.startsWith('/weights/')) {
      directory = weights; relative = url.pathname.slice('/weights/'.length);
    } else if (url.pathname.startsWith('/artifacts/')) {
      directory = artifacts; relative = url.pathname.slice('/artifacts/'.length);
    } else {
      directory = harness; relative = url.pathname === '/' ? 'single.html' : url.pathname.slice(1);
    }
    const target = resolve(directory, decodeURIComponent(relative));
    if (target !== directory && !target.startsWith(directory + sep)) return json(response, { error: 'outside root' }, 403);
    const info = await stat(target);
    if (!info.isFile()) return json(response, { error: 'not a file' }, 404);
    response.writeHead(200, { 'content-type': mime[extname(target)] ?? 'application/octet-stream',
      'content-length': info.size, 'cache-control': 'no-store' });
    if (request.method === 'HEAD') return response.end();
    response.end(await readFile(target));
  } catch (error) {
    json(response, { error: String(error.stack ?? error) }, error.code === 'ENOENT' ? 404 : 500);
  }
});
server.listen(port, '0.0.0.0', () => {
  event({ status: 'ready', message: 'server listening', port, project, weights, artifacts });
  console.log(`single image: http://localhost:${port}/single.html?image=/artifacts/input.png`);
  console.log(`GPU probe:    http://localhost:${port}/probe.html`);
});
