// Serve the original OpenDLSS modules and collect a standalone benchmark result.
// No source files in OpenDLSS are changed by this harness.
import { createServer } from 'node:http';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { basename, extname, isAbsolute, relative, resolve, sep } from 'node:path';

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.json': 'application/json; charset=utf-8',
  '.wgsl': 'text/plain; charset=utf-8', '.bin': 'application/octet-stream',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.wasm': 'application/wasm',
};
const BODY_LIMIT = 32 * 1024 * 1024;

function json(response, value, status = 200) {
  response.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' });
  response.end(JSON.stringify(value));
}

async function jsonBody(request) {
  const chunks = [];
  let length = 0;
  for await (const chunk of request) {
    length += chunk.length;
    if (length > BODY_LIMIT) {
      const error = new Error('JSON body exceeds 32 MiB');
      error.statusCode = 413;
      throw error;
    }
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

function containedPath(directory, pathname) {
  const target = resolve(directory, decodeURIComponent(pathname));
  const suffix = relative(resolve(directory), target);
  if (suffix === '..' || suffix.startsWith(`..${sep}`) || isAbsolute(suffix)) {
    const error = new Error('static path is outside its mounted directory');
    error.statusCode = 403;
    throw error;
  }
  return target;
}

export async function createBenchServer(options) {
  const { harness, upstream, weights, input, output, width, height, warmups, iterations, port } = options;
  await mkdir(output, { recursive: true });
  const config = {
    width, height, warmups, iterations,
    image: `/inputs/${encodeURIComponent(basename(input))}`,
    weights: '/weights',
  };
  const state = { status: 'ready', stage: 'ready', startedAt: new Date().toISOString(), config, events: [] };
  const event = value => {
    const entry = { at: new Date().toISOString(), ...value };
    state.events.push(entry);
    if (state.events.length > 250) state.events.shift();
    options.onEvent?.(entry);
  };
  const server = createServer(async (request, response) => {
    try {
      const url = new URL(request.url, 'http://127.0.0.1');
      if (request.method === 'GET' && url.pathname === '/config') return json(response, config);
      if (request.method === 'GET' && url.pathname === '/status') return json(response, state);
      if (url.pathname === '/favicon.ico') { response.writeHead(204); return response.end(); }
      if (request.method === 'POST' && url.pathname === '/progress') {
        const update = await jsonBody(request);
        state.status = update.status ?? 'running';
        state.stage = update.stage ?? update.message ?? state.stage;
        event(update);
        return json(response, { ok: true });
      }
      if (request.method === 'POST' && url.pathname === '/result') {
        const result = await jsonBody(request);
        if (!['completed', 'failed'].includes(result.status)) throw new Error('result status must be completed or failed');
        const { pngBase64, inputPngBase64, headBase64, ...metadata } = result;
        if (pngBase64) await writeFile(resolve(output, 'output.png'), Buffer.from(pngBase64, 'base64'));
        if (inputPngBase64) await writeFile(resolve(output, 'browser_input.png'), Buffer.from(inputPngBase64, 'base64'));
        if (headBase64) await writeFile(resolve(output, 'head.f32'), Buffer.from(headBase64, 'base64'));
        const host = options.hostEnvironment ?? {};
        const hostGpuName = host.gpu_query?.split(/\r?\n/)[1]?.split(',')[0]?.trim();
        metadata.environment = { ...(metadata.environment ?? {}), host,
          host_gpu_query: host.gpu_query ?? null,
        };
        if (!metadata.environment.gpu_name && hostGpuName) metadata.environment.gpu_name = hostGpuName;
        metadata.input_file_sha256 = host.input_sha256 ?? null;
        metadata.harness = {
          source_root: upstream, weights_directory: weights, input_file: input,
          output_directory: output, server_host: '127.0.0.1',
        };
        await writeFile(resolve(output, 'results.json'), JSON.stringify(metadata, null, 2) + '\n');
        state.status = metadata.status;
        state.stage = metadata.stage ?? metadata.status;
        state.result = metadata;
        event({ status: metadata.status, stage: state.stage, message: 'result saved' });
        return json(response, { ok: true });
      }
      if (!['GET', 'HEAD'].includes(request.method)) return json(response, { error: 'method not allowed' }, 405);
      let directory = harness;
      let pathname = url.pathname === '/' ? 'bench.html' : url.pathname.slice(1);
      if (url.pathname.startsWith('/open-dlss/')) {
        directory = upstream; pathname = url.pathname.slice('/open-dlss/'.length);
      } else if (url.pathname.startsWith('/weights/')) {
        directory = weights; pathname = url.pathname.slice('/weights/'.length);
      } else if (url.pathname.startsWith('/inputs/')) {
        directory = resolve(input, '..'); pathname = url.pathname.slice('/inputs/'.length);
      }
      const path = containedPath(directory, pathname);
      const info = await stat(path);
      if (!info.isFile()) return json(response, { error: 'not a file' }, 404);
      response.writeHead(200, {
        'content-type': MIME[extname(path).toLowerCase()] ?? 'application/octet-stream',
        'content-length': info.size, 'cache-control': 'no-store',
      });
      if (request.method === 'HEAD') return response.end();
      response.end(await readFile(path));
    } catch (error) {
      if (!response.headersSent) json(response, { error: String(error.stack ?? error) }, error.statusCode ?? (error.code === 'ENOENT' ? 404 : 500));
      else response.destroy(error);
    }
  });
  await new Promise((done, fail) => {
    server.once('error', fail);
    server.listen(port, '127.0.0.1', () => { server.off('error', fail); done(); });
  });
  return {
    server, state, config,
    url: `http://127.0.0.1:${server.address().port}/`,
    async close() {
      server.closeAllConnections();
      await new Promise(done => server.close(done));
    },
  };
}
