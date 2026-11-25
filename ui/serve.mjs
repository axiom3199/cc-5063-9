import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const PORT = Number(process.env.PORT ?? 5173);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
};

function resolveWithin(base, urlPath) {
  const decoded = decodeURIComponent(urlPath.split('?')[0]);
  const rel = decoded === '/' ? 'index.html' : decoded.replace(/^\/+/, '');
  const p = normalize(join(base, rel));
  return p.startsWith(base + sep) || p === base ? p : null;
}

// Assets are resolved from ui/ first (index.html, app.js) and then from the repo
// root (so a module's relative `../src/...` import resolves to /src/...).
async function locate(urlPath) {
  for (const base of [join(ROOT, 'ui'), ROOT]) {
    const p = resolveWithin(base, urlPath);
    if (!p) continue;
    try {
      const s = await stat(p);
      if (s.isFile()) return p;
      if (s.isDirectory()) {
        const idx = join(p, 'index.html');
        if ((await stat(idx)).isFile()) return idx;
      }
    } catch { /* try next base */ }
  }
  return null;
}

const server = createServer(async (req, res) => {
  const p = await locate(req.url ?? '/');
  if (!p) { res.writeHead(404, { 'content-type': 'text/plain' }).end('not found: ' + req.url); return; }
  try {
    const body = await readFile(p);
    res.writeHead(200, { 'content-type': MIME[extname(p)] ?? 'application/octet-stream' }).end(body);
  } catch {
    res.writeHead(404, { 'content-type': 'text/plain' }).end('not found: ' + req.url);
  }
});

server.listen(PORT, () => {
  console.log(`developable-unfold UI  ->  http://localhost:${PORT}/`);
  console.log(`serving ${ROOT}`);
});
