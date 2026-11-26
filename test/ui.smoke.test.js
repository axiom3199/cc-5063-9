// UI smoke test: drives the real ui/app.js controls under jsdom and reads the
// window.__app snapshot. This is the "surface-editing window" check called for by
// the task: change a ruling, recompute, and confirm the 3D form and flat pattern
// come from the (patched) kernel and that degenerate input yields honest failure
// feedback rather than non-finite coordinates.
//
// jsdom is an optional dev dependency; without it this file skips so the core
// `npm test` stays dependency-free.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { createRequire } from 'node:module';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const require = createRequire(import.meta.url);

let JSDOM = null;
try { ({ JSDOM } = require('jsdom')); } catch { /* optional */ }

const skip = JSDOM ? false : 'jsdom not installed (run: npm i -D jsdom)';

function ctxStub() {
  const noop = () => {};
  return new Proxy({}, {
    get(_, p) {
      if (p === 'canvas') return { width: 800, height: 400 };
      if (p === 'measureText') return () => ({ width: 10 });
      return noop;
    },
    set() { return true; },
  });
}

async function bootUi() {
  const html = await readFile(join(ROOT, 'ui', 'index.html'), 'utf8');
  const dom = new JSDOM(html, { url: 'http://localhost/', runScripts: 'outside-only', pretendToBeVisual: true });
  const { window } = dom;
  window.HTMLCanvasElement.prototype.getContext = () => ctxStub();
  window.HTMLCanvasElement.prototype.getBoundingClientRect = () => ({ width: 800, height: 400, left: 0, top: 0, right: 800, bottom: 400 });
  window.URL.createObjectURL = () => 'blob:stub';
  window.URL.revokeObjectURL = () => {};
  global.window = window;
  global.document = window.document;
  global.requestAnimationFrame = (f) => setTimeout(f, 0);
  global.devicePixelRatio = 1;
  await import(new URL('../ui/app.js', import.meta.url).href + `?t=${Date.now()}`);
  await new Promise((r) => setTimeout(r, 50));
  return window;
}

function hasNonFinite(o) {
  if (typeof o === 'number') return !Number.isFinite(o);
  if (Array.isArray(o)) return o.some(hasNonFinite);
  if (o && typeof o === 'object') return Object.values(o).some(hasNonFinite);
  return false;
}

test('UI: every preset recomputes with finite snapshot values', { skip }, async () => {
  const window = await bootUi();
  const app = window.__app;
  assert.ok(app, 'window.__app must be exposed');
  const presets = [...window.document.querySelectorAll('#preset option')].map((o) => o.value);
  assert.ok(presets.length >= 6, 'presets must populate the selector');
  for (const key of presets) {
    app.loadPreset(key);
    await new Promise((r) => setTimeout(r, 10));
    const snap = app.snapshot();
    assert.equal(hasNonFinite(snap), false, key + ' produced a non-finite snapshot value');
    const status = window.document.getElementById('status').textContent;
    assert.ok(status.length > 0, key + ' rendered no status');
  }
});

test('UI: editing a ruling drives the kernel and re-unfolds isometrically', { skip }, async () => {
  const window = await bootUi();
  const app = window.__app;
  app.loadPreset('cylinder');
  await new Promise((r) => setTimeout(r, 10));
  const before = app.snapshot();
  // Edit the ruling family: turn the cylinder into a cone (concurrent rulings).
  app.state.params = { ...app.state.params, kind: 'cone', apex: [0, 0, 1.6], scale: 0.5 };
  app.recompute();
  await new Promise((r) => setTimeout(r, 10));
  const after = app.snapshot();
  // The flat shape comes from the edited geometry, not a stale cache.
  assert.equal(after.developable, true);
  assert.equal(after.flattenable, true);
  assert.ok(after.unfold.maxRel < 1e-8, 'edited developable ruling must unfold isometrically: ' + after.unfold.maxRel);
  assert.equal(after.unfold.polyFinite, true);
  assert.notDeepEqual(after.unfold.polyArea, before.unfold.polyArea, 'flat area must reflect the edited ruling');
});

test('UI: degenerate ruling edit yields honest retreat, not distorted coordinates', { skip }, async () => {
  const window = await bootUi();
  const app = window.__app;
  app.loadPreset('cone');
  await new Promise((r) => setTimeout(r, 10));
  // Rulings through one point: ruling length collapses at u=0.5 inside the strip.
  app.state.params = {
    kind: 'direct',
    curve0: { kind: 'line', a: [-1, 0, 0], b: [1, 0, 0] },
    curve1: { kind: 'line', a: [-1, 0, 1], b: [1, 0, -1] },
    nu: 32, nv: 6,
  };
  app.state.nu = 32; app.state.nv = 6;
  app.recompute();
  await new Promise((r) => setTimeout(r, 10));
  const snap = app.snapshot();
  // The kernel must not emit a non-finite pattern: either a finite retreated piece
  // or an explicit withheld failure — never NaN coordinates presented as success.
  assert.equal(hasNonFinite(snap), false, 'degenerate edit must not leak non-finite values');
  if (snap.flattenable) {
    assert.equal(snap.retreated, true, 'a collapsed ruling must be excluded from the retained window');
    assert.equal(snap.unfold.polyFinite, true);
    assert.ok(snap.unfold.maxRel < 1e-8, 'retreated window must be isometric: ' + snap.unfold.maxRel);
  } else {
    assert.ok(snap.failure, 'withheld pattern must name its failure');
    assert.ok(window.document.getElementById('c2d') !== null);
  }
});

test('UI: non-developable edit is reported as NOT DEVELOPABLE, not faked flat', { skip }, async () => {
  const window = await bootUi();
  const app = window.__app;
  app.loadPreset('cone');
  await new Promise((r) => setTimeout(r, 10));
  app.state.params = {
    kind: 'direct',
    curve0: { kind: 'line', a: [-1.2, 0, 0], b: [1.2, 0, 0] },
    curve1: { kind: 'line', a: [-0.35, 1.25, 1.15], b: [0.35, -1.25, 1.15] },
    nu: 48, nv: 6,
  };
  app.state.nu = 48; app.state.nv = 6;
  app.recompute();
  await new Promise((r) => setTimeout(r, 10));
  const snap = app.snapshot();
  assert.equal(snap.developable, false);
  assert.equal(snap.flattenable, false, 'a non-developable strip must not report a valid flat pattern');
  assert.ok(window.document.getElementById('status').textContent.includes('NOT DEVELOPABLE'));
});
