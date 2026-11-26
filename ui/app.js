import { PRESETS } from '../src/curves.js';
import { buildModel, measureDevelopability, diagnose, sampleMesh } from '../src/ruled.js';
import { unfoldStrip, planTabs, tradeoffSegments, compareTabFeasibility, exportOBJ, exportParamsJSON } from '../src/pipeline.js';
import { polygonArea } from '../src/layout.js';
import { sub, cross, unit } from '../src/math.js';
import { pairingDiagnostics } from '../src/correspondence.js';

const $ = (id) => document.getElementById(id);
// JSON-safe copy: non-finite numbers become explicit strings (e.g. "Infinity")
// so a snapshot never silently reports a real coordinate failure as "unknown".
function jsonSafe(v) {
  if (typeof v === 'number') return Number.isFinite(v) ? v : String(v);
  if (Array.isArray(v)) return v.map(jsonSafe);
  if (v && typeof v === 'object') { const o = {}; for (const k of Object.keys(v)) o[k] = jsonSafe(v[k]); return o; }
  return v;
}
const state = {
  presetKey: 'cylinder',
  params: null,
  nu: 64, nv: 6,
  tabDepth: 0.34, nTabs: 3,
  angleTol: 1e-3,
  view3d: { yaw: 0.7, pitch: 0.35, zoom: 1, panx: 0, pany: 0 },
  result: null,
};

// ---------- preset / parameter state ----------
function loadPreset(key) {
  const p = PRESETS[key]();
  state.presetKey = key;
  state.params = JSON.parse(JSON.stringify(p.model));
  state.nu = p.model.nu ?? 64;
  state.nv = p.model.nv ?? 6;
  $('pNu').value = state.nu;
  $('pNv').value = state.nv;
  $('preset').value = key;
  $('presetNote').textContent = p.note || '';
  recompute();
}

function currentModel() {
  const m = JSON.parse(JSON.stringify(state.params));
  m.nu = state.nu; m.nv = state.nv;
  return buildModel(m);
}

// ---------- recompute ----------
let recomputeTimer = null;
function schedule() { clearTimeout(recomputeTimer); recomputeTimer = setTimeout(recompute, 40); }

function recompute() {
  const tStart = performance.now();
  const model = currentModel();
  const dev = measureDevelopability(model, { nu: Math.max(60, state.nu * 2), angleTol: state.angleTol });
  const diag = diagnose(model, { nu: Math.max(80, state.nu * 2) });

  // The pipeline owns the degeneracy policy: unfold the requested span, and if a
  // degenerate ruling makes the coordinates unplaceable it retreats to the largest
  // singular-free window (and reports what it excluded). The UI never re-implements
  // that (an earlier version did, ignoring collapsed rulings, and rendered distorted
  // patterns). Non-developable strain is NOT clipped: it is measured and shown.
  const piece = unfoldStrip(model, { nu: state.nu, nv: state.nv });
  const flattenable = piece.ok;
  const withheld = !piece.ok;
  const uA = piece.uA, uB = piece.uB;
  const clipped = piece.retreated;
  const clipNote = piece.retreated
    ? `valid window u∈[${uA.toFixed(3)}, ${uB.toFixed(3)}] (degenerate locus excluded)`
    : '';

  const layout = flattenable
    ? planTabs(piece, [
        { onSide: 'u0', nTabs: state.nTabs, priority: 1, label: 'seam-u0' },
        { onSide: 'u1', nTabs: state.nTabs, priority: 1, label: 'seam-u1' },
      ], { tabDepth: state.tabDepth })
    : null;

  let corrInfo = null;
  if (model.kind === 'direct' && model.cfg.curve0 && model.cfg.curve1) {
    corrInfo = pairingDiagnostics(model.cfg.curve0, model.cfg.curve1, { nu: 48 });
  }

  state.result = {
    model, dev, diag, piece, layout, corrInfo, clipped, clipNote, uA, uB,
    flattenable, withheld, failure: piece.failure, excluded: piece.excluded,
    ms: performance.now() - tStart,
  };
  renderStatus();
  renderLayout();
  drawAll();
}

// ---------- status panel ----------
const pill = (text, cls) => `<span class="pill ${cls}">${text}</span>`;
const fmt = (x, d = 2) => (Number.isFinite(x) ? x.toExponential(d) : '—');

function renderStatus() {
  const { dev, diag, piece, corrInfo, clipped, clipNote } = state.result;
  const rows = [];

  let verdict;
  if (!dev.developable) verdict = pill('NOT DEVELOPABLE (measured)', 'bad');
  else if (diag.regressionEdge) verdict = pill('SINGULAR — regression edge', 'warn');
  else if (diag.hasSingularity) verdict = pill('DEVELOPABLE, but has a degenerate ruling', 'warn');
  else verdict = pill('DEVELOPABLE (measured)', 'good');
  rows.push(`<div style="margin-bottom:6px">${verdict}</div>`);
  if (!dev.developable && diag.regressionEdge) {
    rows.push(`<div style="margin-bottom:6px">${pill('singular locus located', 'warn')}</div>`);
  }

  rows.push(`<table>
    <tr><td class="k">tangent-plane max angle</td><td>${fmt(dev.maxAngle)}</td></tr>
    <tr><td class="k">det-condition residual</td><td>${fmt(dev.maxDet)}</td></tr>
    <tr><td class="k">angle tolerance</td><td>${state.angleTol.toExponential(0)}</td></tr>
    <tr><td class="k">violating rulings</td><td>${dev.badRulings.length}</td></tr>
  </table>`);

  if (corrInfo) {
    const ci = corrInfo.illConditioned
      ? (corrInfo.flat ? 'FLAT in v (pairing underdetermined)' : 'ill-conditioned')
      : 'well-conditioned';
    const col = corrInfo.illConditioned ? 'var(--warn)' : 'var(--good)';
    rows.push(`<div class="hd" style="margin-top:10px">Correspondence solve</div>`);
    rows.push(`<table>
      <tr><td class="k">conditioning</td><td style="color:${col}">${ci}</td></tr>
      <tr><td class="k">F residual</td><td>${fmt(corrInfo.maxResidual)}</td></tr>
      <tr><td class="k">near-developable</td><td>${corrInfo.nearDevelopable ? 'yes' : 'no'}</td></tr>
    </table>`);
  }

  if (diag.regressionEdge) {
    const re = diag.regressionEdge;
    rows.push(`<div style="margin-top:8px;color:var(--bad);font-size:11px">regression edge located: u∈[${re.uMin.toFixed(2)}, ${re.uMax.toFixed(2)}] at v≈${re.vActual.toFixed(3)}</div>`);
  } else if (diag.singular.length) {
    rows.push(`<div class="hint" style="color:var(--warn)">${diag.singular.length} singular sample(s) located</div>`);
  }
  if (clipped) rows.push(`<div class="hint" style="color:var(--warn)">retreated: ${clipNote}</div>`);

  if (piece && piece.ok) {
    const e = piece.error;
    const ok = e.maxRel < 1e-5 && e.flips === 0;
    rows.push(`<div class="hd" style="margin-top:10px">Unfold error (correspondence-measured)</div>`);
    rows.push(`<table>
      <tr><td class="k">max edge rel</td><td style="color:${ok ? 'var(--good)' : 'var(--bad)'}">${fmt(e.maxRel)}</td></tr>
      <tr><td class="k">rms edge rel</td><td>${fmt(e.rmsRel)}</td></tr>
      <tr><td class="k">max angle err</td><td>${e.maxAngleDeg.toExponential(2)}°</td></tr>
      <tr><td class="k">area rel</td><td style="color:${e.areaRel < 1e-5 ? 'var(--good)' : 'var(--bad)'}">${fmt(e.areaRel)}</td></tr>
      <tr><td class="k">folded triangles</td><td style="color:${e.flips ? 'var(--bad)' : 'var(--good)'}">${e.flips}</td></tr>
      <tr><td class="k">3D / 2D area</td><td>${e.areaTotal3.toFixed(3)} / ${e.areaTotal2.toFixed(3)}</td></tr>
    </table>`);
  } else {
    const why = piece ? failureText(state.result.failure) : 'no flattenable window';
    rows.push(`<div style="margin-top:8px;color:var(--bad);font-size:11px">flat pattern withheld — ${why}</div>`);
    rows.push(`<div class="hint">the degenerate region is located and reported; no distorted pattern is emitted in its place.</div>`);
  }
  $('status').innerHTML = rows.join('');
}

function failureText(f) {
  return ({
    'unplaced-vertices': 'a collapsed ruling left vertices with no finite 2D image',
    'nonfinite-coordinates': 'non-finite coordinates from a degenerate ruling',
    'folded-triangles': 'the strip would fold over itself (genuine strain / singularity)',
    'nonfinite-error': 'the unfolded metric is undefined',
    'strained': 'the unfolded metric is strained beyond tolerance',
  })[f] || 'the request is not flattenable';
}

function renderLayout() {
  const { piece, layout } = state.result;
  if (!piece || !layout) { $('layout').innerHTML = '<span class="hint">—</span>'; return; }
  const oriented = layout.tabs.every((t) => t.orientation === 1);
  $('layout').innerHTML = `<table>
    <tr><td class="k">intended tabs</td><td>${layout.counts.intended}</td></tr>
    <tr><td class="k">tabs placed</td><td style="color:${layout.counts.tabs ? 'var(--good)' : 'var(--muted)'}">${layout.counts.tabs}</td></tr>
    <tr><td class="k">slits (cut retreat)</td><td style="color:${layout.counts.slits ? 'var(--warn)' : 'var(--good)'}">${layout.counts.slits}</td></tr>
    <tr><td class="k">coverage</td><td>${(layout.coverage * 100).toFixed(0)}%</td></tr>
    <tr><td class="k">orientation</td><td style="color:${oriented ? 'var(--good)' : 'var(--bad)'}">${oriented ? 'all exterior' : 'MIXED'}</td></tr>
  </table>`;
}

// ---------- canvases ----------
const c3d = $('c3d'), ctx3 = c3d.getContext('2d');
const c2d = $('c2d'), ctx2 = c2d.getContext('2d');

function fitCanvas(canvas) {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  }
  return { w, h, dpr };
}

function project3d(p, cam, cx, cy, scale) {
  const cyaw = Math.cos(cam.yaw), syaw = Math.sin(cam.yaw);
  const x1 = p[0] * cyaw - p[1] * syaw, y1 = p[0] * syaw + p[1] * cyaw, z1 = p[2];
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const z2 = y1 * sp + z1 * cp;
  return { x: cx + x1 * scale, y: cy - z2 * scale };
}

function drawAll() { draw3d(); draw2d(); }

function draw3d() {
  const { w, h, dpr } = fitCanvas(c3d);
  ctx3.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx3.clearRect(0, 0, w, h);
  const { model, dev, diag, uA, uB } = state.result;

  const nu = state.nu, nv = state.nv;
  const mesh = sampleMesh(model, nu, nv);

  let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity, minz = Infinity, maxz = -Infinity;
  for (const row of mesh.rows) for (const p of row) {
    minx = Math.min(minx, p[0]); maxx = Math.max(maxx, p[0]);
    miny = Math.min(miny, p[1]); maxy = Math.max(maxy, p[1]);
    minz = Math.min(minz, p[2]); maxz = Math.max(maxz, p[2]);
  }
  const ctr = [(minx + maxx) / 2, (miny + maxy) / 2, (minz + maxz) / 2];
  const span = Math.max(maxx - minx, maxy - miny, maxz - minz, 1e-6);
  const scale = (Math.min(w, h) * 0.72 / span) * state.view3d.zoom;
  const cx = w / 2 + state.view3d.panx, cy = h / 2 + state.view3d.pany;
  const proj = (p) => project3d([p[0] - ctr[0], p[1] - ctr[1], p[2] - ctr[2]], state.view3d, cx, cy, scale);

  const uNorm = (uu) => (uu - model.uMin) / model.uSpan;
  const badSet = new Set();
  for (const r of dev.perRuling) {
    if (r.degenerate) continue;
    const idx = Math.round(uNorm(r.u) * nu);
    if (r.angle > state.angleTol || r.detRes > dev.detTol) badSet.add(idx);
  }

  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nv; j++) {
      const a = proj(mesh.rows[i][j]), b = proj(mesh.rows[i + 1][j]);
      const c = proj(mesh.rows[i + 1][j + 1]), d = proj(mesh.rows[i][j + 1]);
      const nrm = surfaceNormalAt(mesh, i, j);
      const shade = 0.4 + 0.6 * Math.abs(nrm[2]);
      const isBad = badSet.has(i);
      const col = isBad
        ? `rgba(${Math.round(225 * shade)},${Math.round(95 * shade)},${Math.round(85 * shade)},0.55)`
        : `rgba(${Math.round(90 * shade + 45)},${Math.round(150 * shade + 45)},${Math.round(210 * shade + 35)},0.42)`;
      ctx3.beginPath();
      ctx3.moveTo(a.x, a.y); ctx3.lineTo(b.x, b.y); ctx3.lineTo(c.x, c.y); ctx3.lineTo(d.x, d.y); ctx3.closePath();
      ctx3.fillStyle = col; ctx3.fill();
    }
  }

  ctx3.strokeStyle = 'rgba(200,215,235,0.26)'; ctx3.lineWidth = 0.7;
  const stride = Math.max(1, Math.floor(nu / 26));
  for (let i = 0; i <= nu; i += stride) {
    const a = proj(mesh.rows[i][0]), b = proj(mesh.rows[i][nv]);
    ctx3.beginPath(); ctx3.moveTo(a.x, a.y); ctx3.lineTo(b.x, b.y); ctx3.stroke();
  }

  for (const jb of [0, nv]) {
    ctx3.strokeStyle = jb === 0 ? '#6db3f2' : '#5fd08a'; ctx3.lineWidth = 2;
    ctx3.beginPath();
    for (let i = 0; i <= nu; i++) {
      const q = proj(mesh.rows[i][jb]);
      if (i === 0) ctx3.moveTo(q.x, q.y); else ctx3.lineTo(q.x, q.y);
    }
    ctx3.stroke();
  }

  // active u-window, when clipped to avoid a singular locus
  if (uA > model.uMin + 1e-6 || uB < model.uMax - 1e-6) {
    ctx3.strokeStyle = 'rgba(240,184,102,0.9)'; ctx3.lineWidth = 2; ctx3.setLineDash([6, 5]);
    for (const ub of [uA, uB]) {
      const a = proj(model.point(ub, 0)), b = proj(model.point(ub, 1));
      ctx3.beginPath(); ctx3.moveTo(a.x, a.y); ctx3.lineTo(b.x, b.y); ctx3.stroke();
    }
    ctx3.setLineDash([]);
  }

  // regression edge marker
  if (diag.regressionEdge) {
    ctx3.fillStyle = '#ef6b6b';
    const re = diag.regressionEdge;
    const v0 = model.cfg.v0 ?? 0, v1 = model.cfg.v1 ?? 1;
    const vN = (re.vActual - v0) / (v1 - v0);
    for (let i = 0; i <= 24; i++) {
      const u = re.uMin + (re.uMax - re.uMin) * i / 24;
      const p = model.point(u, Math.max(0, Math.min(1, vN)));
      const q = proj(p);
      ctx3.beginPath(); ctx3.arc(q.x, q.y, 2.3, 0, 7); ctx3.fill();
    }
  }
}

function surfaceNormalAt(mesh, i, j) {
  const p00 = mesh.rows[i][j], p10 = mesh.rows[i + 1][j], p01 = mesh.rows[i][j + 1];
  return unit(cross(sub(p10, p00), sub(p01, p00)));
}

function draw2d() {
  const { w, h, dpr } = fitCanvas(c2d);
  ctx2.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx2.clearRect(0, 0, w, h);
  const { piece, layout, failure } = state.result;

  if (!piece || !piece.ok) {
    ctx2.fillStyle = '#ef6b6b'; ctx2.font = '13px ui-monospace';
    ctx2.fillText('flat pattern withheld — ' + failureText(failure), 16, 30);
    ctx2.fillStyle = '#8b95a5'; ctx2.font = '11px ui-monospace';
    ctx2.fillText('degenerate region located; no distorted coordinates emitted', 16, 50);
    return;
  }

  const poly = piece.outline.poly;
  const xs = poly.map((p) => p[0]), ys = poly.map((p) => p[1]);
  const b = {
    xmin: Math.min(...xs), ymin: Math.min(...ys),
    w: Math.max(...xs) - Math.min(...xs), h: Math.max(...ys) - Math.min(...ys),
  };
  const pad = 48;
  const s = Math.min((w - pad * 2) / Math.max(b.w, 1e-9), (h - pad * 2) / Math.max(b.h, 1e-9));
  const ox = (w - b.w * s) / 2 - b.xmin * s;
  const oy = (h - b.h * s) / 2 - b.ymin * s;
  const T = (p) => [p[0] * s + ox, h - (p[1] * s + oy)];

  ctx2.beginPath();
  for (let i = 0; i < poly.length; i++) { const q = T(poly[i]); if (i === 0) ctx2.moveTo(q[0], q[1]); else ctx2.lineTo(q[0], q[1]); }
  ctx2.closePath();
  ctx2.fillStyle = 'rgba(109,179,242,0.16)'; ctx2.fill();
  ctx2.strokeStyle = '#6db3f2'; ctx2.lineWidth = 2; ctx2.stroke();

  if (piece.topology) {
    ctx2.strokeStyle = 'rgba(200,215,235,0.16)'; ctx2.lineWidth = 0.7;
    for (const r of piece.topology.rulings) {
      const a = T(r[0]), c = T(r[1]);
      ctx2.beginPath(); ctx2.moveTo(a[0], a[1]); ctx2.lineTo(c[0], c[1]); ctx2.stroke();
    }
    const nu = piece.mesh.nu, nv = piece.mesh.nv;
    for (let i = 1; i < nu; i++) {
      ctx2.beginPath();
      for (let j = 0; j <= nv; j++) {
        const k = i * (nv + 1) + j;
        const q = T([piece.unfold.pos[2 * k], piece.unfold.pos[2 * k + 1]]);
        if (j === 0) ctx2.moveTo(q[0], q[1]); else ctx2.lineTo(q[0], q[1]);
      }
      ctx2.stroke();
    }
  }

  if (layout) {
    for (const sl of layout.slits) {
      const pp = T(sl.base[0]), qq = T(sl.base[1]);
      ctx2.strokeStyle = '#ef6b6b'; ctx2.lineWidth = 2.6; ctx2.setLineDash([5, 4]);
      ctx2.beginPath(); ctx2.moveTo(pp[0], pp[1]); ctx2.lineTo(qq[0], qq[1]); ctx2.stroke();
      ctx2.setLineDash([]);
    }
    for (const t of layout.tabs) {
      ctx2.beginPath();
      t.tab.forEach((p, i) => { const q = T(p); if (i === 0) ctx2.moveTo(q[0], q[1]); else ctx2.lineTo(q[0], q[1]); });
      ctx2.closePath();
      ctx2.fillStyle = 'rgba(224,164,74,0.30)'; ctx2.fill();
      ctx2.strokeStyle = '#e0a44a'; ctx2.lineWidth = 1.4; ctx2.stroke();
    }
  }

  const e = piece.error;
  ctx2.fillStyle = '#8b95a5'; ctx2.font = '10.5px ui-monospace';
  ctx2.fillText(`flat area ${Math.abs(polygonArea(poly)).toFixed(4)}`, 12, h - 12);
  ctx2.fillText(`max edge err ${e.maxRel.toExponential(2)}`, 12, h - 28);
  if (e.maxRel > 1e-4 || e.flips > 0) {
    ctx2.fillStyle = '#ef6b6b';
    ctx2.fillText(`STRAINED: ${e.flips} folded tris, edge err ${e.maxRel.toExponential(1)}`, 12, h - 44);
  }
}

// ---------- interaction ----------
let drag = null;
c3d.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY }; c3d.setPointerCapture(e.pointerId); });
c3d.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag = { x: e.clientX, y: e.clientY };
  state.view3d.yaw += dx * 0.01;
  state.view3d.pitch = Math.max(-1.4, Math.min(1.4, state.view3d.pitch + dy * 0.01));
  draw3d();
});
c3d.addEventListener('pointerup', () => { drag = null; });
c3d.addEventListener('pointercancel', () => { drag = null; });
c3d.addEventListener('wheel', (e) => { e.preventDefault(); state.view3d.zoom *= e.deltaY < 0 ? 1.1 : 0.9; draw3d(); }, { passive: false });
window.addEventListener('resize', drawAll);

$('preset').addEventListener('change', (e) => loadPreset(e.target.value));
$('pNu').addEventListener('input', (e) => { state.nu = clampInt(e.target.value, 8, 400); schedule(); });
$('pNv').addEventListener('input', (e) => { state.nv = clampInt(e.target.value, 2, 24); schedule(); });
$('pTabDepth').addEventListener('input', (e) => { state.tabDepth = clampNum(e.target.value, 0.05, 0.6); schedule(); });
$('pNTabs').addEventListener('input', (e) => { state.nTabs = clampInt(e.target.value, 1, 8); schedule(); });
$('pAngleTol').addEventListener('input', (e) => { state.angleTol = clampNum(e.target.value, 1e-6, 1); schedule(); });

$('btnTradeoff').addEventListener('click', () => {
  const model = currentModel();
  const tr = tradeoffSegments(model, { nu: state.nu, nv: state.nv, segments: [1, 2, 3, 4, 6] });
  const one = tr[0];
  const best = tr.reduce((a, b) => (b.maxRel < a.maxRel ? b : a));
  const helps = best.segments > 1 && best.maxRel < one.maxRel * 0.99;
  const lines = tr.map((t) => `  ${t.segments} piece(s): max edge err ${t.maxRel.toExponential(2)}  rms ${t.rmsRel.toExponential(2)}`);
  const verdict = helps
    ? `DECISION: segment into ${best.segments} — measured unfolded edge error drops ${((1 - best.maxRel / one.maxRel) * 100).toFixed(0)}%.`
    : `DECISION: keep one piece and add a slit — segmenting does NOT lower the measured error (${one.maxRel.toExponential(2)} → ${best.maxRel.toExponential(2)}).`;
  $('tradeoffOut').innerHTML = `<pre style="white-space:pre-wrap;font-size:10.5px;color:var(--muted);margin:0">${lines.join('\n')}\n${verdict}</pre>`;
});

$('btnAB').addEventListener('click', () => {
  const model = currentModel();
  // Each re-sliced half is a narrow sector; its two seam rulings are short and meet
  // near the apex, so it packs one tab per seam cleanly (nTabsSplit=1). Demanding
  // more per seam hits a genuine packing limit and confuses the A/B decision, which
  // is about tab feasibility given similar unfold error, not about cramming tabs.
  let cmp;
  try {
    cmp = compareTabFeasibility(model, { nu: state.nu, nv: state.nv, tabDepth: state.tabDepth, nTabs: state.nTabs, nTabsSplit: 1 });
  } catch (e) {
    $('abOut').innerHTML = `<pre style="white-space:pre-wrap;font-size:10.5px;color:var(--bad);margin:0">A/B comparison unavailable: ${e.message}</pre>`;
    return;
  }
  const wE = cmp.whole.error.rmsRel, sE = Math.max(cmp.split.A.error.rmsRel, cmp.split.B.error.rmsRel);
  // machine-precision errors compare as "similar" on an absolute floor, matching
  // compareTabFeasibility.errorsSimilar (a pure ratio would split round-off).
  const sim = cmp.errorsSimilar;
  const orient = cmp.gluedPairs.map((g) => `${g.piece}:${g.edge}${g.orientation === 1 ? '↑out' : '↓IN'}`);
  const lines = [
    `whole: ${cmp.whole.layout.counts.tabs} tabs / ${cmp.whole.layout.counts.slits} slits  (rms err ${wE.toExponential(2)})`,
    `split: A ${cmp.split.layoutA.counts.tabs}/${cmp.split.layoutA.counts.slits}  B ${cmp.split.layoutB.counts.tabs}/${cmp.split.layoutB.counts.slits}  (rms err ${sE.toExponential(2)})`,
    `errors similar: ${sim ? 'YES' : 'no'}   whole fits: ${cmp.wholeFits ? 'yes' : 'NO'}   split fits: ${cmp.splitFits ? 'YES' : 'no'}`,
    `glued pairs: ${orient.join(', ')}`,
    `locked fold: ${cmp.lockedFolds[0]?.label} (cut in flat pattern, glued back after assembly)`,
    cmp.splitFits && !cmp.wholeFits
      ? 'DECISION: re-slice — similar measured error, but only the split accommodates every tab.'
      : cmp.wholeFits && cmp.splitFits
        ? 'DECISION: single piece is tab-feasible here; re-slicing is not required.'
        : !cmp.wholeFits && !cmp.splitFits
          ? 'DECISION: neither the whole nor the split accommodates every tab at this tab depth / count; retreat a cut to a slit or reduce tabs.'
          : 'DECISION: the whole piece already fits; the split is not needed.',
  ];
  $('abOut').innerHTML = `<pre style="white-space:pre-wrap;font-size:10.5px;color:var(--muted);margin:0">${lines.join('\n')}</pre>`;
});

$('btnObj').addEventListener('click', () => {
  const { model, piece, failure } = state.result;
  if (!piece || !piece.ok) { log('OBJ export skipped: flat pattern withheld (' + failureText(failure) + ').'); return; }
  download(`developable-${state.presetKey}.obj`, exportOBJ(model, piece));
});
$('btnJson').addEventListener('click', () => {
  const { model } = state.result;
  download(`developable-${state.presetKey}.json`, exportParamsJSON(model));
});
$('btnPng').addEventListener('click', () => {
  const out = document.createElement('canvas');
  out.width = c3d.width + c2d.width; out.height = Math.max(c3d.height, c2d.height);
  const octx = out.getContext('2d');
  octx.fillStyle = '#14161a'; octx.fillRect(0, 0, out.width, out.height);
  octx.drawImage(c3d, 0, 0); octx.drawImage(c2d, c3d.width, 0);
  out.toBlob((blob) => {
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = `developable-${state.presetKey}.png`; a.click();
  });
});

function download(name, text) {
  const blob = new Blob([text], { type: 'text/plain' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click();
}
function clampInt(v, lo, hi) { const n = parseInt(v, 10); return Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : lo; }
function clampNum(v, lo, hi) { const n = parseFloat(v); return Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : lo; }
function log(msg) { $('logtext').textContent = msg + '\n' + $('logtext').textContent; }

// ---------- init ----------
$('legend').innerHTML = `<span class="sw" style="background:#6db3f2"></span>C0 &nbsp; <span class="sw" style="background:#5fd08a"></span>C1 &nbsp; <span class="sw" style="background:#ef6b6b"></span>violating ruling`;

for (const key of Object.keys(PRESETS)) {
  const opt = document.createElement('option');
  opt.value = key; opt.textContent = PRESETS[key]().label;
  $('preset').appendChild(opt);
}
$('preset').value = 'cylinder';
$('pTabDepth').value = state.tabDepth;
$('pNTabs').value = state.nTabs;
$('pAngleTol').value = state.angleTol;
loadPreset('cylinder');

// Plain-JSON snapshot for automated UI verification.
window.__app = {
  state,
  loadPreset,
  recompute,
  drawAll,
  snapshot() {
    const r = state.result;
    if (!r) return null;
    return {
      preset: state.presetKey,
      developable: r.dev.developable,
      maxAngle: r.dev.maxAngle,
      maxDet: r.dev.maxDet,
      badRulings: r.dev.badRulings.length,
      singular: r.diag.singular.length,
      regressionEdge: r.diag.regressionEdge,
      flattenable: r.flattenable,
      failure: r.failure || null,
      retreated: !!r.clipped,
      window: [r.uA, r.uB],
      excluded: r.excluded || [],
      unfold: r.piece && r.piece.ok ? {
        maxRel: r.piece.error.maxRel,
        rmsRel: r.piece.error.rmsRel,
        maxAngleDeg: r.piece.error.maxAngleDeg,
        areaRel: r.piece.error.areaRel,
        flips: r.piece.error.flips,
        nonFinite2d: r.piece.error.nonFinite2d,
        area3: r.piece.error.areaTotal3,
        area2: r.piece.error.areaTotal2,
        polyArea: Math.abs(polygonArea(r.piece.outline.poly)),
        polyFinite: r.piece.outline.poly.every((p) => p.every(Number.isFinite)),
      } : (r.piece ? { ok: false, failure: r.piece.failure, nonFinite2d: r.piece.error.nonFinite2d } : null),
      layout: r.layout ? {
        intended: r.layout.counts.intended,
        tabs: r.layout.counts.tabs,
        slits: r.layout.counts.slits,
        coverage: r.layout.coverage,
        orientations: r.layout.tabs.map((t) => t.orientation),
      } : null,
      // jsonSafe: ill-conditioning is legitimately Infinity; keep it out of raw
      // JSON so automated checks don't read Infinity as a coordinate failure.
      pairing: r.corrInfo ? jsonSafe(r.corrInfo) : null,
    };
  },
};
