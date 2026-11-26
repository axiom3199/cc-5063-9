// Pipeline: model -> measured developability -> intrinsic unfolding -> cut outline
// -> tab layout (with cut retreat). Also implements the A/B comparison of two
// unfoldings with similar edge-length error where only one accommodates tabs.

import { buildModel, measureDevelopability, diagnose, sampleMesh } from './ruled.js';
import { unfoldMesh, measureUnfoldError, bounds2d, transform2d, unfoldTopology } from './unfold.js';
import { layoutTabs, polygonArea, tabOrientation } from './layout.js';
import { createCurve } from './curves.js';
import { dist } from './math.js';

// Unfold the strip for a given u-window [uA,uB].
// Returns the 2D outline (closed polygon) plus per-outline-edge metadata marking
// which edges are the uA / uB seam rulings.
//
// Degeneracy policy:
//   * If the caller names a window (opts.uA/uB) we unfold exactly it; a caller that
//     already retreated gets a straight answer, never a silent re-widening.
//   * If the caller asks for the whole model we first probe; on a finite, fold-free
//     result we keep it, otherwise we retreat to the largest singular-free u-window
//     (from diagnose.validURanges) and unfold there, recording what was excluded.
//     If no usable window remains we return a patch with ok=false and no faked coords.
export function unfoldStrip(model, opts = {}) {
  const nu = opts.nu ?? model.cfg.nu ?? 64;
  const nv = opts.nv ?? model.cfg.nv ?? 6;
  const explicit = opts.uA != null || opts.uB != null;
  const uA = opts.uA ?? model.uMin;
  const uB = opts.uB ?? model.uMax;

  let patch = unfoldWindow(model, uA, uB, nu, nv, opts);
  let retreated = false, excluded = [], diagInfo = null;
  // Retreat ONLY to escape a coordinate/placement failure caused by a degenerate
  // ruling (non-finite images the BFS could not place). We never retreat a merely
  // non-developable surface: folding there is real strain and must stay visible,
  // not be hidden by cropping to the largest "clean" window.
  const coordinateFailure = patch.failure === 'unplaced-vertices' || patch.failure === 'nonfinite-coordinates';
  if (!explicit && coordinateFailure) {
    diagInfo = diagnose(model, { nu: Math.max(80, opts.diagNu ?? Math.max(nu, 120)) });
    const win = chooseDevelopableWindow(diagInfo, opts.minSpan ?? 0.02);
    if (win && (win[0] > model.uMin + 1e-9 || win[1] < model.uMax - 1e-9)) {
      const trial = unfoldWindow(model, win[0], win[1], nu, nv, opts);
      if (trial.ok) {
        patch = trial;
        retreated = true;
        excluded = [{ start: model.uMin, end: win[0] }, { start: win[1], end: model.uMax }]
          .filter((r) => r.end - r.start > 1e-9);
      }
    }
  }
  patch.retreated = retreated;
  patch.excluded = excluded;
  patch.diagWindow = retreated ? diagInfo : null;
  return patch;
}

// The engine behind unfoldStrip for one fixed window. Never widens, never
// fabricates: ok reflects that every vertex got a finite, fold-free image.
function unfoldWindow(model, uA, uB, nu, nv, opts) {
  const sub = subModel(model, uA, uB);
  const mesh = sampleMesh(sub, nu, nv);
  const uf = unfoldMesh(mesh, { diagonal: opts.diagonal || 'main' });
  const err = measureUnfoldError(mesh, uf, opts);
  const outline = outlineOf(mesh, uf);
  const ok = uf.ok && err.nonFinite2d === 0 && Number.isFinite(err.maxRel) && err.flips === 0;
  return {
    model: sub, mesh, unfold: uf, error: err, outline, topology: unfoldTopology(mesh, uf),
    uA, uB, nu, nv, ok, failure: ok ? null : classifyFailure(uf, err),
  };
}

function classifyFailure(uf, err) {
  if (uf.nonFinite > 0) return 'unplaced-vertices';
  if (err.nonFinite2d > 0) return 'nonfinite-coordinates';
  if (err.flips > 0) return 'folded-triangles';
  if (!Number.isFinite(err.maxRel)) return 'nonfinite-error';
  if (err.maxRel >= 1e-4) return 'strained';
  return null;
}

// Largest singular-free window, or null if none survives the minimum span.
export function chooseDevelopableWindow(diagInfo, minSpan = 0.02) {
  const usable = diagInfo.validURanges.filter((r) => r[1] - r[0] > minSpan);
  if (!usable.length) return null;
  return usable.slice().sort((a, b) => (b[1] - b[0]) - (a[1] - a[0]))[0];
}

// Restrict a model's u-range to [uA,uB] by reparametrising. u in [0,1] on the
// returned model maps affinely onto [uA,uB] of the parent. The parent's own
// uMin/uMax is the domain of uA/uB, NOT the sub-range (using it here would make
// the retreat inert and re-emit the singular locus).
export function subModel(model, uA, uB) {
  const lo = uA ?? model.uMin;
  const hi = uB ?? model.uMax;
  const span = hi - lo;
  const inner = (u) => lo + span * u;
  const sub = {
    ...model,
    uMin: 0, uMax: 1,
    uSpan: 1,
    frame: (u) => model.frame(inner(u)),
    point: (u, v) => model.point(inner(u), v),
    ruling: (u) => model.ruling(inner(u)),
    dU: (u, v) => model.dU(inner(u), v),
    dV: (u) => model.dV(inner(u)),
    normal: (u, v) => model.normal(inner(u), v),
    _parent: model,
    _mapU: inner,
    _uA: lo, _uB: hi,
  };
  // rescale derivatives with chain rule (du_inner/du = span)
  sub.dU = (u, v) => {
    const d = model.dU(inner(u), v);
    return [d[0] * span, d[1] * span, d[2] * span];
  };
  sub.normal = (u, v) => model.normal(inner(u), v); // direction unaffected by positive scale
  return sub;
}

// Walk the grid boundary to a closed 2D polygon. Edge metadata identifies the
// two seam edges (the rulings at i=0 and i=nu) vs the two boundary curves.
export function outlineOf(mesh, uf) {
  const { nu, nv } = mesh;
  const vid = (i, j) => i * (nv + 1) + j;
  const pt = (i, j) => [uf.pos[2 * vid(i, j)], uf.pos[2 * vid(i, j) + 1]];
  const verts = [];
  const edges = [];
  const push = (i, j) => { verts.push(pt(i, j)); };
  const pushEdge = (kind, side, i, j) => { edges.push({ kind, side, a: [i, j] }); };

  // bottom boundary j=0, i=0..nu
  for (let i = 0; i <= nu; i++) push(i, 0);
  // right seam i=nu, j=1..nv
  for (let j = 1; j <= nv; j++) push(nu, j);
  // top boundary j=nv, i=nu-1..0
  for (let i = nu - 1; i >= 0; i--) push(i, nv);
  // left seam i=0, j=nv-1..1
  for (let j = nv - 1; j >= 1; j--) push(0, j);

  // Now annotate the outline edges. The outline vertex indices in order:
  // [0..nu] = bottom; [nu+1..nu+nv] = right seam; [nu+nv+1..2nu+nv] = top; ...
  const bottom = [];
  for (let i = 0; i <= nu; i++) bottom.push({ kind: 'boundary', side: 'v0', i });
  const rightSeam = [];
  for (let j = 1; j <= nv; j++) rightSeam.push({ kind: 'seam', side: 'u1', j });
  const top = [];
  for (let i = nu - 1; i >= 0; i--) top.push({ kind: 'boundary', side: 'v1', i });
  const leftSeam = [];
  for (let j = nv - 1; j >= 1; j--) leftSeam.push({ kind: 'seam', side: 'u0', j });
  const meta = [...bottom, ...rightSeam, ...top, ...leftSeam];

  // seam rulings are edges at the transition: the whole right seam is one ruling
  // (j from 0..nv) and whole left seam is one ruling. Outline edge index for the
  // right seam is the first rightSeam edge; left seam the first leftSeam edge.
  const rightSeamStart = bottom.length;              // outline edge index
  const topStart = bottom.length + rightSeam.length;
  const leftSeamStart = topStart + top.length;

  return {
    poly: verts,
    meta,
    edgeIndex: {
      v0: [0, bottom.length - 1],
      u1: [rightSeamStart, rightSeamStart + rightSeam.length - 1],
      v1: [topStart, topStart + top.length - 1],
      u0: [leftSeamStart, leftSeamStart + leftSeam.length - 1],
    },
    seamRight: rightSeamStart,
    seamLeft: leftSeamStart,
    nu, nv,
  };
}

// Split the strip at u=splitFrac into two pieces. Returns both unfolded pieces
// and the intended glue-pair specification connecting them.
export function splitStrip(model, splitFrac, opts = {}) {
  const uS = model.uMin + (model.uMax - model.uMin) * splitFrac;
  const A = unfoldStrip(model, { ...opts, uA: model.uMin, uB: uS });
  const B = unfoldStrip(model, { ...opts, uA: uS, uB: model.uMax });
  return { A, B, uS, splitFrac };
}

// Place tabs on a single unfolded piece.
// glueSpec: array of { onSide: 'u0'|'u1'|'v0'|'v1', nTabs, priority, label, partner }
// For an open strip the natural glue is u0 <-> u1 (close the tube). Tabs go on
// the chosen seam range; overlap is resolved by shrink-then-slit.
export function planTabs(piece, glueSpec, opts = {}) {
  const outline = piece.outline;
  const seams = [];
  for (const g of glueSpec) {
    let range;
    if (g.onSide === 'u0') range = outline.edgeIndex.u0;
    else if (g.onSide === 'u1') range = outline.edgeIndex.u1;
    else if (g.onSide === 'v0') range = outline.edgeIndex.v0;
    else if (g.onSide === 'v1') range = outline.edgeIndex.v1;
    else continue;
    seams.push({
      side: g.onSide, edges: range, nTabs: g.nTabs ?? 3, priority: g.priority ?? 1,
      partner: g.partner ?? null, label: g.label || g.onSide,
    });
  }
  return layoutTabs(outline.poly, seams, opts);
}

// Full single-piece pipeline for the UI / tests.
export function analyzePreset(cfg, opts = {}) {
  const model = buildModel(cfg.model);
  const diag = diagnose(model, { nu: opts.diagNu ?? 200 });
  const dev = diag.developability;
  const piece = unfoldStrip(model, opts);
  return { model, diag, dev, piece };
}

// ---- A/B comparison: similar edge error, only one accommodates tabs ----

// Build a cone with a tunable sector width and try (A) a single wide piece and
// (B) a re-sliced two-piece layout. Report which accommodates tabs, the glued
// pairs, tab orientation, and the locked fold lines.
export function compareTabFeasibility(model, opts = {}) {
  const nT = opts.nTabs ?? 3;           // tabs per tube seam on the whole piece
  const nSplit = opts.nTabsSplit ?? 2;  // tabs per seam on each split piece

  const whole = unfoldStrip(model, { ...opts, uA: model.uMin, uB: model.uMax });
  const wholeLayout = planTabs(whole,
    [{ onSide: 'u0', nTabs: nT, priority: 1, label: 'tube-seam-a' },
     { onSide: 'u1', nTabs: nT, priority: 1, label: 'tube-seam-b' }],
    opts.layout);

  const split = splitStrip(model, opts.splitFrac ?? 0.5, opts);
  // Piece A keeps the outer u0 seam and gains the internal split seam (its u1).
  const glueA = [
    { onSide: 'u0', nTabs: nSplit, priority: 2, label: 'tube-seam' },
    { onSide: 'u1', nTabs: nSplit, priority: 2, label: 'internal-split' },
  ];
  const layoutA = planTabs(split.A, glueA, opts.layout);
  // Piece B keeps the outer u1 seam and the matching internal split seam (its u0).
  const glueB = [
    { onSide: 'u1', nTabs: nSplit, priority: 2, label: 'tube-seam' },
    { onSide: 'u0', nTabs: nSplit, priority: 2, label: 'internal-split' },
  ];
  const layoutB = planTabs(split.B, glueB, opts.layout);

  const gluedPairs = [];
  for (const a of layoutA.tabs) gluedPairs.push({ piece: 'A', edge: a.side, partnersWith: a.partner, label: a.label, orientation: a.orientation });
  for (const a of layoutB.tabs) gluedPairs.push({ piece: 'B', edge: a.side, partnersWith: a.partner, label: a.label, orientation: a.orientation });

  // Locked fold: the internal split ruling, cut in the flat pattern and re-formed
  // as a glued fold during assembly. Shared by A (u1 seam) and B (u0 seam).
  const lockedFolds = [{
    label: 'internal-split',
    pieceA: foldPolyline(split.A.outline, 'u1'),
    pieceB: foldPolyline(split.B.outline, 'u0'),
    note: 'cut in the flat pattern; re-forms as a glued fold after assembly',
  }];

  return {
    whole: { piece: whole, layout: wholeLayout, error: whole.error },
    split: { A: split.A, B: split.B, layoutA, layoutB, splitFrac: split.splitFrac },
    gluedPairs,
    lockedFolds,
    // "similar error" must be robust when both errors are ~machine zero (a clean
    // developable cone): a purely relative test would divide two round-off values
    // and call them different. Compare on an absolute floor as well.
    errorsSimilar: similarError(whole.error.rmsRel, 0.5 * (split.A.error.rmsRel + split.B.error.rmsRel), opts.similarTol ?? 1e-6),
    wholeFits: wholeLayout.tabs.length > 0 && wholeLayout.slits.length === 0,
    splitFits: layoutA.tabs.length > 0 && layoutB.tabs.length > 0 && layoutA.slits.length === 0 && layoutB.slits.length === 0,
  };
}

// Same-error test robust at machine zero: within tol both absolutely and (for
// non-negligible errors) relatively.
function similarError(a, b, tol) {
  const absTol = tol * Math.max(1, a, b);
  if (Math.abs(a - b) <= absTol) return true;
  const denom = Math.max(a, b, 1e-30);
  return Math.abs(a - b) / denom < tol;
}

function foldPolyline(outline, side) {
  const [lo, hi] = outline.edgeIndex[side];
  const pts = [];
  for (let e = lo; e <= hi; e++) pts.push(outline.poly[e]);
  pts.push(outline.poly[(hi + 1) % outline.poly.length]);
  return pts;
}

function isSeamVertex(outline, k) {
  const m = outline.meta[k];
  return m && m.kind === 'seam';
}

// ---- Segmentation vs slits tradeoff for near-parallel / ill-conditioned case ----
// We unfold at several segment counts and record the error; also try adding slits.
export function tradeoffSegments(model, opts = {}) {
  const results = [];
  for (const segs of (opts.segments ?? [1, 2, 3, 4, 6])) {
    const pieces = [];
    let maxRel = 0, rmsRel2 = 0, n = 0, totalArea3 = 0, totalArea2 = 0;
    for (let s = 0; s < segs; s++) {
      const uA = s / segs, uB = (s + 1) / segs;
      const p = unfoldStrip(model, { ...opts, uA, uB, nu: Math.max(8, Math.ceil((opts.nu ?? 64) / segs)) });
      pieces.push(p);
      maxRel = Math.max(maxRel, p.error.maxRel);
      rmsRel2 += p.error.rmsRel * p.error.rmsRel; n++;
      totalArea3 += p.error.areaTotal3; totalArea2 += p.error.areaTotal2;
    }
    results.push({
      segments: segs,
      pieces: pieces.length,
      maxRel,
      rmsRel: Math.sqrt(rmsRel2 / Math.max(1, n)),
      areaRel: Math.abs(totalArea2 - totalArea3) / Math.max(1e-12, totalArea3),
    });
  }
  return results;
}

// Serialise an unfolded piece to OBJ + a JSON of parameters for export.
export function exportOBJ(model, piece, opts = {}) {
  const { mesh, unfold } = piece;
  const { nu, nv } = mesh;
  const lines = ['# developable-unfold-prototype', '# 3D strip and 2D flattening'];
  lines.push('o strip_3d');
  for (let i = 0; i <= nu; i++) for (let j = 0; j <= nv; j++) {
    const p = mesh.rows[i][j];
    lines.push(`v ${p[0].toFixed(6)} ${p[1].toFixed(6)} ${p[2].toFixed(6)}`);
  }
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    const a = i * (nv + 1) + j + 1, b = (i + 1) * (nv + 1) + j + 1;
    const c = (i + 1) * (nv + 1) + j + 2, d = i * (nv + 1) + j + 2;
    lines.push(`f ${a} ${b} ${c}`);
    lines.push(`f ${a} ${c} ${d}`);
  }
  const bounds = bounds2d(unfold.pos);
  lines.push('o flat_2d');
  for (let i = 0; i <= nu; i++) for (let j = 0; j <= nv; j++) {
    const k = i * (nv + 1) + j;
    lines.push(`v ${unfold.pos[2 * k].toFixed(6)} ${unfold.pos[2 * k + 1].toFixed(6)} 0`);
  }
  const base = (nu + 1) * (nv + 1);
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    const a = base + i * (nv + 1) + j + 1, b = base + (i + 1) * (nv + 1) + j + 1;
    const c = base + (i + 1) * (nv + 1) + j + 2, d = base + i * (nv + 1) + j + 2;
    lines.push(`f ${a} ${b} ${c}`);
    lines.push(`f ${a} ${c} ${d}`);
  }
  return lines.join('\n');
}

export function exportParamsJSON(model) {
  return JSON.stringify({ kind: model.kind, label: model.label, cfg: model.cfg }, null, 2);
}
