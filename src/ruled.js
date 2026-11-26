// Ruled surface model with analytic frames and *measured* developability.
//
// X(u,v) = P0(u) + v * (P1(u) - P0(u)),  u in [0,1], v in [0,1].
// Ruling R(u) = P1(u) - P0(u).
//
// Developability is NOT implied by "ruled". A ruled surface is developable iff
// the tangent plane is constant along each ruling, equivalently
//      det(P0'(u), R(u), R'(u)) = 0.
// We measure the tangent-plane relation directly: the normals at several v on
// the same ruling must be parallel. The residual is the maximum angle between
// them. This is the geometric quantity the condition encodes, checked rather
// than assumed.

import {
  add, sub, mul, dot, cross, unit, norm, dist, det3, angleBetween,
} from './math.js';
import { createCurve } from './curves.js';

const TINY = 1e-12;

export function buildModel(cfg) {
  const periodicU = !!cfg.periodicU;
  const uMin = cfg.uMin ?? 0;
  const uMax = cfg.uMax ?? 1;

  let frame;
  let label = cfg.label || 'ruled surface';
  let kind = cfg.kind;

  if (kind === 'cylinder') {
    const c0 = createCurve(cfg.curve0);
    const off = cfg.offset;
    frame = (u) => {
      const p0 = c0.eval(u), d0 = c0.deriv(u);
      return { p0, p1: add(p0, off), d0, d1: d0 };
    };
  } else if (kind === 'cone') {
    const c0 = createCurve(cfg.curve0);
    const apex = cfg.apex, scale = cfg.scale;
    frame = (u) => {
      const p0 = c0.eval(u), d0 = c0.deriv(u);
      return { p0, p1: add(apex, mul(sub(p0, apex), scale)), d0, d1: mul(d0, scale) };
    };
  } else if (kind === 'tangent') {
    const spine = createCurve(cfg.spine);
    const v0 = cfg.v0 ?? 0.15;
    const v1 = cfg.v1 ?? 1.0;
    frame = (u) => {
      const g = spine.eval(u);
      const gd = spine.deriv(u);
      const gdd = spine.second ? spine.second(u) : [0, 0, 0];
      const s = norm(gd);
      const T = s > TINY ? mul(gd, 1 / s) : [0, 0, 0];
      // T' = (g'' s^2 - g' (g'·g'')) / s^3
      const gdgd = dot(gd, gdd);
      const Tp = s > TINY
        ? mul(sub(mul(gdd, s * s), mul(gd, gdgd)), 1 / (s * s * s))
        : [0, 0, 0];
      return {
        p0: add(g, mul(T, v0)),
        p1: add(g, mul(T, v1)),
        d0: add(gd, mul(Tp, v0)),
        d1: add(gd, mul(Tp, v1)),
        gamma: g, tangent: T, tangentP: Tp, speed: s,
      };
    };
  } else if (kind === 'direct') {
    const c0 = createCurve(cfg.curve0);
    const c1 = createCurve(cfg.curve1);
    frame = (u) => ({ p0: c0.eval(u), p1: c1.eval(u), d0: c0.deriv(u), d1: c1.deriv(u) });
  } else {
    throw new Error('unknown ruled kind: ' + kind);
  }

  const model = {
    kind, label, periodicU, uMin, uMax, frame, cfg,
    uSpan: uMax - uMin,
    point(u, v) {
      const f = frame(u);
      return add(f.p0, mul(sub(f.p1, f.p0), v));
    },
    ruling(u) {
      const f = frame(u);
      return { p0: f.p0, p1: f.p1, dir: sub(f.p1, f.p0), len: dist(f.p0, f.p1) };
    },
    dU(u, v) {
      const f = frame(u);
      return add(f.d0, mul(sub(f.d1, f.d0), v));
    },
    dV(u) {
      const f = frame(u);
      return sub(f.p1, f.p0);
    },
    normal(u, v) {
      return cross(model.dU(u, v), model.dV(u));
    },
  };
  return model;
}

// Per-ruling measured developability. Returns sampled loss and classification.
export function measureDevelopability(model, opts = {}) {
  const nu = Math.max(8, opts.nu || 160);
  const vSamples = opts.vSamples || [0, 0.25, 0.5, 0.75, 1];
  const angleTol = opts.angleTol ?? 1e-3;      // radians
  const detTol = opts.detTol ?? 1e-3;          // dimensionless sine
  const perRuling = [];
  let maxAngle = 0, maxDet = 0, maxDetU = 0;

  for (let i = 0; i <= nu; i++) {
    const u = model.uMin + (model.uSpan * i) / nu;
    const f = model.frame(u);
    const R = sub(f.p1, f.p0);
    const Rp = sub(f.d1, f.d0);
    const L = norm(R);
    const Lp = norm(Rp);
    const Ld = norm(f.d0);
    const scale = Math.max(L, Ld, 1e-9);

    // normals along the ruling
    const normals = vSamples.map((v) => cross(add(f.d0, mul(sub(f.d1, f.d0), v)), R));
    const n0 = normals[0];
    let ang = 0;
    let degenerate = false;
    for (let k = 0; k < normals.length; k++) {
      if (norm(normals[k]) < 1e-9 * (scale * scale)) degenerate = true;
      const a = angleBetween(n0, normals[k]);
      if (a > ang) ang = a;
    }
    // declared condition residual: |det(P0', R, R')| / (|P0'| |R| |R'|)
    // when R'~0 (parallel rulings) the condition holds identically by continuity.
    // When L~0 (collapsed ruling) the ratio is 0/0: the condition carries no
    // information there, so report 0 and rely on the degenerate flag rather than
    // emitting a non-finite residual. A non-finite residual must never leak out.
    let detRes = 0;
    if (L > 1e-12 * scale && Lp > 1e-9 * scale && Ld > 1e-12) {
      detRes = Math.abs(det3(f.d0, R, Rp)) / (Ld * L * Lp);
    }
    if (L < 1e-12 * scale) degenerate = true;
    perRuling.push({ u, angle: ang, detRes, degenerate, len: L, dir: L > TINY ? mul(R, 1 / L) : [0, 0, 0] });
    if (!degenerate) {
      if (ang > maxAngle) { maxAngle = ang; maxDetU = u; }
      if (detRes > maxDet) maxDet = detRes;
    }
  }

  const bad = perRuling.filter((r) => !r.degenerate && (r.angle > angleTol || r.detRes > detTol));
  const developable = maxAngle <= angleTol && maxDet <= detTol;
  return {
    perRuling, maxAngle, maxDet, maxAtU: maxDetU,
    angleTol, detTol, developable,
    badRulings: bad.map((r) => r.u),
    classification: developable ? 'developable' : 'not-developable',
  };
}

// Fine scan of one ruling: returns the singular signature, or null if clean.
// A sign flip of the normal across an interior v means the strip crosses the
// edge of regression; a near-vanishing |normal| means the tangent plane is
// undefined. Both invalidate unfolding across that ruling.
function singularAtU(model, u, scale) {
  const f = model.frame(u);
  const R = sub(f.p1, f.p0);
  if (norm(R) < 1e-7 * scale) return { type: 'collapsed-ruling', minV: 0.5 };
  let minMag = Infinity, minV = 0;
  const normalsV = [];
  for (let k = 0; k <= 40; k++) {
    const v = k / 40;
    const n = model.normal(u, v);
    const mag = norm(n);
    normalsV.push({ v, n, mag });
    if (mag < minMag) { minMag = mag; minV = v; }
  }
  let flipAt = null;
  for (let k = 1; k < normalsV.length; k++) {
    const a = normalsV[k - 1], b = normalsV[k];
    if (a.mag > 1e-9 && b.mag > 1e-9 && dot(a.n, b.n) < 0) { flipAt = 0.5 * (a.v + b.v); break; }
  }
  if (flipAt !== null) return { type: 'normal-flip', minV, flipAt };
  if (minMag < 1e-5 * scale * scale) return { type: 'degenerate-normal', minV };
  if (model.kind === 'tangent' && f.speed < 1e-6 * scale) return { type: 'regression-edge', minV: 0.5 };
  return null;
}

// Locate singular regions and self-intersections. Never continue silently.
export function diagnose(model, opts = {}) {
  const nu = Math.max(16, opts.nu || 200);
  const dev = measureDevelopability(model, { ...opts, nu });
  const scale = Math.max(1, ...modelScale(model, nu));
  const singular = [];

  for (let i = 0; i <= nu; i++) {
    const u = model.uMin + (model.uSpan * i) / nu;
    const sig = singularAtU(model, u, scale);
    if (!sig) continue;
    if (sig.type === 'collapsed-ruling') {
      singular.push({ type: 'collapsed-ruling', u, v: null, detail: 'ruling length ~ 0' });
      continue;
    }
    // Degenerate-normal detection. A sign flip of the normal across an interior v
    // (the tangent plane turns over -> the strip crosses the edge of regression)
    // is the robust signature; the exact v is polished by golden-section.
    const vr = refineDegenerateV(model, u, sig.minV);
    singular.push({
      type: sig.type,
      u, v: vr.v,
      detail: sig.type === 'normal-flip'
        ? 'normal turns over across ruling (edge of regression)'
        : sig.type === 'regression-edge' ? 'spine speed ~ 0'
        : 'tangent plane nearly undefined',
      vActual: vr.vActual,
    });
  }

  // ruling self-intersection: nearby rulings crossing inside the strip
  const crossings = [];
  for (let i = 0; i <= nu; i += 1) {
    const ui = model.uMin + (model.uSpan * i) / nu;
    const fi = model.frame(ui);
    for (let j = i + 2; j <= Math.min(nu, i + 12); j++) {
      const uj = model.uMin + (model.uSpan * j) / nu;
      const fj = model.frame(uj);
      const hit = segmentSegments(fi.p0, fi.p1, fj.p0, fj.p1, 1e-6 * scale);
      if (hit) crossings.push({ type: 'ruling-crossing', u: ui, u2: uj, v: hit.s, v2: hit.t, detail: 'rulings intersect inside strip' });
    }
  }

  const merged = dedupe(singular.concat(crossings));
  const validURanges = developableRegions(model, nu, scale);

  // Regression-edge u-locus: where a normal flip / degeneracy occurs at interior v.
  const regression = merged.filter((s) => s.type === 'normal-flip' || s.type === 'degenerate-normal');
  const regressionU = regression.map((s) => s.u);
  const regressionVActual = regression.length ? regression[0].vActual : null;

  return {
    developability: dev,
    singular: merged,
    hasSingularity: merged.length > 0,
    regressionEdge: regression.length
      ? { uMin: Math.min(...regressionU), uMax: Math.max(...regressionU), vActual: regressionVActual, samples: regression.length }
      : null,
    validURanges,
    ok: !dev.developable ? 'not-developable' : merged.length ? 'singular' : 'ok',
  };
}

// Golden-section refinement of the v that minimises |normal| on ruling u.
function refineDegenerateV(model, u, vGuess) {
  let a = Math.max(0, vGuess - 0.05), b = Math.min(1, vGuess + 0.05);
  const g = (0.6180339887);
  let c = b - g * (b - a), d = a + g * (b - a);
  const fn = (v) => norm(model.normal(u, v));
  for (let k = 0; k < 60; k++) {
    if (fn(c) < fn(d)) { b = d; d = c; c = b - g * (b - a); }
    else { a = c; c = d; d = a + g * (b - a); }
  }
  const v = 0.5 * (a + b);
  // actual ruling parameter (for tangent: v_actual = v0 + (v1-v0)*v)
  let vActual = v;
  if (model.kind === 'tangent') vActual = (model.cfg.v0 ?? 0) + ((model.cfg.v1 ?? 1) - (model.cfg.v0 ?? 0)) * v;
  return { v, vActual };
}

function modelScale(model, nu) {
  let m = 0;
  for (let i = 0; i <= nu; i++) {
    const u = model.uMin + model.uSpan * i / nu;
    const f = model.frame(u);
    m = Math.max(m, norm(f.p0), norm(f.p1));
  }
  return [m];
}

function dedupe(list) {
  const seen = new Set();
  const out = [];
  for (const e of list) {
    const k = e.type + ':' + e.u.toFixed(4) + ':' + (e.v == null ? '-' : e.v.toFixed(3));
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(e);
  }
  return out;
}

// continuous u-ranges free of singular rulings (same detector as diagnose, so
// the clip window and the located singularities agree)
function developableRegions(model, nu, scale) {
  const flags = [];
  for (let i = 0; i <= nu; i++) {
    const u = model.uMin + model.uSpan * i / nu;
    flags.push(singularAtU(model, u, scale) ? 0 : 1);
  }
  const ranges = [];
  let start = null;
  for (let i = 0; i <= nu; i++) {
    if (flags[i] && start === null) start = i;
    if ((!flags[i] || i === nu) && start !== null) {
      const end = flags[i] ? i : i - 1;
      ranges.push([model.uMin + model.uSpan * start / nu, model.uMin + model.uSpan * end / nu]);
      start = null;
    }
  }
  return ranges;
}

function segmentSegments(a0, a1, b0, b1, tol) {
  const u = sub(a1, a0), v = sub(b1, b0);
  const w0 = sub(a0, b0);
  const uu = dot(u, u), uv = dot(u, v), vv = dot(v, v);
  const uw = dot(u, w0), vw = dot(v, w0);
  const den = uu * vv - uv * uv;
  if (Math.abs(den) < 1e-14 * Math.max(1, uu * vv)) return null;
  const s = (uv * vw - vv * uw) / den;
  const t = (uu * vw - uv * uw) / den;
  if (s < -1e-6 || s > 1 + 1e-6 || t < -1e-6 || t > 1 + 1e-6) return null;
  const p = add(a0, mul(u, s));
  const q = add(b0, mul(v, t));
  if (dist(p, q) < tol) return { s, t, p };
  return null;
}

// Sample the surface into a grid for display and fabrication.
// For periodicU the last u-column is a duplicate of the first (the seam).
export function sampleMesh(model, nu, nv) {
  const rows = [];
  for (let i = 0; i <= nu; i++) {
    const u = model.uMin + model.uSpan * i / nu;
    const f = model.frame(u);
    const row = [];
    for (let j = 0; j <= nv; j++) {
      const v = j / nv;
      row.push(add(f.p0, mul(sub(f.p1, f.p0), v)));
    }
    rows.push(row);
  }
  return { rows, nu, nv, periodicU: model.periodicU };
}
