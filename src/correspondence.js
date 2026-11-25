// Solve the *declared* developability condition for a correspondence between two
// boundary curves.
//
// Given boundary curves C0(u) and C1(v), a ruling connects C0(u) to C1(v). The
// ruled strip is developable iff the tangent plane is constant along each ruling:
//      det( C0'(u), C1(v) - C0(u), C1'(v) ) = 0                    (D)
// (equivalently, C1'(v) lies in the plane spanned by C0'(u) and the ruling).
//
// We solve (D) for v as a function of u by continuation from a starting pair,
// choosing the branch that keeps v monotone and arc-length-consistent. Where the
// solution is ill-conditioned (near-parallel rulings, tangential contact) we
// report the conditioning rather than emitting a shaky correspondence.
//
// This is the "solve, then unfold" path. The "direct" ruled surfaces (pairing
// C0(u) with C1(u)) are handled by ruled.js and *measured* -- they are not
// assumed developable.

import {
  sub, add, mul, dot, cross, unit, norm, dist, det3, closestPointsOnLines,
} from './math.js';
import { createCurve } from './curves.js';

// Residual of the developability condition for a given (u,v) pair.
export function developabilityResidual(c0, c1, u, v) {
  const p0 = c0.eval(u), p1 = c1.eval(v);
  const d0 = c0.deriv(u), d1 = c1.deriv(v);
  const R = sub(p1, p0);
  const lR = norm(R), l0 = norm(d0), l1 = norm(d1);
  const signed = det3(d0, R, d1);
  const scale = l0 * lR * l1;
  return { raw: Math.abs(signed), signed, normalized: scale > 1e-14 ? Math.abs(signed) / scale : 0, sinAngle: scale > 1e-14 ? Math.abs(signed) / scale : 0 };
}

// Root sensitivity: a root of F(u,.)=0 is well determined when |dF/dv| is large
// relative to F's scale. When rulings are near parallel, dF/dv -> 0 and the root
// can be moved a long way by a tiny residual => ill-conditioned inverse problem.
// kappa = normalisationScale / |dF/dv| (dimensionless, grows with ill-conditioning).
//
// We estimate dF/dv by a *one-sided* difference away from the root: a central
// difference AT the root suffers catastrophic cancellation (F(v+h) and F(v-h)
// are both ~0, so their difference is lost in floating point). Stepping to
// v+delt and v+2*delt with delt >> the residual keeps the signal.
export function rootConditioning(c0, c1, u, v) {
  const F = (vv) => developabilityResidual(c0, c1, u, vv).raw;
  const delt = 1e-3 * Math.max(1, Math.abs(v));
  const v1 = Math.min(1, v + delt);
  const v2 = Math.min(1, v + 2 * delt);
  let dFdv;
  if (v2 - v1 > 1e-9) {
    dFdv = (F(v2) - F(v1)) / (v2 - v1);
  } else {
    const v0 = Math.max(0, v - delt);
    dFdv = (F(v) - F(v0)) / Math.max(1e-12, v - v0);
  }
  dFdv = Math.abs(dFdv);
  const p0 = c0.eval(u), p1 = c1.eval(v);
  const d0 = c0.deriv(u), d1 = c1.deriv(v);
  const scale = norm(d0) * norm(sub(p1, p0)) * norm(d1);
  if (scale < 1e-14) return { dFdv, scale, kappa: 0, flat: false };
  const flat = dFdv < 1e-9 * Math.max(1e-9, scale);   // F essentially constant in v
  const kappa = dFdv > 1e-14 ? scale / dFdv : Infinity;
  return { dFdv, scale, kappa, flat };
}

// A structural measure of whether the declared condition can pin the pairing at
// all. Near-parallel rulings make F(u,v) nearly constant in v, so dF/dv ~ 0 and
// the root is not isolated: the inverse problem is ill-conditioned. We report
//   flatness = |dF/dv| / (|F| scale)
// small (-> 0) means the condition is uninformative; combined with a nonzero
// residual it means "ruled, near-developable, but no reliable pairing".
export function pairingDiagnostics(c0spec, c1spec, opts = {}) {
  const c0 = createCurve(c0spec);
  const c1 = createCurve(c1spec);
  const nu = opts.nu ?? 48;
  let maxResidual = 0, maxKappa = 0, anyFlat = false;
  for (let i = 0; i <= nu; i++) {
    const u = i / nu;
    const v = opts.pair?.(u) ?? u;
    const F0 = developabilityResidual(c0, c1, u, v).raw;
    maxResidual = Math.max(maxResidual, Math.abs(F0));
    // one-sided conditioning avoids the cancellation of a central difference at a
    // V-shaped root. kappa = scale / |dF/dv|; large => uninformative condition.
    const cond = rootConditioning(c0, c1, u, v);
    if (cond.flat) anyFlat = true;
    if (Number.isFinite(cond.kappa)) maxKappa = Math.max(maxKappa, cond.kappa);
  }
  return {
    maxResidual,
    maxCondition: anyFlat ? Infinity : maxKappa,
    illConditioned: anyFlat || maxKappa > 1e3,
    flat: anyFlat,
    nearDevelopable: maxResidual < 1e-1,
  };
}

// Find all v in [0,1] satisfying (D) for a fixed u, by scanning and bisection.
// Returns roots with a crude conditioning estimate.
export function solveVForU(c0, c1, u, opts = {}) {
  const nScan = opts.nScan ?? 400;
  const vMin = opts.vMin ?? 0, vMax = opts.vMax ?? 1;
  // Use the SIGNED residual: F is transverse at a simple root, so the sign change
  // is the reliable bracket. Taking |F| hides the crossing and turns every root
  // into a tangential minimum, which bisection cannot find.
  const f = (v) => developabilityResidual(c0, c1, u, v).signed;
  const roots = [];
  let vPrev = vMin;
  let fPrev = f(vPrev);
  const samples = [{ v: vPrev, f: fPrev }];
  for (let i = 1; i <= nScan; i++) {
    const v = vMin + (vMax - vMin) * i / nScan;
    const fv = f(v);
    samples.push({ v, f: fv });
    if (fPrev === 0) {
      roots.push(vPrev);
    } else if (fPrev * fv < 0) {
      const root = bisect(f, vPrev, v, 80);
      roots.push(polishRoot(f, root, vPrev, v));
    }
    vPrev = v; fPrev = fv;
  }
  // dedupe close roots
  const uniq = [];
  for (const r of roots.sort((a, b) => a - b)) {
    if (!uniq.length || Math.abs(r - uniq[uniq.length - 1]) > 1e-6) uniq.push(r);
  }
  return { roots: uniq, samples };
}

function bisect(f, a, b, iters) {
  let fa = f(a), fb = f(b);
  if (fa === 0) return a;
  if (fb === 0) return b;
  for (let k = 0; k < iters; k++) {
    const m = 0.5 * (a + b);
    const fm = f(m);
    if (fa * fm <= 0) { b = m; fb = fm; } else { a = m; fa = fm; }
  }
  return 0.5 * (a + b);
}

// Newton polish of a bracketed root, using a secant slope. Bisection alone stops
// at the scan grid; polynomial residual minima are parabolic so Newton converges
// to machine precision in a few steps.
function polishRoot(f, v, lo, hi) {
  let x = v;
  for (let k = 0; k < 30; k++) {
    const fx = f(x);
    if (Math.abs(fx) < 1e-15) break;
    const h = Math.max(1e-9, 1e-6 * Math.max(1, Math.abs(x)));
    const xp = Math.min(hi, x + h), xm = Math.max(lo, x - h);
    const slope = (f(xp) - f(xm)) / (xp - xm);
    if (!Number.isFinite(slope) || Math.abs(slope) < 1e-18) break;
    let xn = x - fx / slope;
    if (xn < lo || xn > hi) break;
    if (Math.abs(xn - x) < 1e-14) { x = xn; break; }
    x = xn;
  }
  return x;
}

// Golden-section minimisation of |F| on [lo,hi] for tangential (double) roots.
function polishMinimum(f, lo, hi) {
  const g = 0.6180339887498949;
  let a = lo, b = hi;
  let c = b - g * (b - a), d = a + g * (b - a);
  const fc = Math.abs(f(c)), fd = Math.abs(f(d));
  for (let k = 0; k < 100; k++) {
    if (fc < fd) { b = d; d = c; c = b - g * (b - a); }
    else { a = c; c = d; d = a + g * (b - a); }
    if (b - a < 1e-15 * Math.max(1, Math.abs(b))) break;
  }
  return 0.5 * (a + b);
}

// Continuation: walk u from 0 to 1, tracking v(u) along a chosen branch.
// Branch selection: prefer the root closest to the previous v (continuity),
// with a monotonicity preference for papercraft-friendly rulings.
export function solveCorrespondence(c0spec, c1spec, opts = {}) {
  const c0 = createCurve(c0spec);
  const c1 = createCurve(c1spec);
  const nu = opts.nu ?? 96;
  const startV = opts.startV ?? 0;
  const vMin = opts.vMin ?? 0, vMax = opts.vMax ?? 1;

  const us = [];
  const vs = [];
  const condition = []; // multiply-dependent conditioning: 1/sin(angle between R and d1) etc.
  const residuals = [];

  let vPrev = startV;
  for (let i = 0; i <= nu; i++) {
    const u = i / nu;
    const { roots, samples } = solveVForU(c0, c1, u, { nScan: 400, vMin, vMax });
    let v;
    let found = false;
    if (roots.length) {
      // closest-to-previous branch
      let best = roots[0];
      let bd = Math.abs(best - vPrev);
      for (const r of roots) {
        const d = Math.abs(r - vPrev);
        if (d < bd) { bd = d; best = r; }
      }
      v = best;
      found = true;
    } else {
      // No sign change: |F| may touch zero tangentially (a double root, as happens
      // when F is a perfect square in v). Refine the minimum of |F| seeded from the
      // previous v (the branch is smooth), then golden-section around it.
      const fAbs = (vv) => Math.abs(developabilityResidual(c0, c1, u, vv).raw);
      let mi = 0, mf = Infinity;
      for (let k = 0; k < samples.length; k++) {
        if (samples[k].f < mf) { mf = samples[k].f; mi = k; }
      }
      // prefer the minimum near vPrev if it is comparably small (continuation)
      let seed = samples[mi].v;
      if (fAbs(vPrev) < 10 * mf + 1e-9) seed = vPrev;
      const dV = 1.5 / samples.length;   // ~1.5 scan steps
      const lo = Math.max(vMin, seed - dV), hi = Math.min(vMax, seed + dV);
      v = polishMinimum(fAbs, lo, hi);
    }
    const res = developabilityResidual(c0, c1, u, v);
    const cond = rootConditioning(c0, c1, u, v);
    us.push(u); vs.push(v); residuals.push(res.normalized); condition.push(cond.kappa);
    vPrev = v;
  }

  // arc-length consistency: monotone v => no fold in the correspondence
  let monotone = true;
  for (let i = 1; i < vs.length; i++) if (vs[i] < vs[i - 1] - 1e-6) monotone = false;
  const maxResidual = Math.max(...residuals);
  const finiteConds = condition.filter((k) => Number.isFinite(k));
  const maxCondition = finiteConds.length ? Math.max(...finiteConds) : 0;

  return {
    c0, c1, us, vs, residuals, condition,
    maxResidual, maxCondition,
    monotone,
    feasible: maxResidual < (opts.residualTol ?? 1e-2) && monotone,
    // a feasibility classification for the UI / tests
    status: !monotone ? 'branch-break'
      : maxResidual >= (opts.residualTol ?? 1e-2) ? 'no-developable-correspondence'
      : maxCondition > (opts.conditionTol ?? 1e3) ? 'ill-conditioned'
      : 'ok',
  };
}
