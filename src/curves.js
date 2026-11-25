// Parametric boundary curves. Analytic kinds (line, arc) keep the developability
// tests exact to machine precision; bezier/polyline cover free-form input.

import { add, sub, mul, unit, cross, dot, norm, lerp3 } from './math.js';

function arcSecond(spec, t) {
  const { radius, axis, ref, a0, a1 } = spec;
  const ang = a0 + (a1 - a0) * t;
  const nrm = unit(axis);
  const r = unit(sub(ref, mul(nrm, dot(ref, nrm))));
  const b = cross(nrm, r);
  const c = Math.cos(ang), s = Math.sin(ang);
  const d = (a1 - a0) * radius;
  const dd = d * d;
  return add(mul(r, -dd * c), mul(b, -dd * s));
}

function arcEval(spec, t) {
  const { center, radius, axis, ref, a0, a1 } = spec;
  const ang = a0 + (a1 - a0) * t;
  const nrm = unit(axis);
  const r = unit(sub(ref, mul(nrm, dot(ref, nrm))));
  const b = cross(nrm, r);
  const c = Math.cos(ang), s = Math.sin(ang);
  return add(center, add(mul(r, radius * c), mul(b, radius * s)));
}
function arcDeriv(spec, t) {
  const { radius, axis, ref, a0, a1 } = spec;
  const ang = a0 + (a1 - a0) * t;
  const nrm = unit(axis);
  const r = unit(sub(ref, mul(nrm, dot(ref, nrm))));
  const b = cross(nrm, r);
  const c = Math.cos(ang), s = Math.sin(ang);
  const d = (a1 - a0) * radius;
  return add(mul(r, -d * s), mul(b, d * c));
}

function bezEval(pts, t) {
  let cur = pts.map((p) => p.slice());
  const n = cur.length;
  for (let r = 1; r < n; r++) {
    for (let i = 0; i < n - r; i++) {
      cur[i] = lerp3(cur[i], cur[i + 1], t);
    }
  }
  return cur[0];
}
function bezDeriv(pts, t) {
  const n = pts.length - 1;
  if (n <= 0) return [0, 0, 0];
  const dpts = [];
  for (let i = 0; i < n; i++) dpts.push(mul(sub(pts[i + 1], pts[i]), n));
  return bezEval(dpts, t);
}
function bezSecond(pts, t) {
  const n = pts.length - 1;
  if (n <= 1) return [0, 0, 0];
  const dd = [];
  for (let i = 0; i < n - 1; i++) {
    dd.push(mul(add(sub(pts[i + 2], mul(pts[i + 1], 2)), pts[i]), n * (n - 1)));
  }
  return bezEval(dd, t);
}

function polyEval(pts, t, closed) {
  const segs = closed ? pts.length : pts.length - 1;
  if (segs <= 0) return pts[0].slice();
  let x = t * segs;
  let i = Math.floor(x);
  if (i >= segs) i = segs - 1;
  if (i < 0) i = 0;
  const f = x - i;
  const a = pts[i];
  const b = pts[closed ? (i + 1) % pts.length : i + 1];
  return lerp3(a, b, f);
}
function polyDeriv(pts, t, closed) {
  const segs = closed ? pts.length : pts.length - 1;
  if (segs <= 0) return [0, 0, 0];
  let x = t * segs;
  let i = Math.floor(x);
  if (i >= segs) i = segs - 1;
  if (i < 0) i = 0;
  const a = pts[i];
  const b = pts[closed ? (i + 1) % pts.length : i + 1];
  return mul(sub(b, a), segs);
}

export function createCurve(spec) {
  if (!spec || !spec.kind) throw new Error('curve spec needs a kind');
  const s = spec;
  switch (s.kind) {
    case 'line':
      return {
        kind: 'line',
        spec: s,
        eval: (t) => lerp3(s.a, s.b, t),
        deriv: () => sub(s.b, s.a),
        second: () => [0, 0, 0],
      };
    case 'arc':
      return {
        kind: 'arc',
        spec: s,
        eval: (t) => arcEval(s, t),
        deriv: (t) => arcDeriv(s, t),
        second: (t) => arcSecond(s, t),
      };
    case 'bezier':
      return {
        kind: 'bezier',
        spec: s,
        eval: (t) => bezEval(s.points, t),
        deriv: (t) => bezDeriv(s.points, t),
        second: (t) => bezSecond(s.points, t),
      };
    case 'polyline':
      return {
        kind: 'polyline',
        spec: s,
        eval: (t) => polyEval(s.points, t, !!s.closed),
        deriv: (t) => polyDeriv(s.points, t, !!s.closed),
        second: () => [0, 0, 0],
      };
    default:
      throw new Error('unknown curve kind: ' + s.kind);
  }
}

export function translateCurve(curve, offset) {
  const s = curve.spec;
  if (s.kind === 'line') return createCurve({ kind: 'line', a: add(s.a, offset), b: add(s.b, offset) });
  if (s.kind === 'arc') return createCurve({ ...s, center: add(s.center, offset) });
  if (s.kind === 'bezier') return createCurve({ ...s, points: s.points.map((p) => add(p, offset)) });
  if (s.kind === 'polyline') return createCurve({ ...s, points: s.points.map((p) => add(p, offset)) });
  throw new Error('cannot translate ' + s.kind);
}

export function tangentAt(curve, t) {
  return unit(curve.deriv(t));
}

export function arcLength(curve, t0, t1, steps = 128) {
  let len = 0;
  let prev = curve.eval(t0);
  for (let i = 1; i <= steps; i++) {
    const t = t0 + (t1 - t0) * i / steps;
    const p = curve.eval(t);
    len += norm(sub(p, prev));
    prev = p;
  }
  return len;
}

// Preset configurations. Each declares the intended ground truth, which the test
// suite verifies against the *measured* developability result (the app never
// trusts the label).
export const PRESETS = {
  cylinder: () => ({
    label: 'Cylinder band',
    groundTruth: 'developable',
    note: 'Parallel rulings. Exact: zero angle defect and zero det residual.',
    model: {
      kind: 'cylinder',
      curve0: {
        kind: 'arc', center: [0, 0, 0], radius: 1, axis: [0, 0, 1], ref: [1, 0, 0],
        a0: -Math.PI * 0.62, a1: Math.PI * 0.62,
      },
      offset: [0, 0, 1.4],
      nu: 64, nv: 6,
    },
  }),

  cone: () => ({
    label: 'Cone frustum',
    groundTruth: 'developable',
    note: 'Concurrent rulings. Exact developable; unfolding is a circular sector.',
    model: {
      kind: 'cone',
      curve0: {
        kind: 'arc', center: [0, 0, 0], radius: 1, axis: [0, 0, 1], ref: [1, 0, 0],
        a0: -Math.PI * 0.58, a1: Math.PI * 0.58,
      },
      apex: [0, 0, 2.0],
      scale: 0.40,
      nu: 64, nv: 6,
    },
  }),

  tangent: () => ({
    label: 'Tangent developable (curved spine)',
    groundTruth: 'developable',
    note: 'Rulings are spine tangents. Exact developable away from the regression edge.',
    model: {
      kind: 'tangent',
      spine: {
        kind: 'bezier',
        points: [
          [-1.35, -0.35, 0.05],
          [-0.55, 0.65, 0.55],
          [0.35, -0.45, 1.0],
          [1.35, 0.55, 1.45],
        ],
      },
      v0: 0.12, v1: 0.95,
      nu: 72, nv: 6,
    },
  }),

  skewLines: () => ({
    label: 'Two skew lines (hyperbolic paraboloid)',
    groundTruth: 'not-developable',
    note: 'Classic ruled-but-not-developable: Gaussian curvature nonzero everywhere.',
    model: {
      kind: 'direct',
      curve0: { kind: 'line', a: [-1.2, 0, 0], b: [1.2, 0, 0] },
      curve1: { kind: 'line', a: [-0.35, 1.25, 1.15], b: [0.35, -1.25, 1.15] },
      nu: 48, nv: 6,
    },
  }),

  twistedRings: () => ({
    label: 'Two tilted circular arcs',
    groundTruth: 'not-developable',
    note: 'Ruled between two circular boundaries whose planes differ: not developable.',
    model: {
      kind: 'direct',
      curve0: {
        kind: 'arc', center: [0, 0, 0], radius: 1, axis: [0, 0, 1], ref: [1, 0, 0],
        a0: -Math.PI * 0.5, a1: Math.PI * 0.5,
      },
      curve1: {
        kind: 'arc', center: [0, 0, 1.35], radius: 0.72, axis: [0.42, 0.12, 1], ref: [0.8, 0.6, 0],
        a0: Math.PI * 0.12, a1: Math.PI * 1.05,
      },
      nu: 48, nv: 6,
    },
  }),

  nearParallel: () => {
    // Two nearly parallel skew lines. Rulings are near-parallel (angle varies by
    // ~0.4 deg), so the developability function F(u,v)=det(C0',R,C1') is
    // essentially independent of v (dF/dv ~ 0): the condition carries almost no
    // information about which v pairs with a given u. Recovering the developable
    // correspondence is ill-conditioned / underdetermined. The surface is very
    // slightly non-developable, and *segmenting increases* the unfolded edge
    // error, so the correct choice is to add a slit rather than subdivide.
    return {
      label: 'Near-parallel rulings (ill-conditioned correspondence)',
      groundTruth: 'near-developable-ill-conditioned',
      note: 'Two nearly parallel skew lines. dF/dv~0 => pairing underdetermined (ill-conditioned). Slightly non-developable; segmenting raises unfolded error, so choose a slit over segmentation. Decide by measured edge error.',
      model: {
        kind: 'direct',
        curve0: { kind: 'line', a: [-1.4, 0, 0], b: [1.4, 0, 0] },
        curve1: { kind: 'line', a: [-1.4, 1.5, 0.0], b: [1.4, 1.5, 0.02] },
        nu: 96, nv: 8,
      },
    };
  },

  regression: () => ({
    label: 'Tangent developable across its regression edge',
    groundTruth: 'singular',
    note: 'v0 < 0 < v1: the strip spans the spine (edge of regression/cuspidal edge) where normals flip. Must be located and excluded, not output.',
    model: {
      kind: 'tangent',
      spine: {
        kind: 'bezier',
        points: [
          [-1.05, 0.0, 0.15],
          [-0.25, 0.95, 0.55],
          [0.65, -0.15, 0.95],
          [1.45, 1.05, 1.2],
        ],
      },
      v0: -0.45, v1: 0.7,
      nu: 72, nv: 6,
    },
  }),

  wideCone: () => ({
    label: 'Wide cone frustum (A/B re-slice demo)',
    groundTruth: 'developable',
    note: 'Nearly full-circle developable cone. Both unfoldings measure nearly the same edge error, but the single wide piece cannot host its tabs (a cut line retreats to a slit); re-slicing at u=1/2 accommodates every tab. Press "Compare whole vs split".',
    model: {
      kind: 'cone',
      curve0: {
        kind: 'arc', center: [0, 0, 0], radius: 1, axis: [0, 0, 1], ref: [1, 0, 0],
        a0: -Math.PI * 0.98, a1: Math.PI * 0.98,
      },
      apex: [0, 0, 1.2],
      scale: 0.8,
      nu: 64, nv: 8,
    },
  }),
};
