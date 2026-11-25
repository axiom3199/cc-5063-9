// Small dense linear algebra and vector helpers. No dependencies.

export const EPS = 1e-12;

export function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
export function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
export function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
export function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
export function mul(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
export function norm(a) { return Math.hypot(a[0], a[1], a[2]); }
export function dist(a, b) { return norm(sub(a, b)); }
export function unit(a) {
  const n = norm(a);
  return n < 1e-15 ? [0, 0, 0] : [a[0] / n, a[1] / n, a[2] / n];
}
export function lerp3(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

// Intersection of two 3D lines a + s*u and b + t*v. Returns {p,q,s,t,dist}.
// Lines are not assumed to intersect (ruled surfaces are generally skew).
export function closestPointsOnLines(a, u, b, v) {
  const w0 = sub(a, b);
  const uu = dot(u, u), uv = dot(u, v), vv = dot(v, v);
  const uw = dot(u, w0), vw = dot(v, w0);
  const den = uu * vv - uv * uv;
  if (Math.abs(den) < 1e-14 * Math.max(1, uu * vv)) {
    // parallel: project b onto first line
    const s = uw / uu;
    const p = add(a, mul(u, s));
    const q = p;
    return { p, q, s, t: 0, dist: dist(p, b), parallel: true };
  }
  const s = (uv * vw - vv * uw) / den;
  const t = (uu * vw - uv * uw) / den;
  const p = add(a, mul(u, s));
  const q = add(b, mul(v, t));
  return { p, q, s, t, dist: dist(p, q), parallel: false };
}

// Eigen-decomposition of a symmetric 2x2 [[a,b],[b,c]].
export function eigSym2(a, b, c) {
  const tr = a + c;
  const det = a * c - b * b;
  const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1 = tr / 2 + disc;
  const l2 = tr / 2 - disc;
  // eigenvector for l1
  let v1;
  if (Math.abs(b) < 1e-14) {
    v1 = Math.abs(a - l1) < Math.abs(c - l1) ? [1, 0] : [0, 1];
  } else {
    v1 = [b, l1 - a];
  }
  const n1 = Math.hypot(v1[0], v1[1]) || 1;
  v1 = [v1[0] / n1, v1[1] / n1];
  const v2 = [-v1[1], v1[0]];
  return { l1, l2, v1, v2 };
}

// Solve dense linear system A x = b with partial pivoting. A is n x n (array of rows), b length n.
export function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((row, i) => Float64Array.from([...row, b[i]]));
  for (let col = 0; col < n; col++) {
    let piv = col;
    let best = Math.abs(M[col][col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(M[r][col]);
      if (v > best) { best = v; piv = r; }
    }
    if (best < 1e-15) return null;
    if (piv !== col) { const tmp = M[piv]; M[piv] = M[col]; M[col] = tmp; }
    const pv = M[col][col];
    for (let r = col + 1; r < n; r++) {
      const f = M[r][col] / pv;
      if (f === 0) continue;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let c = r + 1; c < n; c++) s -= M[r][c] * x[c];
    x[r] = s / M[r][r];
  }
  return Array.from(x);
}

// Solve symmetric positive-semidefinite system via normal equations is fragile;
// use this robust symmetric banded solve for small n (dense Cholesky w/ fallback to LU).
export function solveSymmetricSPD(A, b) {
  const n = b.length;
  const L = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 1e-15) return solveLinear(A, b);
        L[i][j] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
    y[i] = s / L[i][i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return Array.from(x);
}

export function det3(a, b, c) {
  return dot(a, cross(b, c));
}

export function angleBetween(a, b) {
  const n = norm(a) * norm(b);
  if (n < 1e-15) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot(a, b) / n)));
}

// signed angle from a to b about unit normal n
export function signedAngle(a, b, n) {
  const c = cross(a, b);
  return Math.atan2(dot(c, n), dot(a, b));
}
