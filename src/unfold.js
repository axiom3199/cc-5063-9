// Intrinsic unfolding of the sampled ruled mesh.
//
// We DO NOT project 3D vertices into the plane. Instead we triangulate the mesh,
// then flatten it intrinsically by laying triangles edge-by-edge, each new
// triangle placed from its three measured 3D side lengths (law of cosines). This
// realises the mesh's metric in 2D up to the residual strain of the discretisation.
// Every 3D edge gets a correspondence-tagged 2D image, and errors are measured
// directly against that correspondence.
//
// A mesh is triangulated with a consistent split direction (both diagonals used
// alternately would break the ruling correspondence, so we use the (i,j)-(i+1,j+1)
// diagonal everywhere; the alternative is a parameter).

import { sub, add, mul, dot, cross, unit, norm, dist, angleBetween } from './math.js';

export function triangulate(mesh, diagonal = 'main') {
  const { rows, nu, nv } = mesh;
  const tris = [];
  const vid = (i, j) => i * (nv + 1) + j;
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nv; j++) {
      if (diagonal === 'main') {
        // (i,j)-(i+1,j+1) diagonal
        tris.push([vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)]);
        tris.push([vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)]);
      } else {
        tris.push([vid(i, j), vid(i + 1, j), vid(i, j + 1)]);
        tris.push([vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]);
      }
    }
  }
  return { tris, vid, rows, nu, nv, nVerts: (nu + 1) * (nv + 1) };
}

// 3D edge length from the sampled mesh (exact for the sampled ruling endpoints,
// since rulings are straight: |P(i,j+1)-P(i,j)| equals the sub-chord only approx;
// for j-internal edges it is the sampled chord, which is what the mesh realises).
function edge3(mesh, a, b) {
  const { nu, nv } = mesh;
  const ij = (k) => [Math.floor(k / (nv + 1)), k % (nv + 1)];
  const [i0, j0] = ij(a), [i1, j1] = ij(b);
  return dist(mesh.rows[i0][j0], mesh.rows[i1][j1]);
}

// Flatten the triangulation intrinsically.
// Returns { pos, tri2d, metrics, ok, nonFinite, degenerateEdges }.
//
// Degenerate ruling policy: a ruling whose sampled edge length is ~0 (a collapsed
// ruling, or a near-parallel pair closer than `tol`) carries no metric. We do NOT
// invent a direction for it (the old code shoved the vertex to a neighbour + an
// arbitrary 1e-3 offset, silently producing folded triangles that looked placed).
// Such vertices are left NaN, counted, and `ok=false`. Callers either retreat to a
// developable u-window (pipeline.unfoldStrip) or surface the failure.
export function unfoldMesh(mesh, opts = {}) {
  const tri = triangulate(mesh, opts.diagonal || 'main');
  const { tris, nVerts, nu, nv } = tri;
  const pos = new Float64Array(nVerts * 2).fill(NaN);
  const placed = new Uint8Array(nVerts);

  // 3D positions for length queries
  const P3 = new Array(nVerts);
  for (let i = 0; i <= nu; i++) for (let j = 0; j <= nv; j++) P3[tri.vid(i, j)] = mesh.rows[i][j];
  const len3 = (a, b) => dist(P3[a], P3[b]);

  // characteristic length and the degeneracy threshold (relative, scale-free)
  let charLen = 0;
  for (const t of tris) for (let e = 0; e < 3; e++) charLen = Math.max(charLen, len3(t[e], t[(e + 1) % 3]));
  const tol = 1e-9 * Math.max(1, charLen);
  let degenerateEdges = 0;
  const edgeDegenerate = new Map();
  for (const t of tris) {
    for (let e = 0; e < 3; e++) {
      const a = t[e], b = t[(e + 1) % 3];
      const k = a < b ? a + ':' + b : b + ':' + a;
      if (!edgeDegenerate.has(k)) {
        const deg = len3(a, b) < tol;
        edgeDegenerate.set(k, deg);
        if (deg) degenerateEdges++;
      }
    }
  }

  // adjacency
  const edgeMap = new Map(); // "a:b" -> list of tri index
  tris.forEach((t, ti) => {
    for (let e = 0; e < 3; e++) {
      const a = t[e], b = t[(e + 1) % 3];
      const k = a < b ? a + ':' + b : b + ':' + a;
      if (!edgeMap.has(k)) edgeMap.set(k, []);
      edgeMap.get(k).push(ti);
    }
  });

  // seed: first triangle with a non-degenerate base edge and non-zero area
  let seed = -1;
  for (let ti = 0; ti < tris.length; ti++) {
    const t = tris[ti];
    if (len3(t[0], t[1]) >= tol && len3(t[0], t[2]) >= tol && len3(t[1], t[2]) >= tol) { seed = ti; break; }
  }
  if (seed < 0) {
    // whole mesh is metric-degenerate: nothing can be placed soundly
    return { ...tri, pos, placed, P3, ok: false, nonFinite: nVerts, degenerateEdges };
  }
  const t0 = tris[seed];
  placeSeed(t0);
  const done = new Uint8Array(tris.length);
  done[seed] = 1;
  const queue = [seed];

  const kOf = (a, b) => (a < b ? a + ':' + b : b + ':' + a);
  // BFS across shared edges. We only cross an edge whose 3D length is non-degenerate:
  // a collapsed ruling cannot determine the third vertex, so we leave it unresolved
  // rather than fabricating a placement (which would fake a fold-free flattening).
  while (queue.length) {
    const ti = queue.shift();
    const t = tris[ti];
    for (let e = 0; e < 3; e++) {
      const a = t[e], b = t[(e + 1) % 3], c = t[(e + 2) % 3];
      const k = kOf(a, b);
      if (edgeDegenerate.get(k)) continue;      // collapsed edge: no metric
      const nbrs = edgeMap.get(k);
      for (const tj of nbrs) {
        if (tj === ti || done[tj]) continue;
        const nt = tris[tj];
        const nvtx = nt.find((x) => x !== a && x !== b);
        if (placed[c] && placed[nvtx] && placed[a] && placed[b]) { done[tj] = 1; queue.push(tj); continue; }
        if (!placed[a] || !placed[b]) continue;
        const lengthAC = len3(a, nvtx);
        const lengthBC = len3(b, nvtx);
        if (lengthAC < tol || lengthBC < tol) continue;  // vertex coincides: undetermined
        const pa = get(pos, a), pb = get(pos, b);
        const cand = placeThird(pa, pb, lengthAC, lengthBC);
        const pc = get(pos, c);
        const sideRef = sideOf(pa, pb, pc);
        let chosen = cand[0];
        const s0 = sideOf(pa, pb, cand[0]);
        const s1 = sideOf(pa, pb, cand[1]);
        if (placed[c]) {
          const opposite = -sideRef;
          chosen = Math.abs(s0 - opposite) <= Math.abs(s1 - opposite) ? cand[0] : cand[1];
        } else {
          chosen = cand[0];
        }
        if (!placed[nvtx]) {
          set(pos, nvtx, chosen);
          placed[nvtx] = 1;
        }
        done[tj] = 1;
        queue.push(tj);
      }
    }
  }

  // Local fallback for rulings the BFS could not cross because they collapsed.
  // A collapsed u-column lies at a single 3D point, so every v along it must share
  // one 2D image; copy it from the column's already-placed vertices. This is the
  // exact image of a genuinely zero-length column, not an invented direction, so it
  // cannot bend or scale the fabric. Vertices with no coincident placed neighbour
  // stay NaN -> ok=false, and the caller retreats or reports failure.
  for (let i = 0; i <= nu; i++) {
    let anchorK = -1;
    for (let j = 0; j <= nv; j++) { const k = tri.vid(i, j); if (placed[k]) { anchorK = k; break; } }
    if (anchorK < 0) continue;
    const anchorP = get(pos, anchorK);
    for (let j = 0; j <= nv; j++) {
      const k = tri.vid(i, j);
      if (placed[k]) continue;
      if (dist(mesh.rows[i][j], P3[anchorK]) < tol) {
        set(pos, k, [anchorP[0], anchorP[1]]);
        placed[k] = 1;
      }
    }
  }

  let nonFinite = 0;
  for (let k = 0; k < nVerts; k++) if (!placed[k]) nonFinite++;

  return { ...tri, pos, placed, P3, ok: nonFinite === 0, nonFinite, degenerateEdges };

  function placeSeed(t) {
    set(pos, t[0], [0, 0]);
    const l01 = len3(t[0], t[1]);
    set(pos, t[1], [l01, 0]);
    const l02 = len3(t[0], t[2]);
    const l12 = len3(t[1], t[2]);
    const pt0 = [0, 0], pt1 = [l01, 0];
    const cand = placeThird(pt0, pt1, l02, l12);
    // deterministic: +y side
    const chosen = cand[0][1] >= 0 ? cand[0] : cand[1];
    set(pos, t[2], chosen);
    placed[t[0]] = placed[t[1]] = placed[t[2]] = 1;
  }
}

function get(pos, k) { return [pos[2 * k], pos[2 * k + 1]]; }
function set(pos, k, p) { pos[2 * k] = p[0]; pos[2 * k + 1] = p[1]; }
function sideOf(a, b, c) { return Math.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) || 0; }

function placeThird(a, b, lac, lbc) {
  const d = Math.hypot(b[0] - a[0], b[1] - a[1]);
  if (d < 1e-12) return [[a[0], a[1] + lac], [a[0], a[1] - lac]];
  const ex = [(b[0] - a[0]) / d, (b[1] - a[1]) / d];
  const ey = [-ex[1], ex[0]];
  let x = (lac * lac - lbc * lbc + d * d) / (2 * d);
  let h2 = lac * lac - x * x;
  if (h2 < 0) h2 = 0;
  const h = Math.sqrt(h2);
  return [
    [a[0] + ex[0] * x + ey[0] * h, a[1] + ex[1] * x + ey[1] * h],
    [a[0] + ex[0] * x - ey[0] * h, a[1] + ex[1] * x - ey[1] * h],
  ];
}

// ---- Direct correspondence-based error measurement ----
// For each 3D mesh edge (u-edge, v-edge, ruling-diagonal) compare its length to
// the 2D image of the same vertex pair. This is the required "directly measured
// by correspondence" error; no visual flatness proxy.
export function measureUnfoldError(mesh, unfold, opts = {}) {
  const { nu, nv } = mesh;
  const vid = (i, j) => i * (nv + 1) + j;
  const pos = unfold.pos;
  const get2 = (k) => [pos[2 * k], pos[2 * k + 1]];
  const rows = [];
  let maxAbs = 0, maxRel = 0, sumRel2 = 0, n = 0, nFiniteFail = 0;
  const addErr = (kind, i, j, a, b) => {
    const l3 = dist(mesh.rows[a[0]][a[1]], mesh.rows[b[0]][b[1]]);
    const qa = get2(vid(a[0], a[1])), qb = get2(vid(b[0], b[1]));
    const finite2d = Number.isFinite(qa[0]) && Number.isFinite(qa[1]) && Number.isFinite(qb[0]) && Number.isFinite(qb[1]);
    const l2 = finite2d ? Math.hypot(qa[0] - qb[0], qa[1] - qb[1]) : NaN;
    const abs = Math.abs(l2 - l3);
    const rel = finite2d ? (l3 > 1e-9 ? abs / l3 : 0) : Infinity;
    rows.push({ kind, i, j, l3, l2, abs, rel, finite: finite2d });
    maxAbs = Math.max(maxAbs, abs); maxRel = Math.max(maxRel, rel);
    if (finite2d) { sumRel2 += rel * rel; n++; } else { nFiniteFail++; }
  };
  const vAfter = (j) => j / nv;
  for (let i = 0; i <= nu; i++) {
    for (let j = 0; j <= nv; j++) {
      if (j < nv) addErr('u-edge', i, j, [i, j], [i, j + 1]);
      if (i < nu) addErr('v-edge', i, j, [i, j], [i + 1, j]);
      if (i < nu && j < nv) addErr('diag', i, j, [i, j], [i + 1, j + 1]);
    }
  }
  const rmsRel = Math.sqrt(sumRel2 / Math.max(1, n));

  // A non-finite 2D image means the flattening could not place that vertex. That
  // is a hard failure, not a small residual: flag it and never let a NaN quietly
  // satisfy "maxRel < tol".
  let nonFinite2d = 0;
  for (let k = 0; k < pos.length; k++) if (!Number.isFinite(pos[k])) nonFinite2d++;

  // boundary arc-length comparison and end-to-end closure
  const boundary = [];
  for (const jb of [0, nv]) {
    let l3 = 0, l2 = 0, finite = true;
    for (let i = 0; i < nu; i++) {
      l3 += dist(mesh.rows[i][jb], mesh.rows[i + 1][jb]);
      const qa = get2(vid(i, jb)), qb = get2(vid(i + 1, jb));
      if (!Number.isFinite(qa[0]) || !Number.isFinite(qa[1]) || !Number.isFinite(qb[0]) || !Number.isFinite(qb[1])) { finite = false; continue; }
      l2 += Math.hypot(qa[0] - qb[0], qa[1] - qb[1]);
    }
    boundary.push({ side: jb === 0 ? 'C0' : 'C1', length3: l3, length2: finite ? l2 : NaN, rel: finite ? Math.abs(l2 - l3) / Math.max(1e-12, l3) : Infinity });
  }

  // angle preservation: for each quad corner compare 3D angle to 2D angle
  let maxAngleDeg = 0, angleSamples = 0, sumAng2 = 0;
  for (let i = 0; i <= nu; i++) {
    for (let j = 0; j <= nv; j++) {
      const triples = [];
      if (i > 0 && j > 0) triples.push([[i, j], [i - 1, j], [i, j - 1]]);
      if (i < nu && j > 0) triples.push([[i, j], [i + 1, j], [i, j - 1]]);
      if (i > 0 && j < nv) triples.push([[i, j], [i - 1, j], [i, j + 1]]);
      if (i < nu && j < nv) triples.push([[i, j], [i + 1, j], [i, j + 1]]);
      for (const [c, a, b] of triples) {
        const a3 = angleBetween(sub(mesh.rows[a[0]][a[1]], mesh.rows[c[0]][c[1]]), sub(mesh.rows[b[0]][b[1]], mesh.rows[c[0]][c[1]]));
        const pc = get2(vid(c[0], c[1])), pa = get2(vid(a[0], a[1])), pb = get2(vid(b[0], b[1]));
        if (![...pc, ...pa, ...pb].every(Number.isFinite)) { maxAngleDeg = Infinity; continue; }
        const a2 = angleBetween([pa[0] - pc[0], pa[1] - pc[1], 0], [pb[0] - pc[0], pb[1] - pc[1], 0]);
        const diff = Math.abs(a3 - a2) * 180 / Math.PI;
        maxAngleDeg = Math.max(maxAngleDeg, diff);
        sumAng2 += diff * diff; angleSamples++;
      }
    }
  }
  const rmsAngleDeg = Math.sqrt(sumAng2 / Math.max(1, angleSamples));

  // area comparison per triangle
  let maxAreaRel = 0, sumAreaRel2 = 0, triN = 0, total3 = 0, total2 = 0;
  const tri = triangulate(mesh, opts.diagonal || 'main');
  for (const t of tri.tris) {
    const p = t.map((k) => mesh.rows[Math.floor(k / (nv + 1))][k % (nv + 1)]);
    const a3 = 0.5 * norm(cross(sub(p[1], p[0]), sub(p[2], p[0])));
    const q = t.map((k) => get2(k));
    const a2 = q.every((c) => Number.isFinite(c[0]) && Number.isFinite(c[1]))
      ? 0.5 * Math.abs((q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1]))
      : NaN;
    const rel = Number.isFinite(a2) ? Math.abs(a2 - a3) / Math.max(1e-12, a3) : Infinity;
    maxAreaRel = Math.max(maxAreaRel, rel); if (Number.isFinite(rel)) sumAreaRel2 += rel * rel; triN++;
    total3 += a3; if (Number.isFinite(a2)) total2 += a2;
  }
  const rmsAreaRel = Math.sqrt(sumAreaRel2 / Math.max(1, triN));

  // fold check: compare triangle orientation (all should be +1 if we unfolded consistently)
  let flips = 0;
  for (const t of tri.tris) {
    const q = t.map((k) => get2(k));
    if (!q.every((c) => Number.isFinite(c[0]) && Number.isFinite(c[1]))) { flips++; continue; }
    const s = Math.sign((q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1]));
    if (s <= 0) flips++;
  }

  return {
    edges: rows, maxAbs, maxRel, rmsRel,
    boundary, maxAngleDeg, rmsAngleDeg,
    maxAreaRel, rmsAreaRel, areaTotal3: total3, areaTotal2: total2, areaRel: Math.abs(total2 - total3) / Math.max(1e-12, total3),
    flips, nTris: tri.tris.length,
    nonFinite2d, nonFiniteEdges: nFiniteFail,
    // A pass requires every edge image to be finite AND within tolerance with no
    // folded triangles. Non-finite coordinates are a hard failure, never a pass.
    passed: nonFinite2d === 0 && nFiniteFail === 0 && maxRel < (opts.relTol ?? 1e-6) && flips === 0,
  };
}

export function bounds2d(pos) {
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let k = 0; k < pos.length; k += 2) {
    if (Number.isNaN(pos[k])) continue;
    xmin = Math.min(xmin, pos[k]); xmax = Math.max(xmax, pos[k]);
    ymin = Math.min(ymin, pos[k + 1]); ymax = Math.max(ymax, pos[k + 1]);
  }
  return { xmin, xmax, ymin, ymax, w: xmax - xmin, h: ymax - ymin };
}

export function transform2d(pos, scale, tx, ty) {
  const out = new Float64Array(pos.length);
  for (let k = 0; k < pos.length; k += 2) {
    out[k] = pos[k] * scale + tx;
    out[k + 1] = pos[k + 1] * scale + ty;
  }
  return out;
}

// Polygons for display: boundary loops and ruling lines.
export function unfoldTopology(mesh, unfold) {
  const { nu, nv } = mesh;
  const vid = (i, j) => i * (nv + 1) + j;
  const pos = unfold.pos;
  const pt = (i, j) => [pos[2 * vid(i, j)], pos[2 * vid(i, j) + 1]];
  const rulings = [];
  for (let i = 0; i <= nu; i++) rulings.push([pt(i, 0), pt(i, nv)]);
  const boundary0 = [];
  const boundary1 = [];
  for (let i = 0; i <= nu; i++) { boundary0.push(pt(i, 0)); boundary1.push(pt(i, nv)); }
  const tris2d = [];
  const tri = triangulate(mesh);
  for (const t of tri.tris) {
    tris2d.push(t.map((k) => [pos[2 * k], pos[2 * k + 1]]));
  }
  return { rulings, boundary0, boundary1, tris2d };
}
