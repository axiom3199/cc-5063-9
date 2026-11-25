// Papercraft tab layout on unfolded 2D pieces.
//
// The unfold produces a closed outline polygon. Its edges are classified into
// ranges: the two boundary curves (v0, v1) and the two seam rulings (u0, u1).
// A "glue seam" is a range of outline edges that must be joined to a partner
// range. Tabs are placed on seam ranges: each range is subdivided into a small
// number of contiguous tab segments, and a trapezoid tab is built on each
// segment's chord, on the exterior side.
//
// Overlap handling (the "retreat a cut line" rule):
//   1. first shrink the tab depth (down to a floor);
//   2. if it still overlaps, drop that tab and slit the seam segment -- i.e.
//      the seam is "retreated" to an open cut, and recorded as a slit;
//   3. the remainder of the layout stays valid and non-overlapping.
//
// Every geometric decision is a 2D polygon test on the correspondence itself.

function dist2(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }

export function polygonArea(poly) {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i], q = poly[(i + 1) % poly.length];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return a / 2;
}

export function polygonPerimeter(poly) {
  let l = 0;
  for (let i = 0; i < poly.length; i++) l += dist2(poly[i], poly[(i + 1) % poly.length]);
  return l;
}

export function pointInPolygon(pt, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1];
    const xj = poly[j][0], yj = poly[j][1];
    if (((yi > pt[1]) !== (yj > pt[1])) &&
      (pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi + 1e-30) + xi)) inside = !inside;
  }
  return inside;
}

function segSegCross(a, b, c, d) {
  const r1x = b[0] - a[0], r1y = b[1] - a[1];
  const r2x = d[0] - c[0], r2y = d[1] - c[1];
  const den = r1x * r2y - r1y * r2x;
  if (Math.abs(den) < 1e-14) return false;
  const t = ((c[0] - a[0]) * r2y - (c[1] - a[1]) * r2x) / den;
  const u = ((c[0] - a[0]) * r1y - (c[1] - a[1]) * r1x) / den;
  return t > 1e-9 && t < 1 - 1e-9 && u > 1e-9 && u < 1 - 1e-9;
}

// Convex/convex-ish overlap: edge crossings or a contained vertex.
export function polygonsOverlap(A, B) {
  for (let i = 0; i < A.length; i++) {
    const a = A[i], b = A[(i + 1) % A.length];
    for (let j = 0; j < B.length; j++) {
      if (segSegCross(a, b, B[j], B[(j + 1) % B.length])) return true;
    }
  }
  for (const p of A) if (pointInPolygon(p, B)) return true;
  for (const p of B) if (pointInPolygon(p, A)) return true;
  return false;
}

// Build a trapezoid tab on chord p->q, offset to the exterior of a CCW outline.
// exterior normal for CCW is the right-hand normal of the edge direction.
export function makeTab(p, q, depth, flareFrac = 0.22) {
  const ex = q[0] - p[0], ey = q[1] - p[1];
  const len = Math.hypot(ex, ey) || 1e-9;
  const ux = ex / len, uy = ey / len;
  const nx = uy, ny = -ux;               // right normal = exterior (CCW)
  const f = flareFrac * Math.min(len, depth * 2);
  return [
    [p[0] + ux * f * 0, p[1] + uy * f * 0],
    [q[0], q[1]],
    [q[0] + nx * depth + ux * f, q[1] + ny * depth + uy * f],
    [p[0] + nx * depth - ux * f, p[1] + ny * depth - uy * f],
  ];
}

// A tab must lie outside the piece. We test only the *offset* vertices (the base
// edge lies on the outline by construction), and the centroid.
export function tabOutsidePiece(tab, outline) {
  const base0 = tab[0], base1 = tab[1];
  const cx = (base0[0] + base1[0]) / 2 + (tab[2][0] + tab[3][0] - base0[0] - base1[0]) / 2;
  const cy = (base0[1] + base1[1]) / 2 + (tab[2][1] + tab[3][1] - base0[1] - base1[1]) / 2;
  if (pointInPolygon([cx, cy], outline)) return false;
  for (let k = 2; k < tab.length; k++) if (pointInPolygon(tab[k], outline)) return false;
  return true;
}

// Split an inclusive edge-index range [lo,hi] (edges lo..hi of the closed
// polygon) into k contiguous groups, each group a [startEdge, endEdge] pair.
export function splitEdgeRun(lo, hi, k) {
  const count = hi - lo + 1;
  const groups = [];
  let acc = 0;
  for (let g = 0; g < k; g++) {
    const start = lo + Math.round(acc);
    acc += count / k;
    const end = g === k - 1 ? hi : lo + Math.round(acc) - 1;
    if (end >= start) groups.push([start, end]);
  }
  return groups;
}

function chordOf(outline, startEdge, endEdge) {
  const p = outline[startEdge];
  const q = outline[(endEdge + 1) % outline.length];
  return { p, q, len: dist2(p, q) };
}

// Lay out tabs.
//   outline:   closed 2D polygon (CCW)
//   seams:     [{ side, edges:[lo,hi], partner, nTabs, priority, label }]
//   opts:      { tabDepth (fraction of reference), tabFloor, tabFlare }
// reference length = min(piece scale, seam chord * something) -- we use the
// geometric mean of the piece's bounding size and each seam's length so tabs are
// neither hair-thin nor absurdly large.
export function layoutTabs(outline, seams, opts = {}) {
  const tabDepthFrac = opts.tabDepth ?? 0.28;
  const tabFloor = opts.tabFloor ?? 0.08;
  const flare = opts.tabFlare ?? 0.2;
  const area = Math.abs(polygonArea(outline));
  const scale = Math.sqrt(Math.max(area, 1e-12));

  const accepted = [];
  const slits = [];
  const rejected = [];

  const ordered = seams.slice().sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));

  for (const seam of ordered) {
    const [lo, hi] = seam.edges;
    const nTabs = Math.max(1, seam.nTabs ?? 2);
    const groups = splitEdgeRun(lo, hi, nTabs);
    for (const [gs, ge] of groups) {
      const { p, q, len } = chordOf(outline, gs, ge);
      // Tab depth is set by the piece scale (papercraft convention: uniform tab
      // depth), capped so a very long seam does not get an absurdly deep tab.
      const ref = scale;
      let depth = Math.min(tabDepthFrac * ref, 0.9 * Math.max(len, scale * 0.15));
      let placed = null;
      for (let t = 0; t < 6; t++) {
        const tab = makeTab(p, q, depth, flare);
        const clashes = accepted.some((a) => polygonsOverlap(tab, a.tab))
          || !tabOutsidePiece(tab, outline);
        if (!clashes) { placed = tab; break; }
        depth *= 0.65;
        if (depth < tabFloor * ref) break;
      }
      if (placed) {
        accepted.push({
          tab: placed, side: seam.side, partner: seam.partner ?? null,
          group: [gs, ge], base: [p, q], depth, label: seam.label || seam.side,
          orientation: tabOrientation(outline, placed),
        });
      } else {
        slits.push({ side: seam.side, group: [gs, ge], base: [p, q], reason: 'tab-overlap', label: seam.label || seam.side });
        rejected.push({ side: seam.side, group: [gs, ge], reason: 'overlap-after-retreat' });
      }
    }
  }

  const intended = seams.reduce((s, sm) => s + (sm.nTabs ?? 2), 0);
  return {
    outline, tabs: accepted, slits, rejected,
    counts: { intended, tabs: accepted.length, slits: slits.length },
    coverage: intended ? accepted.length / intended : 0,
    pieceArea: area, pieceScale: scale,
    valid: accepted.length + slits.length === intended,
  };
}

// +1 if the tab sits on the exterior side of a CCW piece, as a consistency check.
export function tabOrientation(outline, tab) {
  const cx = tab[2][0] + tab[3][0] + tab[0][0] + tab[1][0];
  const cy = tab[2][1] + tab[3][1] + tab[0][1] + tab[1][1];
  return pointInPolygon([cx / 4, cy / 4], outline) ? -1 : 1;
}

export function summaryLayout(layout) {
  return {
    tabs: layout.counts.tabs,
    slits: layout.counts.slits,
    coverage: layout.coverage,
    orientations: layout.tabs.map((t) => t.orientation),
    sides: layout.tabs.map((t) => t.side),
  };
}
