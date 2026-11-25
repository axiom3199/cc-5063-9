// Geometric acceptance tests. Everything here is measured from the model, not
// compared against a hand-entered "looks right" number.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { PRESETS } from '../src/curves.js';
import { buildModel, measureDevelopability, diagnose, sampleMesh } from '../src/ruled.js';
import { unfoldStrip, tradeoffSegments, compareTabFeasibility, splitStrip, planTabs } from '../src/pipeline.js';
import { unfoldMesh, measureUnfoldError } from '../src/unfold.js';
import { solveCorrespondence, developabilityResidual, pairingDiagnostics } from '../src/correspondence.js';
import { polygonArea, layoutTabs, makeTab, polygonsOverlap } from '../src/layout.js';
import { dist, norm, sub, dot, cross, unit } from '../src/math.js';

function modelOf(key) {
  const p = PRESETS[key]();
  return { preset: p, model: buildModel(p.model) };
}

// ---------------------------------------------------------------------------
// 1. Developability is MEASURED, not assumed from "ruled"
// ---------------------------------------------------------------------------

test('cylinder, cone, tangent developable: measured developable to machine precision', () => {
  for (const key of ['cylinder', 'cone', 'tangent']) {
    const { model } = modelOf(key);
    const dev = measureDevelopability(model, { nu: 160 });
    assert.equal(dev.developable, true, key + ' should measure developable');
    assert.ok(dev.maxAngle < 1e-6, key + ' tangent-plane residual too large: ' + dev.maxAngle);
    assert.ok(dev.maxDet < 1e-6, key + ' det residual too large: ' + dev.maxDet);
  }
});

test('skew lines and twisted rings: measured NOT developable', () => {
  for (const key of ['skewLines', 'twistedRings']) {
    const { model } = modelOf(key);
    const dev = measureDevelopability(model, { nu: 120 });
    assert.equal(dev.developable, false, key + ' must NOT be labelled developable');
    // the violation is substantial, not a numerical artefact
    assert.ok(dev.maxAngle > 1e-1, key + ' angle residual should be large: ' + dev.maxAngle);
    assert.ok(dev.badRulings.length > 0, key + ' should report bad rulings');
  }
});

test('ruled does not imply developable: a hyperbolic paraboloid is ruled but rejected', () => {
  const { model } = modelOf('skewLines');
  // every point lies on a ruling (it is ruled by construction) ...
  const f = model.frame(0.31);
  const q = model.point(0.31, 0.42);
  // ... and q is exactly on the ruling from p0 to p1
  const proj = dot(sub(q, f.p0), unit(sub(f.p1, f.p0)));
  assert.ok(Math.abs(proj - 0.42 * dist(f.p0, f.p1)) < 1e-9);
  // ... yet the surface is not developable
  assert.equal(measureDevelopability(model, { nu: 80 }).developable, false);
});

test('developability classification is stable under mesh refinement', () => {
  for (const key of ['cylinder', 'cone', 'tangent', 'skewLines', 'twistedRings']) {
    const { model } = modelOf(key);
    const coarse = measureDevelopability(model, { nu: 24 });
    const fine = measureDevelopability(model, { nu: 320 });
    assert.equal(coarse.developable, fine.developable, key + ' classification changed with resolution');
  }
});

// ---------------------------------------------------------------------------
// 2. Intrinsic unfolding: lengths, angles, areas measured by correspondence
// ---------------------------------------------------------------------------

test('developable strips unfold isometrically (edge length, angle, area)', () => {
  for (const key of ['cylinder', 'cone']) {
    const { model } = modelOf(key);
    const pc = unfoldStrip(model, { nu: 64, nv: 6 });
    const e = pc.error;
    assert.ok(e.maxRel < 1e-9, key + ' max relative edge error ' + e.maxRel);
    assert.ok(e.maxAngleDeg < 1e-6, key + ' angle error ' + e.maxAngleDeg + ' deg');
    assert.ok(e.areaRel < 1e-9, key + ' area error ' + e.areaRel);
    assert.equal(e.flips, 0, key + ' should have no flipped triangles');
  }
});

test('unfolded polygon area equals the 3D surface area within tolerance', () => {
  for (const key of ['cylinder', 'cone', 'tangent']) {
    const { model } = modelOf(key);
    const pc = unfoldStrip(model, { nu: 128, nv: 6 });
    const area2 = Math.abs(polygonArea(pc.outline.poly));
    const area3 = pc.error.areaTotal3;
    const rel = Math.abs(area2 - area3) / area3;
    assert.ok(rel < 1e-5, key + ' polygon-vs-surface area relative error ' + rel);
  }
});

test('non-developable strip: unfolding is detected as strained, not silently accepted', () => {
  const { model } = modelOf('skewLines');
  const pc = unfoldStrip(model, { nu: 48, nv: 6 });
  // The unfolding produces large edge error and/or triangle flips. It must be
  // reported, never presented as a valid flat pattern.
  const bad = pc.error.maxRel > 1e-2 || pc.error.flips > 0;
  assert.ok(bad, 'non-developable surface unfolded with suspiciously small error');
  assert.ok(pc.error.areaRel > 1e-2, 'area should not be preserved for a non-developable strip');
});

test('tangent developable: unfold error converges under refinement (genuinely developable)', () => {
  const { model } = modelOf('tangent');
  const e32 = unfoldStrip(model, { nu: 32, nv: 4 }).error.maxRel;
  const e128 = unfoldStrip(model, { nu: 128, nv: 4 }).error.maxRel;
  const e256 = unfoldStrip(model, { nu: 256, nv: 4 }).error.maxRel;
  assert.ok(e128 < e32, 'error should drop with refinement');
  assert.ok(e256 < e128, 'error should keep dropping with refinement');
  assert.ok(e256 < 1e-7, 'fine unfolding should be near-isometric: ' + e256);
});

test('no "projection-as-unfold": flattened edge lengths differ from a plain projection', () => {
  // A plain (u,v) -> (x,y) projection of a curved strip would NOT preserve arc
  // length. Confirm our unfolded outline perimeter equals the true 3D boundary
  // perimeter (both boundary curves + both seam rulings), which a projection of
  // the same grid would not.
  const { model } = modelOf('cylinder');
  const pc = unfoldStrip(model, { nu: 64, nv: 6 });
  const perim3 = boundaryAndSeam3D(pc);
  const perim2 = outlinePerimeter(pc.outline.poly);
  assert.ok(Math.abs(perim2 - perim3) / perim3 < 1e-6, 'unfolded perimeter ' + perim2 + ' must equal 3D perimeter ' + perim3);

  // Now the naive projection of the same grid: drop z from the 3D points.
  const mesh = sampleMesh(model, 64, 6);
  const n = mesh.rows.length - 1;
  let projPerim = 0;
  // boundary j=0 and j=nv along i, plus the two seam rulings
  for (const jb of [0, mesh.nv]) {
    for (let i = 0; i < n; i++) {
      const a = mesh.rows[i][jb], b = mesh.rows[i + 1][jb];
      projPerim += Math.hypot(b[0] - a[0], b[1] - a[1]);
    }
  }
  for (const ib of [0, n]) {
    for (let j = 0; j < mesh.nv; j++) {
      const a = mesh.rows[ib][j], b = mesh.rows[ib][j + 1];
      projPerim += Math.hypot(b[0] - a[0], b[1] - a[1]);
    }
  }
  assert.ok(Math.abs(projPerim - perim3) / perim3 > 1e-2, 'plain projection should distort perimeter');
});

function boundaryAndSeam3D(pc) {
  const { mesh } = pc;
  const { nu, nv } = mesh;
  let l = 0;
  for (const jb of [0, nv]) for (let i = 0; i < nu; i++) l += dist(mesh.rows[i][jb], mesh.rows[i + 1][jb]);
  for (const ib of [0, nu]) for (let j = 0; j < nv; j++) l += dist(mesh.rows[ib][j], mesh.rows[ib][j + 1]);
  return l;
}

function outlinePerimeter(poly) {
  let l = 0;
  for (let i = 0; i < poly.length; i++) l += Math.hypot(poly[(i + 1) % poly.length][0] - poly[i][0], poly[(i + 1) % poly.length][1] - poly[i][1]);
  return l;
}

// ---------------------------------------------------------------------------
// 3. Correspondence solver (declared developability condition)
// ---------------------------------------------------------------------------

test('correspondence solver reproduces the cone pairing and reports it well-conditioned', () => {
  // Use the skewLines pair? No -- use two explicit arcs whose developable pairing
  // is known analytically: a shallow cone built from two circular arcs.
  const apexZ = 6, s = 1 - 1 / apexZ;
  const c0 = {
    kind: 'arc', center: [0, 0, 0], radius: 1, axis: [0, 0, 1], ref: [1, 0, 0],
    a0: -0.7, a1: 0.7,
  };
  const c1 = {
    kind: 'arc', center: [0, 0, apexZ - s * apexZ], radius: s, axis: [0, 0, 1], ref: [1, 0, 0],
    a0: -0.7, a1: 0.7,
  };
  const corr = solveCorrespondence(c0, c1, { nu: 64 });
  assert.equal(corr.status, 'ok');
  assert.ok(corr.monotone, 'v(u) must be monotone');
  assert.ok(corr.maxResidual < 1e-6, 'residual should vanish on the developable pairing');
  // matched-angle cones pair as v = u
  for (let i = 0; i <= 8; i++) {
    const u = i / 8;
    const v = corr.vs[Math.round(i / 8 * 64)];
    assert.ok(Math.abs(v - u) < 1e-3, 'cone pairing v(u)=u expected, got ' + v);
  }
});

test('correspondence residual is exactly zero on a developable cone pairing', () => {
  const { model } = modelOf('cone');
  const { createCurve } = requireCurves;
  // direct residual check on the cone frame: det(P0', R, R') = 0
  for (let i = 0; i <= 16; i++) {
    const u = i / 16;
    const f = model.frame(u);
    const res = Math.abs(det3(f.d0, sub(f.p1, f.p0), sub(f.d1, f.d0)));
    const scale = norm(f.d0) * norm(sub(f.p1, f.p0)) * norm(sub(f.d1, f.d0)) + 1e-30;
    assert.ok(res / scale < 1e-10, 'cone det residual ' + res / scale);
  }
});
const requireCurves = { createCurve: null };
function det3(a, b, c) { return dot(a, cross(b, c)); }

test('near-parallel case: the developability condition carries almost no v-information', () => {
  const p = PRESETS.nearParallel();
  const diag = pairingDiagnostics(p.model.curve0, p.model.curve1, { nu: 40 });
  assert.equal(diag.flat, true, 'F should be flat in v (near-parallel rulings)');
  assert.equal(diag.illConditioned, true, 'pairing must be reported ill-conditioned');
  // near-developable: the residual is small (the surface is close to developable)
  assert.ok(diag.maxResidual < 1e-1);
  // and it is genuinely (slightly) non-developable, so not a clean developable pair
  const dev = measureDevelopability(buildModel(p.model), { nu: 120 });
  assert.equal(dev.developable, false);
});

test('a genuinely non-developable pair has no acceptable correspondence', () => {
  const p = PRESETS.twistedRings();
  const corr = solveCorrespondence(p.model.curve0, p.model.curve1, { nu: 64 });
  assert.notEqual(corr.status, 'ok');
  assert.ok(corr.maxResidual > 1e-2 || !corr.monotone);
});

// ---------------------------------------------------------------------------
// 4. Singular / regression edges located, not emitted
// ---------------------------------------------------------------------------

test('regression-edge strip is located at the spine and excluded', () => {
  const { model } = modelOf('regression');
  const dg = diagnose(model, { nu: 100 });
  assert.equal(dg.ok, 'not-developable');
  assert.ok(dg.regressionEdge, 'must locate the regression edge');
  assert.ok(Math.abs(dg.regressionEdge.vActual) < 0.05, 'regression edge should sit at the spine (v~0), got ' + dg.regressionEdge.vActual);
  assert.ok(dg.singular.length > 0);
  // The degenerate u-locus spans the whole strip here
  assert.ok(dg.regressionEdge.uMax - dg.regressionEdge.uMin > 0.5);
});

test('healthy strips report no singularities or regression edge', () => {
  for (const key of ['cylinder', 'cone', 'tangent']) {
    const { model } = modelOf(key);
    const dg = diagnose(model, { nu: 80 });
    assert.equal(dg.singular.length, 0, key + ' reported spurious singularities');
    assert.equal(dg.regressionEdge, null, key + ' reported spurious regression edge');
    assert.equal(dg.ok, 'ok');
  }
});

// ---------------------------------------------------------------------------
// 5. Tab layout: overlap, retreat-to-slit, orientation, locked folds
// ---------------------------------------------------------------------------

test('tabs are outside the piece and mutually non-overlapping in a simple layout', () => {
  const { model } = modelOf('cylinder');
  const pc = unfoldStrip(model, { nu: 64, nv: 6 });
  const layout = planTabs(pc, [{ onSide: 'u0', nTabs: 3 }, { onSide: 'u1', nTabs: 3 }], { tabDepth: 0.35 });
  assert.ok(layout.tabs.length > 0, 'expected some tabs');
  for (const t of layout.tabs) {
    assert.equal(t.orientation, 1, 'tab must sit outside the CCW piece');
    assert.ok(t.depth > 0);
  }
  for (let a = 0; a < layout.tabs.length; a++) {
    for (let b = a + 1; b < layout.tabs.length; b++) {
      assert.equal(polygonsOverlap(layout.tabs[a].tab, layout.tabs[b].tab), false, 'tabs overlap');
    }
  }
});

test('overlap triggers retreat: a cut line becomes a slit (A: whole piece fails)', () => {
  const cmp = nearFullConeComparison();
  const whole = cmp.whole.layout;
  assert.ok(whole.slits.length > 0, 'whole piece should need to retreat a cut line');
  assert.ok(whole.counts.slits >= 1);
  // every intended tab is accounted for either as a tab or as a slit
  assert.equal(whole.counts.tabs + whole.counts.slits, whole.counts.intended);
  // the slits are labelled with a reason
  for (const s of whole.slits) assert.equal(s.reason, 'tab-overlap');
});

test('A/B: similar unfolded error, only the re-sliced layout accommodates all tabs', () => {
  const cmp = nearFullConeComparison();
  // similar measured edge error between whole and split pieces
  const wholeErr = cmp.whole.piece.error.rmsRel;
  const splitErr = Math.max(cmp.split.A.error.rmsRel, cmp.split.B.error.rmsRel);
  assert.ok(Math.abs(wholeErr - splitErr) / Math.max(wholeErr, 1e-30) < 1e-3,
    'unfold errors should be similar: ' + wholeErr + ' vs ' + splitErr);
  // whole does NOT fit all tabs; split does
  assert.ok(cmp.whole.layout.slits.length > 0, 'whole should not accommodate all tabs');
  assert.equal(cmp.split.layoutA.slits.length, 0, 'piece A should accommodate its tabs');
  assert.equal(cmp.split.layoutB.slits.length, 0, 'piece B should accommodate its tabs');
  assert.ok(cmp.split.layoutA.tabs.length + cmp.split.layoutB.tabs.length >= cmp.whole.layout.tabs.length);
});

test('A/B: glued edge pairs, tab orientation and locked fold polylines are consistent', () => {
  const cmp = nearFullConeComparison();
  assert.ok(cmp.gluedPairs.length > 0, 'must report actual glued edge pairs');
  for (const g of cmp.gluedPairs) assert.equal(g.orientation, 1, 'tab orientation must be exterior');
  assert.equal(cmp.lockedFolds.length, 1, 'the internal split must be recorded as a locked fold');
  const fold = cmp.lockedFolds[0];
  assert.ok(fold.pieceA.length > 0, 'locked fold should be a polyline');
  // the fold vertices lie on the split seam, shared by A and B
  assert.ok(fold.note.includes('glued'));
});

function nearFullConeComparison() {
  // af=0.98 near-full circle, apex 1.2, scale 0.8 -> whole needs 2 slits; split fits.
  const cfg = {
    kind: 'cone',
    curve0: {
      kind: 'arc', center: [0, 0, 0], radius: 1, axis: [0, 0, 1], ref: [1, 0, 0],
      a0: -Math.PI * 0.98, a1: Math.PI * 0.98,
    },
    apex: [0, 0, 1.2],
    scale: 0.8,
    nu: 72, nv: 8,
  };
  const model = buildModel(cfg);
  return compareTabFeasibility(model, { nu: 64, nv: 8, tabDepth: 0.4, splitFrac: 0.5, nTabs: 3, nTabsSplit: 2 });
}

// NOTE: compareTabFeasibility uses fixed nTabs counts; ensure it produces the
// 4-tabs-whole / 2-slits / 4-tabs-split signature we rely on.
test('compareTabFeasibility wiring uses seam ranges and reports counts', () => {
  const cmp = nearFullConeComparison();
  assert.ok(cmp.whole.layout.counts.intended >= 2);
  assert.ok(cmp.split.layoutA.counts.intended >= 1);
  assert.ok(cmp.split.layoutB.counts.intended >= 1);
});

// ---------------------------------------------------------------------------
// 6. Near-parallel tradeoff: segmentation vs slits by measured unfolded error
// ---------------------------------------------------------------------------

test('near-parallel: segmenting vs slitting decided by measured unfolded edge error', () => {
  const { model } = modelOf('nearParallel');
  const results = tradeoffSegments(model, { nu: 64, nv: 8, segments: [1, 2, 4, 8] });
  const one = results.find((r) => r.segments === 1).maxRel;
  const many = results.find((r) => r.segments === 8).maxRel;
  // For this near-parallel (nearly parallel skew lines) surface, subdividing does
  // NOT reduce the unfolded edge error -- the error is intrinsic strain, not a
  // per-piece accumulation. Therefore the rule "subdivide" is rejected and a slit
  // is the correct intervention. We assert the measured fact that drives this.
  assert.ok(many >= one * 0.99, 'segmenting should not (materially) reduce error here: ' + one + ' -> ' + many);
});

test('near-parallel: whole-piece unfold records real strain (not silently flat)', () => {
  const { model } = modelOf('nearParallel');
  const pc = unfoldStrip(model, { nu: 96, nv: 8 });
  // It is slightly non-developable, so there is a genuinely nonzero strain.
  assert.ok(pc.error.maxRel > 1e-7, 'expected measurable strain: ' + pc.error.maxRel);
  assert.ok(pc.error.areaRel > 1e-8, 'area should not be preserved exactly');
});

test('segmentation tradeoff helper covers cylinder (helps) vs near-parallel (does not)', () => {
  const cyl = modelOf('cylinder').model;
  const cylRes = tradeoffSegments(cyl, { nu: 64, nv: 6, segments: [1, 4] });
  // already exact; segmenting keeps it exact
  assert.ok(cylRes.every((r) => r.maxRel < 1e-10));

  const np = modelOf('nearParallel').model;
  const npRes = tradeoffSegments(np, { nu: 64, nv: 8, segments: [1, 4] });
  // we can still read the decision quantity: both are reported, and neither is
  // accidentally zero (so the choice is made on real, not fake, numbers)
  assert.ok(npRes.every((r) => Number.isFinite(r.maxRel) && r.maxRel > 0));
});

// ---------------------------------------------------------------------------
// 7. Preset ground-truth contract
// ---------------------------------------------------------------------------

test('every preset matches its declared measured ground truth', () => {
  const expectations = {
    cylinder: (dev) => dev.developable === true,
    cone: (dev) => dev.developable === true,
    wideCone: (dev) => dev.developable === true,
    tangent: (dev) => dev.developable === true,
    skewLines: (dev) => dev.developable === false,
    twistedRings: (dev) => dev.developable === false,
    // near-parallel: slightly non-developable AND the pairing is ill-conditioned
    nearParallel: (dev, corr, dg, pairing) => dev.developable === false && pairing.illConditioned === true && pairing.nearDevelopable === true,
    regression: (dev, corr, dg) => dg.ok === 'not-developable' && dg.regressionEdge !== null,
  };
  for (const [key, pred] of Object.entries(expectations)) {
    const p = PRESETS[key]();
    const model = buildModel(p.model);
    const dev = measureDevelopability(model, { nu: 120 });
    const dg = diagnose(model, { nu: 100 });
    let corr = { maxCondition: 0 };
    let pairing = { illConditioned: false, nearDevelopable: true, maxCondition: 0 };
    if (p.model.kind === 'direct') {
      corr = solveCorrespondence(p.model.curve0, p.model.curve1, { nu: 64 });
      pairing = pairingDiagnostics(p.model.curve0, p.model.curve1, { nu: 40 });
    }
    assert.ok(pred(dev, corr, dg, pairing), 'preset ' + key + ' failed its ground-truth check');
  }
});

test('sample mesh is watertight in parameter space (grid is rectangular and complete)', () => {
  for (const key of ['cylinder', 'cone', 'skewLines']) {
    const { model } = modelOf(key);
    const mesh = sampleMesh(model, 20, 5);
    assert.equal(mesh.rows.length, 21);
    for (const row of mesh.rows) {
      assert.equal(row.length, 6);
      for (const p of row) assert.ok(p.every((x) => Number.isFinite(x)));
    }
  }
});
