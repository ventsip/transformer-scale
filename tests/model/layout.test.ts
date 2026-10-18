import { describe, expect, it } from 'vitest';
import { THICKNESS_PER_BYTE, boxBounds, enclosingBox, layoutTensors, type Box, type PlacedTensor } from '../../src/layout';
import { MODELS } from '../../src/model/config';
import { DTYPES } from '../../src/model/dtypes';
import { buildModel, floorCount, layerFloor } from '../../src/model/tensors';
import { MODEL_IDS, baseOptions, type Case } from '../support/matrix';

/** Settings whose layouts differ most: eager score grids, expanded chains, static caches, decode. */
const LAYOUT_CASES: Case[] = MODEL_IDS.flatMap((id) => {
  const cfg = MODELS[id];
  return [
    baseOptions(cfg, { T: 700 }),
    baseOptions(cfg, { T: 700, attention: 'eager', elementwise: 'expanded' }),
    baseOptions(cfg, { mode: 'decode', T: 700, kvCache: 'static', attention: 'eager', elementwise: 'expanded', cacheCapacity: 4096 }),
    baseOptions(cfg, { mode: 'decode', T: 700, ...(cfg.mla ? { mlaCache: 'latent' as const } : {}) }),
  ].map((options) => ({ name: `${id} ${options.mode} ${options.attention} ${options.kvCache} ${options.elementwise}`, id, cfg, options }));
});

/** Pairs of boxes of different tensors that overlap on the floor plan (XZ), found with a sweep along X. */
function overlaps(placed: PlacedTensor[]): string[] {
  const boxes = placed.flatMap((p) => p.boxes.map((b) => ({ id: p.spec.id, b })));
  boxes.sort((a, c) => a.b.cx - a.b.sx / 2 - (c.b.cx - c.b.sx / 2));
  const found: string[] = [];
  const eps = 1e-6;
  for (let i = 0; i < boxes.length; i++) {
    const a = boxes[i];
    const aMaxX = a.b.cx + a.b.sx / 2;
    for (let j = i + 1; j < boxes.length && boxes[j].b.cx - boxes[j].b.sx / 2 < aMaxX - eps; j++) {
      const c = boxes[j];
      if (a.id !== c.id && Math.abs(a.b.cz - c.b.cz) * 2 < a.b.sz + c.b.sz - eps) found.push(`${a.id} × ${c.id}`);
    }
  }
  return found;
}

describe.each(LAYOUT_CASES)('layout: $name', (c) => {
  const model = buildModel(c.cfg, c.options);
  const focusFloor = layerFloor(1);
  const layout = layoutTensors(model.tensors, { floorSpacing: 8000, thickness: 'bytes', focusFloor });

  it('places every tensor, with one entry per box', () => {
    expect(layout.placed).toHaveLength(model.tensors.length);
    for (const p of layout.placed) {
      expect(p.boxes.length).toBeGreaterThan(0);
      for (const list of [p.heads, p.rows, p.roles, p.experts]) expect(list).toHaveLength(p.boxes.length);
    }
  });

  it('gives finite boxes of positive size, as thick as their bytes', () => {
    for (const p of layout.placed) {
      for (const b of p.boxes) {
        for (const v of Object.values(b)) expect(Number.isFinite(v), p.spec.id).toBe(true);
        expect(b.sx, p.spec.id).toBeGreaterThan(0);
        expect(b.sz, p.spec.id).toBeGreaterThan(0);
        expect(b.sy, p.spec.id).toBeCloseTo(DTYPES[p.spec.dtype].bytes * THICKNESS_PER_BYTE, 9);
      }
    }
  });

  it('keeps a tensor\'s total footprint equal to its element count', () => {
    for (const p of layout.placed) {
      if (p.spec.x === 0 || p.spec.z === 0) continue;
      const area = p.boxes.reduce((a, b) => a + b.sx * b.sz, 0);
      expect(area / (p.spec.x * p.spec.z * p.spec.tiles), p.spec.id).toBeCloseTo(1, 6);
    }
  });

  it('never overlaps two tensors on a floor', () => {
    for (let f = 0; f < floorCount(c.cfg); f++) {
      expect(overlaps(layout.placed.filter((p) => p.spec.floor === f)), `floor ${f}`).toEqual([]);
    }
  });

  it('stacks floors upward, opening a gap around the focused floor', () => {
    const ys = Array.from({ length: floorCount(c.cfg) }, (_, f) => layout.floorY(f));
    for (let f = 1; f < ys.length; f++) expect(ys[f]).toBeGreaterThan(ys[f - 1]);
    expect(ys[focusFloor] - ys[focusFloor - 1]).toBeGreaterThan(8000);
    expect(ys[focusFloor + 1] - ys[focusFloor]).toBeGreaterThan(8000);
    expect(ys[focusFloor + 3] - ys[focusFloor + 2]).toBe(8000);
  });
});

describe('the overlap check', () => {
  const at = (id: string, cx: number, cz: number) =>
    ({ spec: { id }, boxes: [{ cx, cy: 0, cz, sx: 10, sy: 1, sz: 10 }] }) as unknown as PlacedTensor;

  it('finds overlapping tensors and passes touching ones', () => {
    expect(overlaps([at('a', 0, 0), at('b', 5, 5)])).toEqual(['a × b']);
    expect(overlaps([at('a', 0, 0), at('b', 10, 0), at('c', 0, 10)])).toEqual([]);
    expect(overlaps([at('a', 0, 0), at('b', 100, 0), at('c', 3, -4)])).toEqual(['a × c']);
  });
});

describe('thickness', () => {
  it('uniform mode draws every tensor at bf16 thickness', () => {
    const cfg = MODELS['llama-3-8b'];
    const model = buildModel(cfg, baseOptions(cfg, { weightDtype: 'int4', kvDtype: 'fp8' }));
    const layout = layoutTensors(model.tensors, { floorSpacing: 8000, thickness: 'elements', focusFloor: 1 });
    for (const p of layout.placed) for (const b of p.boxes) expect(b.sy).toBe(DTYPES.bf16.bytes * THICKNESS_PER_BYTE);
  });
});

describe('box helpers', () => {
  const boxes: Box[] = [
    { cx: 0, cy: 0, cz: 0, sx: 2, sy: 2, sz: 2 },
    { cx: 5, cy: 1, cz: -3, sx: 2, sy: 4, sz: 2 },
  ];

  it('bounds and encloses', () => {
    expect(boxBounds(boxes)).toEqual({ minX: -1, maxX: 6, minY: -1, maxY: 3, minZ: -4, maxZ: 1 });
    expect(enclosingBox(boxes)).toEqual({ cx: 2.5, cy: 1, cz: -1.5, sx: 7, sy: 4, sz: 5 });
  });
});
