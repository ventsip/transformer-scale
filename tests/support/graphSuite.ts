// Structural invariants of the built model, for every combination of settings: run per model from tests/model/graph/,
// so the models' sweeps run in parallel.
import { describe, expect, it } from 'vitest';
import type { ModelId } from '../../src/model/config';
import { floorCount, isLayerFloor, type Model, type TensorSpec } from '../../src/model/tensors';
import { hasViz } from '../../src/ui/opPopup';
import { allCases, local, modelOf } from './matrix';

/** Tensors no op produces: the token ids and positions the pass starts from, and the cache of the previous step. */
const isInput = (s: TensorSpec) => ['input_ids', 'position_ids'].includes(local(s.id)) || !!s.superseded;

export const graphSuite = (id: ModelId) => describe.each(allCases().filter((c) => c.id === id))('$name', (c) => {
  const m: Model = modelOf(c);
  const byId = new Map(m.tensors.map((s) => [s.id, s]));

  it('has unique tensor and op ids', () => {
    expect(byId.size).toBe(m.tensors.length);
    expect(new Set(m.ops.map((o) => o.id)).size).toBe(m.ops.length);
  });

  it('links every op to existing tensors, on its own floor or below', () => {
    for (const op of m.ops) {
      const out = byId.get(op.output);
      expect(out, `${op.label} → ${op.output}`).toBeDefined();
      for (const id of op.inputs) {
        const input = byId.get(id);
        expect(input, `${op.label} ← ${id}`).toBeDefined();
        // Data flows up: an op reads tensors of its floor or of floors below it.
        expect(input!.floor, `${op.label} ← ${id}`).toBeLessThanOrEqual(op.floor);
      }
      expect(op.inputs.length).toBeGreaterThan(0);
    }
  });

  it('produces every activation by exactly one op, buffers by at most one (computed tables), never a weight', () => {
    const producers = new Map<string, number>();
    for (const op of m.ops) producers.set(op.output, (producers.get(op.output) ?? 0) + 1);
    for (const s of m.tensors) {
      const n = producers.get(s.id) ?? 0;
      if (s.kind === 'weight') expect(n, s.id).toBe(0);
      else if (s.kind === 'buffer') expect(n, s.id).toBeLessThanOrEqual(1);
      else if (isInput(s)) expect(n, s.id).toBe(0);
      else expect(n, s.id).toBe(1);
    }
  });

  it('reads every tensor except the outputs: caches, the next token, and idle experts', () => {
    const read = new Set(m.ops.flatMap((o) => o.inputs));
    for (const s of m.tensors) {
      if (read.has(s.id) || s.kind === 'kv' || local(s.id) === 'next_token') continue;
      expect(s.idle, `${s.id} is never read`).toBe(true);
    }
    // And the converse: an idle expert's weights are really never read.
    for (const s of m.tensors.filter((t) => t.idle)) expect(read.has(s.id), s.id).toBe(false);
  });

  it('has shapes that agree with their dims, and sane sizes', () => {
    for (const s of m.tensors) {
      expect(s.dims.reduce((a, d) => a * d.size, 1), s.id).toBe(s.x * s.z * s.tiles);
      for (const v of [s.x, s.z, s.tiles]) expect(Number.isInteger(v) && v >= 0, `${s.id}: ${v}`).toBe(true);
      expect(s.floor).toBeGreaterThanOrEqual(0);
      expect(s.floor).toBeLessThan(floorCount(c.cfg));
      if (s.filledZ !== undefined) {
        expect(s.filledZ, s.id).toBeLessThanOrEqual(s.z);
        expect(s.freshZ ?? 0, s.id).toBeLessThanOrEqual(s.filledZ);
      }
      if (s.causal) expect(s.x, `${s.id} is square`).toBe(s.z);
      if (s.window !== undefined) {
        expect(s.causal).toBe(true);
        expect(s.window).toBeLessThan(s.x);
      }
      if (s.heads?.axis === 'x') expect(s.x % s.heads.count, s.id).toBe(0);
      if (s.heads?.axis === 'z') expect(s.z % s.heads.count, s.id).toBe(0);
      if (s.heads?.axis === 'tiles') expect(s.tiles, s.id).toBe(s.heads.count);
    }
  });

  it('keeps data flow inside each layer floor in station order', () => {
    // Within a layer's attention section, an op's output never sits in an earlier station than an input of the same
    // section. (The embedding floor puts position_ids after the RoPE tables that read it, on purpose.)
    for (const op of m.ops) {
      const out = byId.get(op.output)!;
      for (const id of op.inputs) {
        const input = byId.get(id)!;
        if (isLayerFloor(c.cfg, out.floor) && input.floor === out.floor && input.section === out.section && out.section === 0 && input.kind !== 'weight') {
          expect(out.station, `${op.label}: ${id} → ${op.output}`).toBeGreaterThanOrEqual(input.station);
        }
      }
    }
  });

  it('ends in one next-token id that the generation loop feeds back', () => {
    expect(local(m.loop.output)).toBe('next_token');
    expect(local(m.loop.input)).toBe('input_ids');
    expect(byId.get(m.loop.output)!.dtype).toBe('int64');
  });

  it('opens a popup for every op tagged with one', () => {
    for (const op of m.ops) expect(hasViz(op), op.label).toBe(!!op.viz);
  });

  it('gives ops finite, non-negative FLOPs and byte counts', () => {
    for (const op of m.ops) {
      if (op.flops !== undefined) expect(Number.isFinite(op.flops) && op.flops >= 0, op.label).toBe(true);
      for (const [id, bytes] of Object.entries(op.inputBytes ?? {})) {
        expect(op.inputs).toContain(id);
        expect(bytes).toBeGreaterThan(0);
      }
    }
  });
});
