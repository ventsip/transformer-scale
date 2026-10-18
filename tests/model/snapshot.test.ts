// Golden snapshots of what the builder produces: totals for every model and setting, and the full tensor and op
// listing of a few floors. Any change to the model shows up here as a diff to review; accept intended ones with
// `npm run test:update`.
import { describe, expect, it } from 'vitest';
import { MODELS } from '../../src/model/config';
import { summarize } from '../../src/model/stats';
import { buildModel, lmHeadFloor, type Model } from '../../src/model/tensors';
import { MODEL_IDS, allCases, baseOptions, local, modelOf } from '../support/matrix';

const round = (v: number) => Math.round(v * 1000) / 1000;

/** One line per tensor and op of `floor`: what it is, its shape and dtype, and what it reads. */
function listing(m: Model, floor: number): string[] {
  const tensors = m.tensors.filter((s) => s.floor === floor).map((s) =>
    `T ${local(s.id)} ${s.kind} ${s.x}×${s.z}${s.tiles > 1 ? `×${s.tiles}` : ''} ${s.dtype} s${s.section}.${s.station}` +
    `${s.causal ? ' causal' : ''}${s.window ? ` window=${s.window}` : ''}${s.filledZ !== undefined ? ` filled=${s.filledZ}` : ''}`);
  const ops = m.ops.filter((o) => o.floor === floor).map((o) =>
    `O ${o.label}: ${o.inputs.map(local).join(', ')} → ${local(o.output)}${o.flops ? ` ${o.flops}` : ''}${o.viz ? ` [${JSON.stringify(o.viz)}]` : ''}`);
  return [...tensors, ...ops];
}

describe('totals', () => {
  it.each(MODEL_IDS)('%s', (id) => {
    const totals = Object.fromEntries(allCases([1, 700]).filter((c) => c.id === id).map((c) => {
      const s = summarize(modelOf(c), c.cfg);
      return [c.name, Object.fromEntries(Object.entries(s).map(([k, v]) => [k, round(v)]))];
    }));
    expect(totals).toMatchSnapshot();
  });
});

describe('floors', () => {
  it.each(MODEL_IDS)('%s', (id) => {
    const cfg = MODELS[id];
    const prefill = buildModel(cfg, baseOptions(cfg, { T: 700, attention: 'eager', elementwise: 'expanded' }));
    const decode = buildModel(cfg, baseOptions(cfg, { mode: 'decode', T: 700, kvCache: 'static', cacheCapacity: 4096 }));
    // The embedding, the first two layers (Gemma: sliding; DeepSeek: dense, then experts), a global Gemma layer, the LM
    // head and sampling.
    const floors = [0, 1, 2, ...(cfg.slidingWindow ? [6] : []), lmHeadFloor(cfg), lmHeadFloor(cfg) + 1];
    expect({
      prefill: Object.fromEntries(floors.map((f) => [f, listing(prefill, f)])),
      decode: Object.fromEntries(floors.map((f) => [f, listing(decode, f)])),
    }).toMatchSnapshot();
  });
});
