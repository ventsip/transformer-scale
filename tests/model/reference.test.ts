// The builder's totals against numbers worked out independently: from the published architectures (HF modeling code
// and config.json), not from the builder.
import { describe, expect, it } from 'vitest';
import { MODELS, isMoeLayer, isSlidingLayer, type ModelConfig, type ModelId } from '../../src/model/config';
import { DTYPES } from '../../src/model/dtypes';
import { summarize } from '../../src/model/stats';
import { buildModel, type BuildOptions } from '../../src/model/tensors';
import { MODEL_IDS, baseOptions } from '../support/matrix';

/** Parameter count from the architecture, layer by layer, as HF's modules define them. */
function expectedParams(cfg: ModelConfig): number {
  const { vocabSize: V, dModel: d, nLayers: L, nHeads: H, nKvHeads: KV, headDim: hd, dFfn: ffn } = cfg;
  if (cfg.arch === 'gpt2') {
    const layer = 2 * d + (d * 3 * d + 3 * d) + (d * d + d) + 2 * d + (d * ffn + ffn) + (ffn * d + d);
    return V * d + cfg.maxContext * d + L * layer + 2 * d;
  }
  let total = V * d * (cfg.tiedEmbedding ? 1 : 2) + d;
  for (let l = 0; l < L; l++) {
    let attn: number;
    if (cfg.mla) {
      const { kvLoraRank: r, qkRopeDim: rope, qkNopeDim: nope, vDim } = cfg.mla;
      attn = d * H * hd + d * (r + rope) + r + r * H * (nope + vDim) + H * vDim * d;
    } else {
      attn = d * H * hd + 2 * d * KV * hd + H * hd * d + (cfg.qkNorm ? 2 * hd : 0);
    }
    const norms = (cfg.postNorms ? 4 : 2) * d;
    let mlp = 3 * d * ffn;
    if (isMoeLayer(cfg, l)) {
      const m = cfg.moe!;
      mlp = d * m.nExperts + m.nExperts * 3 * d * m.expertFfn + (m.sharedFfn ? 3 * d * m.sharedFfn : 0);
    }
    total += attn + norms + mlp;
  }
  return total;
}

/** Totals published by the model authors or computed by HF (model.num_parameters()). */
const PUBLISHED: Record<ModelId, number> = {
  'gpt2': 124_439_808,
  'gpt2-xl': 1_557_611_200,
  'llama-3-8b': 8_030_261_248,
  'llama-3-70b': 70_553_706_496,
  'mixtral-8x7b': 46_702_792_704,
  'deepseek-v2-lite': 15_706_484_224,
  'gemma-3-1b': 999_885_952,
};

describe('parameters', () => {
  it.each(MODEL_IDS)('%s matches the architecture and the published total', (id) => {
    const cfg = MODELS[id];
    const { params } = summarize(buildModel(cfg, baseOptions(cfg)), cfg);
    expect(params).toBe(expectedParams(cfg));
    expect(params).toBe(PUBLISHED[id]);
  });

  it.each(MODEL_IDS)('%s: the parameter count does not depend on the pass', (id) => {
    const cfg = MODELS[id];
    const counts = new Set(
      [baseOptions(cfg), baseOptions(cfg, { mode: 'decode', attention: 'eager', kvCache: 'static', elementwise: 'expanded', T: 700 })]
        .map((o) => summarize(buildModel(cfg, o), cfg).params),
    );
    expect(counts.size).toBe(1);
  });

  it('counts active parameters of a mixture of experts (Mixtral: 12.9B of 46.7B)', () => {
    const cfg = MODELS['mixtral-8x7b'];
    const { activeParams } = summarize(buildModel(cfg, baseOptions(cfg)), cfg);
    expect(activeParams / 1e9).toBeCloseTo(12.88, 2);
  });

  it('weight bytes scale with the weight dtype', () => {
    const cfg = MODELS['llama-3-8b'];
    for (const dtype of ['fp32', 'bf16', 'fp8', 'int4'] as const) {
      const s = summarize(buildModel(cfg, baseOptions(cfg, { weightDtype: dtype })), cfg);
      expect(s.paramBytes).toBe(s.params * DTYPES[dtype].bytes);
    }
  });
});

/** Rows each layer's cache holds after the pass, as HF's cache classes keep them. */
function cacheRows(cfg: ModelConfig, o: BuildOptions, layer: number): number {
  const decode = o.mode === 'decode';
  const keys = o.T + (decode ? 1 : 0);
  const sliding = isSlidingLayer(cfg, layer);
  const W = cfg.slidingWindow?.size ?? Infinity;
  if (o.kvCache === 'static') return sliding ? Math.min(W, o.cacheCapacity) : o.cacheCapacity;
  // Dynamic: a sliding layer keeps the window (511 cached + the new row) in decode; in prefill its view holds all of K.
  if (sliding && decode) return Math.min(o.T, W - 1) + 1;
  return keys;
}

/** Values one layer's cache holds per token. */
function cacheWidth(cfg: ModelConfig, o: BuildOptions): number {
  if (!cfg.mla) return 2 * cfg.nKvHeads * cfg.headDim;
  if (o.mlaCache === 'latent') return cfg.mla.kvLoraRank + cfg.mla.qkRopeDim;
  const v = o.attention === 'fused' ? cfg.headDim : cfg.mla.vDim;
  return cfg.nHeads * (cfg.headDim + v);
}

describe('KV cache', () => {
  const settings: Partial<BuildOptions>[] = [];
  for (const mode of ['prefill', 'decode'] as const)
    for (const kvCache of ['dynamic', 'static'] as const)
      for (const attention of ['fused', 'eager'] as const)
        for (const T of [1, 300, 700, 3000]) settings.push({ mode, kvCache, attention, T, kvDtype: 'fp8' });

  for (const id of MODEL_IDS) {
    const cfg = MODELS[id];
    for (const mlaCache of cfg.mla ? (['hf', 'latent'] as const) : (['hf'] as const)) {
      it.each(settings.filter((s) => s.T! < cfg.maxContext))(`${id}${cfg.mla ? ` ${mlaCache}` : ''}: %o`, (s) => {
        const o = baseOptions(cfg, { ...s, mlaCache, cacheCapacity: 8192 });
        const stats = summarize(buildModel(cfg, o), cfg);
        let values = 0;
        for (let l = 0; l < cfg.nLayers; l++) values += cacheRows(cfg, o, l) * cacheWidth(cfg, o);
        expect(stats.kvElements).toBe(values);
        expect(stats.kvBytes).toBe(values * DTYPES.fp8.bytes);
      });
    }
  }

  it('published per-token sizes (bf16)', () => {
    const perToken = (id: ModelId, o: Partial<BuildOptions> = {}) => {
      const cfg = MODELS[id];
      const s = summarize(buildModel(cfg, baseOptions(cfg, { T: 100, ...o })), cfg);
      return s.kvBytes / 100 / 1024;
    };
    expect(perToken('llama-3-8b')).toBe(128);
    expect(perToken('llama-3-70b')).toBe(320);
    expect(perToken('deepseek-v2-lite', { attention: 'eager' })).toBe(270);
    expect(perToken('deepseek-v2-lite', { attention: 'fused' })).toBe(324);
    expect(perToken('deepseek-v2-lite', { mlaCache: 'latent' })).toBeCloseTo(30.375, 6);
    expect(perToken('gemma-3-1b')).toBe(26);
  });
});

describe('FLOPs', () => {
  it.each(MODEL_IDS)('%s: a decode step costs about 2 FLOPs per active weight it multiplies by', (id) => {
    const cfg = MODELS[id];
    const s = summarize(buildModel(cfg, baseOptions(cfg, { mode: 'decode', T: 100 })), cfg);
    // The embedding lookup does no arithmetic; a tied table is multiplied once, by the LM head.
    const lookupOnly = cfg.arch === 'gpt2' ? cfg.maxContext * cfg.dModel : cfg.tiedEmbedding ? 0 : cfg.vocabSize * cfg.dModel;
    const ratio = s.flops / (2 * (s.activeParams - lookupOnly));
    // Norms and biases add no matmul FLOPs; attention over 101 keys adds a little.
    expect(ratio).toBeGreaterThan(0.99);
    expect(ratio).toBeLessThan(1.05);
  });

  it.each(MODEL_IDS)('%s: grows with the prompt, and eager attention never does less work than fused', (id) => {
    const cfg = MODELS[id];
    const flops = (o: Partial<BuildOptions>) => summarize(buildModel(cfg, baseOptions(cfg, o)), cfg).flops;
    expect(flops({ T: 600 })).toBeGreaterThan(flops({ T: 300 }));
    expect(flops({ T: 600, attention: 'eager' })).toBeGreaterThanOrEqual(flops({ T: 600 }));
  });

  it('Gemma: sliding-window layers make decode cost flat past the window', () => {
    const cfg = MODELS['gemma-3-1b'];
    const attention = (T: number) => summarize(buildModel(cfg, baseOptions(cfg, { mode: 'decode', T })), cfg).flops;
    const llama = MODELS['llama-3-8b'];
    const llamaFlops = (T: number) => summarize(buildModel(llama, baseOptions(llama, { mode: 'decode', T })), llama).flops;
    // Per extra cached token, only the 4 global layers do more work: 4 layers × 4 heads × (QK + PV) × 256 × 2.
    expect(attention(5000) - attention(4000)).toBe(1000 * 4 * 4 * 2 * 256 * 2);
    expect(llamaFlops(5000) - llamaFlops(4000)).toBe(1000 * 32 * 32 * 2 * 128 * 2);
  });
});
