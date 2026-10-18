import { describe, expect, it } from 'vitest';
import { DEEPSEEK_V2_LITE, GEMMA_3_1B, GPT2, LLAMA_3_8B } from '../../src/model/config';
import {
  EMBEDDING_FLOOR, cacheParts, floorCount, floorLayer, floorName, isLayerFloor, keptEntries, layerFloor, lmHeadFloor, onPath,
  samplingFloor, selectionFor,
} from '../../src/model/tensors';

describe('keptEntries', () => {
  /** Counts query × key pairs a causal (optionally windowed) mask keeps, one by one. */
  const brute = (T: number, w = Infinity) => {
    let n = 0;
    for (let q = 0; q < T; q++) for (let k = 0; k <= q; k++) if (q - k < w) n++;
    return n;
  };

  it('matches a brute-force count', () => {
    for (const T of [1, 2, 7, 33, 100]) {
      expect(keptEntries(T)).toBe(brute(T));
      for (const w of [1, 2, 5, 32, 99, 100, 500]) expect(keptEntries(T, w)).toBe(brute(T, w));
    }
  });
});

describe('cacheParts', () => {
  it('K and V per KV head', () => {
    expect(cacheParts(LLAMA_3_8B, { attention: 'fused' })).toEqual([{ name: 'K', width: 1024 }, { name: 'V', width: 1024 }]);
    expect(cacheParts(GPT2, { attention: 'eager' })).toEqual([{ name: 'K', width: 768 }, { name: 'V', width: 768 }]);
    expect(cacheParts(GEMMA_3_1B, { attention: 'fused' })).toEqual([{ name: 'K', width: 256 }, { name: 'V', width: 256 }]);
  });

  it('MLA: K and V expanded as HF caches them, V padded under FlashAttention', () => {
    expect(cacheParts(DEEPSEEK_V2_LITE, { attention: 'eager', mlaCache: 'hf' }))
      .toEqual([{ name: 'K', width: 3072 }, { name: 'V', width: 2048, padding: 0 }]);
    expect(cacheParts(DEEPSEEK_V2_LITE, { attention: 'fused', mlaCache: 'hf' }))
      .toEqual([{ name: 'K', width: 3072 }, { name: 'V', width: 3072, padding: 1024 }]);
  });

  it('MLA latent cache: 512 + 64 values', () => {
    expect(cacheParts(DEEPSEEK_V2_LITE, { attention: 'fused', mlaCache: 'latent' }))
      .toEqual([{ name: 'c_kv', width: 512 }, { name: 'k_pe', width: 64 }]);
  });
});

describe('floors', () => {
  it('numbers embedding, layers, LM head and sampling', () => {
    expect(EMBEDDING_FLOOR).toBe(0);
    expect(layerFloor(0)).toBe(1);
    expect(floorLayer(layerFloor(31))).toBe(31);
    expect(lmHeadFloor(LLAMA_3_8B)).toBe(33);
    expect(samplingFloor(LLAMA_3_8B)).toBe(34);
    expect(floorCount(LLAMA_3_8B)).toBe(35);
    expect(isLayerFloor(LLAMA_3_8B, 0)).toBe(false);
    expect(isLayerFloor(LLAMA_3_8B, 1)).toBe(true);
    expect(isLayerFloor(LLAMA_3_8B, 32)).toBe(true);
    expect(isLayerFloor(LLAMA_3_8B, 33)).toBe(false);
    expect(floorName(LLAMA_3_8B, 0)).toBe('Embedding');
    expect(floorName(LLAMA_3_8B, 5)).toBe('Layer 4');
    expect(floorName(LLAMA_3_8B, 33)).toBe('LM head');
    expect(floorName(LLAMA_3_8B, 34)).toBe('Sampling');
  });
});

describe('head selection', () => {
  const q = { axis: 'x' as const, count: 32, perGroup: 4 };
  const kv = { axis: 'x' as const, count: 8, perGroup: 1 };

  it('a query head selects itself and its KV group', () => {
    const sel = selectionFor(q, 13);
    expect(sel).toEqual({ group: 3, head: 13 });
    expect(onPath(q, 13, sel)).toBe(true);
    expect(onPath(q, 12, sel)).toBe(false);
    expect(onPath(kv, 3, sel)).toBe(true);
    expect(onPath(kv, 2, sel)).toBe(false);
  });

  it('a KV head selects its whole group of query heads', () => {
    const sel = selectionFor(kv, 2);
    expect(sel).toEqual({ group: 2, head: null });
    expect([8, 9, 10, 11].every((h) => onPath(q, h, sel))).toBe(true);
    expect(onPath(q, 12, sel)).toBe(false);
  });
});
