import { describe, expect, it } from 'vitest';
import {
  DEEPSEEK_V2_LITE, GEMMA_3_1B, LLAMA_3_8B, MIXTRAL_8X7B, MODELS, isMoeLayer, isSlidingLayer, layerWindow, ropeDim, vHeadDim,
} from '../../src/model/config';
import { MODEL_IDS } from '../support/matrix';

describe('model configs', () => {
  it.each(MODEL_IDS)('%s is self-consistent', (id) => {
    const cfg = MODELS[id];
    for (const v of [cfg.vocabSize, cfg.dModel, cfg.nLayers, cfg.nHeads, cfg.nKvHeads, cfg.headDim, cfg.dFfn, cfg.maxContext]) {
      expect(Number.isInteger(v) && v > 0).toBe(true);
    }
    expect(cfg.nHeads % cfg.nKvHeads).toBe(0);
    expect(cfg.bos.id).toBeLessThan(cfg.vocabSize);
    expect(cfg.normEps).toBeGreaterThan(0);
    if (cfg.arch === 'llama') expect(cfg.ropeTheta).toBeGreaterThan(0);
    if (cfg.moe) {
      expect(cfg.moe.topK).toBeLessThanOrEqual(cfg.moe.nExperts);
      expect(cfg.moe.nExperts % cfg.moe.gridColumns).toBe(0);
      // Expert ids are stored as bytes in the routing tables.
      expect(cfg.moe.nExperts).toBeLessThanOrEqual(256);
    }
    if (cfg.mla) expect(cfg.mla.qkNopeDim + cfg.mla.qkRopeDim).toBe(cfg.headDim);
    expect(ropeDim(cfg) % 2).toBe(0);
  });

  it('has unique display names', () => {
    const names = MODEL_IDS.map((id) => MODELS[id].name);
    expect(new Set(names).size).toBe(names.length);
  });
});

describe('layer helpers', () => {
  it('Gemma 3: every sixth layer is global, the rest see 512 keys', () => {
    const global = Array.from({ length: GEMMA_3_1B.nLayers }, (_, l) => l).filter((l) => !isSlidingLayer(GEMMA_3_1B, l));
    expect(global).toEqual([5, 11, 17, 23]);
    expect(layerWindow(GEMMA_3_1B, 0)).toBe(512);
    expect(layerWindow(GEMMA_3_1B, 5)).toBe(Infinity);
  });

  it('models without a sliding window are global everywhere', () => {
    for (let l = 0; l < LLAMA_3_8B.nLayers; l++) {
      expect(isSlidingLayer(LLAMA_3_8B, l)).toBe(false);
      expect(layerWindow(LLAMA_3_8B, l)).toBe(Infinity);
    }
  });

  it('MLA rotates only its RoPE dims, and V is narrower', () => {
    expect(ropeDim(DEEPSEEK_V2_LITE)).toBe(64);
    expect(vHeadDim(DEEPSEEK_V2_LITE)).toBe(128);
    expect(ropeDim(LLAMA_3_8B)).toBe(128);
    expect(vHeadDim(LLAMA_3_8B)).toBe(128);
  });

  it('dense first layers of a mixture of experts', () => {
    expect(isMoeLayer(DEEPSEEK_V2_LITE, 0)).toBe(false);
    expect(isMoeLayer(DEEPSEEK_V2_LITE, 1)).toBe(true);
    expect(isMoeLayer(MIXTRAL_8X7B, 0)).toBe(true);
    expect(isMoeLayer(LLAMA_3_8B, 0)).toBe(false);
  });
});
