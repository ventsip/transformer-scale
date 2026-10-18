import { describe, expect, it } from 'vitest';
import { DEEPSEEK_V2_LITE, GEMMA_3_1B, LLAMA_3_8B } from '../../src/model/config';
import { ropeFrequencies, yarnScoreScale } from '../../src/model/rope';

describe('ropeFrequencies', () => {
  it('follows θ_i = base^(−2i/d) without YaRN', () => {
    const f = ropeFrequencies(LLAMA_3_8B);
    expect(f).toHaveLength(64);
    expect(f[0]).toBe(1);
    f.forEach((v, i) => expect(v).toBeCloseTo(500000 ** (-(2 * i) / 128), 12));
  });

  it('decreases strictly', () => {
    for (const cfg of [LLAMA_3_8B, DEEPSEEK_V2_LITE, GEMMA_3_1B]) {
      const f = ropeFrequencies(cfg);
      for (let i = 1; i < f.length; i++) expect(f[i]).toBeLessThan(f[i - 1]);
    }
  });

  it('takes another base (Gemma sliding layers)', () => {
    const local = ropeFrequencies(GEMMA_3_1B, 10000);
    const global = ropeFrequencies(GEMMA_3_1B);
    expect(local).toHaveLength(128);
    expect(local.at(-1)).toBeCloseTo(10000 ** (-254 / 256), 12);
    // A smaller base turns every pair but the first faster.
    for (let i = 1; i < local.length; i++) expect(local[i]).toBeGreaterThan(global[i]);
  });

  it('applies YaRN: fast pairs kept, slow pairs divided by the factor, a ramp between', () => {
    const cfg = DEEPSEEK_V2_LITE;
    const { factor } = cfg.yarn!;
    const f = ropeFrequencies(cfg);
    const d = 64;
    const plain = f.map((_, i) => 10000 ** (-(2 * i) / d));
    expect(f).toHaveLength(32);
    expect(f[0]).toBeCloseTo(plain[0], 12);
    expect(f.at(-1)!).toBeCloseTo(plain.at(-1)! / factor, 12);
    // The share of the slowdown applied never decreases from fast to slow pairs.
    const ramp = f.map((v, i) => (plain[i] / v - 1) / (factor - 1));
    for (let i = 1; i < ramp.length; i++) expect(ramp[i]).toBeGreaterThanOrEqual(ramp[i - 1] - 1e-12);
    for (const r of ramp) {
      expect(r).toBeGreaterThanOrEqual(-1e-12);
      expect(r).toBeLessThanOrEqual(1 + 1e-12);
    }
  });
});

describe('yarnScoreScale', () => {
  it('is 1 without YaRN', () => expect(yarnScoreScale(LLAMA_3_8B)).toBe(1));
  it('is mscale² with YaRN (DeepSeek-V2: ≈ 1.59)', () => {
    expect(yarnScoreScale(DEEPSEEK_V2_LITE)).toBeCloseTo((0.1 * 0.707 * Math.log(40) + 1) ** 2, 12);
    expect(yarnScoreScale(DEEPSEEK_V2_LITE)).toBeCloseTo(1.59, 2);
  });
});
