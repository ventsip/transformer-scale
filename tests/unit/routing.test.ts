import { describe, expect, it } from 'vitest';
import { DEEPSEEK_V2_LITE, MIXTRAL_8X7B, type ModelConfig } from '../../src/model/config';
import { routing } from '../../src/model/routing';

describe.each([MIXTRAL_8X7B, DEEPSEEK_V2_LITE] as ModelConfig[])('routing ($name)', (cfg) => {
  const { nExperts: E, topK: K } = cfg.moe!;
  const r = routing(cfg);

  it('picks top-k distinct experts per token', () => {
    for (let p = 0; p < 500; p++) {
      const e = r.experts(3, p);
      expect(e).toHaveLength(K);
      expect(new Set(e).size).toBe(K);
      for (const x of e) {
        expect(Number.isInteger(x)).toBe(true);
        expect(x).toBeGreaterThanOrEqual(0);
        expect(x).toBeLessThan(E);
      }
    }
  });

  it('counts k picks per token, and counts agree with the picks', () => {
    const counts = r.counts(2, 0, 400);
    expect(counts).toHaveLength(E);
    expect(counts.reduce((a, b) => a + b, 0)).toBe(400 * K);
    const fromPicks = new Array(E).fill(0);
    for (let p = 0; p < 400; p++) for (const e of r.experts(2, p)) fromPicks[e]++;
    expect(fromPicks).toEqual(counts);
  });

  it('adds up over adjacent ranges', () => {
    const a = r.counts(5, 0, 123);
    const b = r.counts(5, 123, 777);
    const whole = r.counts(5, 0, 777);
    expect(a.map((v, i) => v + b[i])).toEqual(whole);
  });

  it('is deterministic and independent of the order positions are asked for', () => {
    const late = r.experts(7, 3000);
    const fresh = routing({ ...cfg }); // a new config object: a new, uncached routing
    expect(fresh.experts(7, 3000)).toEqual(late);
    expect(fresh.counts(7, 0, 50)).toEqual(r.counts(7, 0, 50));
  });

  it('differs between layers', () => {
    expect(r.counts(0, 0, 300)).not.toEqual(r.counts(1, 0, 300));
  });

  it('is only mildly imbalanced: every expert gets some of a long prompt', () => {
    const counts = r.counts(4, 0, 4000);
    const mean = (4000 * K) / E;
    for (const c of counts) {
      expect(c).toBeGreaterThan(mean * 0.4);
      expect(c).toBeLessThan(mean * 2);
    }
  });
});
