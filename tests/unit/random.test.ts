import { describe, expect, it } from 'vitest';
import { gaussian, rng } from '../../src/random';

describe('rng', () => {
  it('is deterministic for a seed', () => {
    const a = rng(7);
    const b = rng(7);
    for (let i = 0; i < 100; i++) expect(a()).toBe(b());
  });

  it('differs between seeds', () => {
    expect(rng(1)()).not.toBe(rng(2)());
  });

  it('stays in [0, 1) and is roughly uniform', () => {
    const r = rng(123);
    const buckets = new Array(10).fill(0);
    const n = 50_000;
    for (let i = 0; i < n; i++) {
      const v = r();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
      buckets[Math.floor(v * 10)]++;
    }
    for (const b of buckets) expect(Math.abs(b - n / 10)).toBeLessThan(n / 100);
  });
});

describe('gaussian', () => {
  it('has mean ≈ 0 and standard deviation ≈ 1', () => {
    const normal = gaussian(rng(5));
    const n = 50_000;
    const xs = Array.from({ length: n }, normal);
    const mean = xs.reduce((a, b) => a + b, 0) / n;
    const variance = xs.reduce((a, x) => a + (x - mean) ** 2, 0) / n;
    expect(Math.abs(mean)).toBeLessThan(0.02);
    expect(Math.abs(Math.sqrt(variance) - 1)).toBeLessThan(0.02);
    expect(xs.every(Number.isFinite)).toBe(true);
  });
});
