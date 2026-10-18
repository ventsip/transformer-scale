import { describe, expect, it } from 'vitest';
import { formatBytes, formatCount, formatFlops, formatInt, formatPercent } from '../../src/format';

describe('formatCount', () => {
  it.each([
    [0, '0'],
    [9_999, '9,999'],
    [10_000, '10.0K'],
    [124_439_808, '124.4M'],
    [8_030_261_248, '8.03B'],
  ])('%d → %s', (n, s) => expect(formatCount(n)).toBe(s));
});

describe('formatBytes', () => {
  it.each([
    [0, '0 B'],
    [1023, '1,023 B'],
    [1024, '1.00 KiB'],
    [1536, '1.50 KiB'],
    // 1,000–1,023 of a unit stay in that unit instead of printing 1.00e+3.
    [1000 * 1024, '1,000 KiB'],
    [2 ** 30, '1.00 GiB'],
    [2 ** 50, '1,024 TiB'],
  ])('%d → %s', (n, s) => expect(formatBytes(n)).toBe(s));
});

describe('formatFlops', () => {
  it.each([
    [999, '999'],
    [1e3, '1.00 K'],
    [2.98e12, '2.98 T'],
    [1.5e15, '1.50 P'],
  ])('%d → %s', (n, s) => expect(formatFlops(n)).toBe(s));
});

describe('formatInt and formatPercent', () => {
  it('groups thousands', () => expect(formatInt(1234567)).toBe('1,234,567'));
  it('gives significant digits', () => {
    expect(formatPercent(0.1543)).toBe('15.4%');
    expect(formatPercent(1)).toBe('100%');
    expect(formatPercent(0.5, 1)).toBe('5e+1%');
  });
});
