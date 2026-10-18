export function formatCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(1)}K`;
  return n.toLocaleString('en-US');
}

const UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];

export function formatBytes(n: number): string {
  let i = 0;
  while (n >= 1024 && i < UNITS.length - 1) {
    n /= 1024;
    i++;
  }
  // 1,000–1,023 of a unit stay in that unit: toPrecision(3) would print them as 1.00e+3.
  return `${i === 0 || n >= 1000 ? Math.round(n).toLocaleString('en-US') : n.toPrecision(3)} ${UNITS[i]}`;
}

export const formatInt = (n: number) => n.toLocaleString('en-US');

export function formatFlops(n: number): string {
  const units: [number, string][] = [[1e15, 'P'], [1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'K']];
  for (const [scale, unit] of units) if (n >= scale) return `${(n / scale).toPrecision(3)} ${unit}`;
  return String(n);
}
