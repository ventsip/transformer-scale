/**
 * Seeded pseudo-random numbers, for the illustrative data (routing, popup values): the same every time for a seed.
 * A linear congruential generator (Numerical Recipes' constants), uniform in [0, 1).
 */
export function rng(seed: number): () => number {
  return () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
}

/** Standard normal samples from a uniform generator (Box–Muller). */
export function gaussian(rand: () => number): () => number {
  return () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
}
