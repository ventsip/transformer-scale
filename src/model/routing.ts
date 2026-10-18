import { gaussian, rng } from '../random';
import type { ModelConfig } from './config';

/**
 * Illustrative expert routing for a mixture-of-experts model. Real routing depends on trained router weights
 * and the actual text, so this is seeded and deterministic, with two properties real routers show:
 * - mild load imbalance: each layer favors some experts a little;
 * - temporal locality: consecutive tokens pick the same first expert more often than chance
 *   (the Mixtral paper, section 5).
 */
export interface Routing {
  /** Experts chosen for the token at `pos` in decoder layer `layer`, highest routing weight first. */
  experts(layer: number, pos: number): number[];
  /** Tokens at positions [from, to) that each expert receives in `layer`. */
  counts(layer: number, from: number, to: number): number[];
}

/** Spread of the per-layer expert popularity (log scale). */
const IMBALANCE = 0.2;
/** Chance that a token's first expert repeats the previous token's. */
const STICKINESS = 0.25;

const cache = new Map<ModelConfig, Routing>();

export function routing(cfg: ModelConfig): Routing {
  const cached = cache.get(cfg);
  if (cached) return cached;
  const { nExperts: E, topK: K } = cfg.moe!;
  const P = cfg.maxContext;
  const rand = rng(42);
  const normal = gaussian(rand);
  // choice[(layer * P + pos) * K + j]: the j-th expert of that token.
  const choice = new Uint8Array(cfg.nLayers * P * K);
  const weights = new Float64Array(E);
  for (let l = 0; l < cfg.nLayers; l++) {
    const popularity = Array.from({ length: E }, () => Math.exp(IMBALANCE * normal()));
    for (let p = 0; p < P; p++) {
      const at = (l * P + p) * K;
      weights.set(popularity);
      for (let j = 0; j < K; j++) {
        let e: number;
        if (j === 0 && p > 0 && rand() < STICKINESS) {
          e = choice[at - K];
        } else {
          // Weighted draw among the experts not yet chosen for this token.
          let r = rand() * weights.reduce((a, b) => a + b, 0);
          e = 0;
          while (e < E - 1 && (r >= weights[e] || weights[e] === 0)) r -= weights[e++];
          // Rounding can run past the last expert still available.
          if (weights[e] === 0) e = weights.findIndex((w) => w > 0);
        }
        choice[at + j] = e;
        weights[e] = 0;
      }
    }
  }
  const r: Routing = {
    experts: (l, p) => Array.from(choice.subarray((l * P + p) * K, (l * P + p + 1) * K)),
    counts: (l, from, to) => {
      const n = new Array<number>(E).fill(0);
      for (let i = (l * P + from) * K; i < (l * P + to) * K; i++) n[choice[i]]++;
      return n;
    },
  };
  cache.set(cfg, r);
  return r;
}
