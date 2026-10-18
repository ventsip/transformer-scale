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

/**
 * Layers are drawn independently and lazily: each layer's tokens are chosen in order (stickiness looks one token back)
 * only as far as a pass asks, so long contexts and many experts cost nothing until they are used.
 */
export function routing(cfg: ModelConfig): Routing {
  const cached = cache.get(cfg);
  if (cached) return cached;
  const { nExperts: E, topK: K } = cfg.moe!;
  const weights = new Float64Array(E);
  const layers = new Map<number, { rand: () => number; popularity: number[]; choice: Uint8Array; filled: number }>();

  /** Layer `l`'s choices for positions [0, to): choice[pos * K + j] is the j-th expert of that token. */
  function chosen(l: number, to: number): Uint8Array {
    let layer = layers.get(l);
    if (!layer) {
      const rand = rng(42 + l);
      const normal = gaussian(rand);
      const popularity = Array.from({ length: E }, () => Math.exp(IMBALANCE * normal()));
      layer = { rand, popularity, choice: new Uint8Array(0), filled: 0 };
      layers.set(l, layer);
    }
    if (to * K > layer.choice.length) {
      // Grow geometrically, up to the context length.
      const grown = new Uint8Array(Math.min(cfg.maxContext, Math.max(to, 2 * layer.choice.length / K)) * K);
      grown.set(layer.choice);
      layer.choice = grown;
    }
    const { rand, popularity, choice } = layer;
    for (let p = layer.filled; p < to; p++) {
      const at = p * K;
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
    layer.filled = Math.max(layer.filled, to);
    return choice;
  }

  const r: Routing = {
    experts: (l, p) => Array.from(chosen(l, p + 1).subarray(p * K, (p + 1) * K)),
    counts: (l, from, to) => {
      const choice = chosen(l, to);
      const n = new Array<number>(E).fill(0);
      for (let i = from * K; i < to * K; i++) n[choice[i]]++;
      return n;
    },
  };
  cache.set(cfg, r);
  return r;
}
