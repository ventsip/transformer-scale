import { ropeDim, type ModelConfig } from './config';

/**
 * RoPE's frequencies θ_i, one per pair of rotated dims: ropeTheta^(−2i/d). With YaRN (DeepSeek-V2's
 * yarn_find_correction_range and linear_ramp_mask), pairs that turn fewer than `betaSlow` times over the original
 * context are slowed by the factor, pairs turning more than `betaFast` times are kept, and the ones between blend
 * linearly. `base`: another RoPE base than the model's (Gemma's sliding layers).
 */
export function ropeFrequencies(cfg: ModelConfig, base = cfg.ropeTheta!): number[] {
  const d = ropeDim(cfg);
  const plain = Array.from({ length: d / 2 }, (_, i) => base ** (-(2 * i) / d));
  const y = cfg.yarn;
  if (!y) return plain;
  // The pair index whose wavelength fits `turns` times into the original context.
  const pairFor = (turns: number) => (d * Math.log(y.originalContext / (turns * 2 * Math.PI))) / (2 * Math.log(base));
  const low = Math.max(Math.floor(pairFor(y.betaFast)), 0);
  let high = Math.min(Math.ceil(pairFor(y.betaSlow)), d - 1);
  if (low === high) high += 0.001;
  return plain.map((f, i) => {
    // 0: keep the frequency; 1: divide it by the factor.
    const ramp = Math.min(1, Math.max(0, (i - low) / (high - low)));
    return f * (1 - ramp) + (f / y.factor) * ramp;
  });
}

/** YaRN's attention temperature: scores are multiplied by mscale², 0.1 · m · ln(factor) + 1 (1 without YaRN). */
export function yarnScoreScale(cfg: ModelConfig): number {
  const y = cfg.yarn;
  if (!y) return 1;
  const mscale = 0.1 * y.mscaleAllDim * Math.log(y.factor) + 1;
  return mscale ** 2;
}
