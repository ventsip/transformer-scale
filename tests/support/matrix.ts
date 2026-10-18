import { MODELS, type ModelConfig, type ModelId } from '../../src/model/config';
import { buildModel, type BuildOptions, type Model } from '../../src/model/tensors';

export const MODEL_IDS = Object.keys(MODELS) as ModelId[];

/** One point of the settings matrix, with a readable name for test titles. */
export interface Case {
  name: string;
  id: ModelId;
  cfg: ModelConfig;
  options: BuildOptions;
}

/**
 * Prompt / cache lengths: a single token, a short prompt, and one past Gemma's 512-token window, so sliding layers are
 * exercised both inside and beyond it.
 */
export const LENGTHS = [1, 300, 700];

export const baseOptions = (cfg: ModelConfig, overrides: Partial<BuildOptions> = {}): BuildOptions => ({
  mode: 'prefill', T: 300, weightDtype: 'bf16', actDtype: 'bf16', kvDtype: 'bf16', attention: 'fused',
  kvCache: 'dynamic', elementwise: 'compact', cacheCapacity: cfg.maxContext, mlaCache: 'hf', ...overrides,
});

/** Every combination of the settings the builder branches on, for every model. */
export function allCases(lengths = LENGTHS): Case[] {
  const cases: Case[] = [];
  for (const id of MODEL_IDS) {
    const cfg = MODELS[id];
    for (const mode of ['prefill', 'decode'] as const)
      for (const attention of ['fused', 'eager'] as const)
        for (const kvCache of ['dynamic', 'static'] as const)
          for (const elementwise of ['compact', 'expanded'] as const)
            for (const mlaCache of cfg.mla ? (['hf', 'latent'] as const) : (['hf'] as const))
              for (const T of lengths) {
                const options = baseOptions(cfg, { mode, T, attention, kvCache, elementwise, mlaCache });
                const name = `${id} ${mode} T=${T} ${attention} ${kvCache} ${elementwise}${cfg.mla ? ` ${mlaCache}` : ''}`;
                cases.push({ name, id, cfg, options });
              }
  }
  return cases;
}

const built = new Map<Case, Model>();
/** The model of a case, built once per test file. */
export function modelOf(c: Case): Model {
  let m = built.get(c);
  if (!m) built.set(c, (m = buildModel(c.cfg, c.options)));
  return m;
}

/** A tensor id without its floor prefix: `3:k_cache` → `k_cache`. */
export const local = (id: string) => id.replace(/^\d+:/, '');
