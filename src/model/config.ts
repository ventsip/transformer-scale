export interface ModelConfig {
  name: string;
  vocabSize: number;
  dModel: number;
  nLayers: number;
  nHeads: number;
  nKvHeads: number;
  headDim: number;
  /** Width of the MLP; for a mixture of experts, of each expert. */
  dFfn: number;
  maxContext: number;
  /** RoPE base: frequencies are ropeTheta^(−2i/headDim). */
  ropeTheta: number;
  rmsNormEps: number;
  /** Sparse mixture of experts in place of the dense MLP: each token runs through `topK` of `nExperts` experts. */
  moe?: { nExperts: number; topK: number };
}

// https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
export const LLAMA_3_8B: ModelConfig = {
  name: 'Llama-3-8B',
  vocabSize: 128256,
  dModel: 4096,
  nLayers: 32,
  nHeads: 32,
  nKvHeads: 8,
  headDim: 128,
  dFfn: 14336,
  maxContext: 8192,
  ropeTheta: 500000,
  rmsNormEps: 1e-5,
};

// https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
export const MIXTRAL_8X7B: ModelConfig = {
  name: 'Mixtral-8x7B',
  vocabSize: 32000,
  dModel: 4096,
  nLayers: 32,
  nHeads: 32,
  nKvHeads: 8,
  headDim: 128,
  dFfn: 14336,
  maxContext: 32768,
  ropeTheta: 1e6,
  rmsNormEps: 1e-5,
  moe: { nExperts: 8, topK: 2 },
};

export type ModelId = 'llama-3-8b' | 'mixtral-8x7b';

export const MODELS: Record<ModelId, ModelConfig> = {
  'llama-3-8b': LLAMA_3_8B,
  'mixtral-8x7b': MIXTRAL_8X7B,
};
