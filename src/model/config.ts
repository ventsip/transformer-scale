export interface ModelConfig {
  name: string;
  /**
   * llama: RoPE, RMSNorm, SwiGLU MLP, separate Q/K/V projections, no biases, untied LM head (Llama 3, Mixtral).
   * gpt2: learned position embeddings, LayerNorm, GELU MLP, one fused Q/K/V projection, biases, LM head tied to the
   * token embedding (GPT-2).
   */
  arch: 'llama' | 'gpt2';
  vocabSize: number;
  dModel: number;
  nLayers: number;
  nHeads: number;
  nKvHeads: number;
  headDim: number;
  /** Width of the MLP; for a mixture of experts, of each expert. */
  dFfn: number;
  maxContext: number;
  /** RoPE base: frequencies are ropeTheta^(−2i/headDim). Only with rotary position embedding (llama). */
  ropeTheta?: number;
  /** ε of the norms (RMSNorm or LayerNorm). */
  normEps: number;
  /** The beginning-of-text token the tokenizer puts first. */
  bos: { text: string; id: number };
  /** Sparse mixture of experts in place of the dense MLP: each token runs through `topK` of `nExperts` experts. */
  moe?: { nExperts: number; topK: number };
}

// https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
export const LLAMA_3_8B: ModelConfig = {
  name: 'Llama-3-8B',
  arch: 'llama',
  vocabSize: 128256,
  dModel: 4096,
  nLayers: 32,
  nHeads: 32,
  nKvHeads: 8,
  headDim: 128,
  dFfn: 14336,
  maxContext: 8192,
  ropeTheta: 500000,
  normEps: 1e-5,
  bos: { text: '<|begin_of_text|>', id: 128000 },
};

// https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json (gated; checked against the public mirror
// NousResearch/Meta-Llama-3-70B). Same architecture as 8B: twice the width, 2.5× the layers, the same 8 KV heads.
export const LLAMA_3_70B: ModelConfig = {
  name: 'Llama-3-70B',
  arch: 'llama',
  vocabSize: 128256,
  dModel: 8192,
  nLayers: 80,
  nHeads: 64,
  nKvHeads: 8,
  headDim: 128,
  dFfn: 28672,
  maxContext: 8192,
  ropeTheta: 500000,
  normEps: 1e-5,
  bos: { text: '<|begin_of_text|>', id: 128000 },
};

// https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
export const MIXTRAL_8X7B: ModelConfig = {
  name: 'Mixtral-8x7B',
  arch: 'llama',
  vocabSize: 32000,
  dModel: 4096,
  nLayers: 32,
  nHeads: 32,
  nKvHeads: 8,
  headDim: 128,
  dFfn: 14336,
  maxContext: 32768,
  ropeTheta: 1e6,
  normEps: 1e-5,
  bos: { text: '<s>', id: 1 },
  moe: { nExperts: 8, topK: 2 },
};

// https://huggingface.co/openai-community/gpt2/blob/main/config.json: multi-head attention (as many KV heads as query
// heads), d_ffn = 4 · d_model (n_inner unset).
export const GPT2: ModelConfig = {
  name: 'GPT-2',
  arch: 'gpt2',
  vocabSize: 50257,
  dModel: 768,
  nLayers: 12,
  nHeads: 12,
  nKvHeads: 12,
  headDim: 64,
  dFfn: 3072,
  maxContext: 1024,
  normEps: 1e-5,
  bos: { text: '<|endoftext|>', id: 50256 },
};

// https://huggingface.co/openai-community/gpt2-xl/blob/main/config.json
export const GPT2_XL: ModelConfig = {
  ...GPT2,
  name: 'GPT-2 XL',
  dModel: 1600,
  nLayers: 48,
  nHeads: 25,
  nKvHeads: 25,
  dFfn: 6400,
};

export type ModelId = 'gpt2' | 'gpt2-xl' | 'llama-3-8b' | 'llama-3-70b' | 'mixtral-8x7b';

export const MODELS: Record<ModelId, ModelConfig> = {
  'gpt2': GPT2,
  'gpt2-xl': GPT2_XL,
  'llama-3-8b': LLAMA_3_8B,
  'llama-3-70b': LLAMA_3_70B,
  'mixtral-8x7b': MIXTRAL_8X7B,
};
