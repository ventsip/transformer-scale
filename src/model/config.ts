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
  /** Width of the dense MLP (for a mixture of experts, see moe.expertFfn). */
  dFfn: number;
  maxContext: number;
  /** RoPE base: frequencies are ropeTheta^(−2i/d), d the rotated dims (see ropeDim). Only with rotary position embedding (llama). */
  ropeTheta?: number;
  /** ε of the norms (RMSNorm or LayerNorm). */
  normEps: number;
  /** The beginning-of-text token the tokenizer puts first. */
  bos: { text: string; id: number };
  /** Sparse mixture of experts in place of the dense MLP. */
  moe?: MoeConfig;
  /** Multi-head latent attention (DeepSeek-V2) in place of separate K and V projections. */
  mla?: MlaConfig;
  /** RoPE context extension: YaRN, as DeepSeek-V2 uses it. */
  yarn?: YarnConfig;
  /** RoPE rotates neighboring dims (2i, 2i+1) together, instead of HF rotate_half's i and i + d/2 (DeepSeek-V2). */
  ropeInterleaved?: boolean;
}

/** Each token runs through `topK` of `nExperts` experts (and, if any, the shared experts). */
export interface MoeConfig {
  nExperts: number;
  topK: number;
  /** Width of each routed expert (moe_intermediate_size). */
  expertFfn: number;
  /** Width of the shared experts, run on every token as one MLP (n_shared_experts × moe_intermediate_size). */
  sharedFfn?: number;
  /** The first decoder layers keep a dense MLP of width dFfn (first_k_dense_replace). */
  denseLayers: number;
  /** Whether the top-k routing weights are divided by their sum (norm_topk_prob; Mixtral always does). */
  renormalize: boolean;
  /** Experts per row of the grid they are drawn in. */
  gridColumns: number;
  /** HF module names: the MoE block, and each expert's gate, up and down projections. */
  names: { block: string; gate: string; up: string; down: string };
}

/**
 * Multi-head latent attention: one projection compresses x into a latent c_kv of `kvLoraRank` (normalized), plus one
 * `qkRopeDim` key shared by all heads that alone carries RoPE; a second projection expands c_kv into each head's K
 * (`qkNopeDim`, no position) and V (`vDim`). Queries are `qkNopeDim + qkRopeDim` per head; headDim is that sum.
 */
export interface MlaConfig {
  kvLoraRank: number;
  qkNopeDim: number;
  qkRopeDim: number;
  vDim: number;
}

/**
 * YaRN: RoPE trained at `originalContext`, stretched `factor`×. Slow frequencies (wavelengths beyond the original
 * context) are divided by the factor, fast ones (from `betaFast` turns over the original context) kept, with a linear
 * ramp between; attention scores are scaled up to compensate (mscale).
 */
export interface YarnConfig {
  factor: number;
  originalContext: number;
  betaFast: number;
  betaSlow: number;
  mscale: number;
  mscaleAllDim: number;
}

/** Dimensions of each head that RoPE rotates: all of them, or MLA's decoupled RoPE part. */
export const ropeDim = (cfg: ModelConfig) => cfg.mla?.qkRopeDim ?? cfg.headDim;
/** Width of each head's V: MLA's may differ from Q and K's. */
export const vHeadDim = (cfg: ModelConfig) => cfg.mla?.vDim ?? cfg.headDim;
/** Whether decoder layer `layer` routes through experts (or keeps a dense MLP). */
export const isMoeLayer = (cfg: ModelConfig, layer: number) => !!cfg.moe && layer >= cfg.moe.denseLayers;

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
  moe: {
    nExperts: 8, topK: 2, expertFfn: 14336, denseLayers: 0, renormalize: true, gridColumns: 4,
    names: { block: 'block_sparse_moe', gate: 'w1', up: 'w3', down: 'w2' },
  },
};

// https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json: multi-head latent attention (no query
// compression: q_lora_rank null), 64 routed experts (top 6, weights not renormalized) plus 2 shared ones, a dense first
// layer, and YaRN-extended RoPE (4K × 40 = 160K).
export const DEEPSEEK_V2_LITE: ModelConfig = {
  name: 'DeepSeek-V2-Lite',
  arch: 'llama',
  vocabSize: 102400,
  dModel: 2048,
  nLayers: 27,
  nHeads: 16,
  nKvHeads: 16,
  headDim: 192,
  dFfn: 10944,
  maxContext: 163840,
  ropeTheta: 10000,
  normEps: 1e-6,
  bos: { text: '<｜begin▁of▁sentence｜>', id: 100000 },
  moe: {
    nExperts: 64, topK: 6, expertFfn: 1408, sharedFfn: 2 * 1408, denseLayers: 1, renormalize: false, gridColumns: 8,
    names: { block: 'mlp', gate: 'gate_proj', up: 'up_proj', down: 'down_proj' },
  },
  mla: { kvLoraRank: 512, qkNopeDim: 128, qkRopeDim: 64, vDim: 128 },
  ropeInterleaved: true,
  yarn: { factor: 40, originalContext: 4096, betaFast: 32, betaSlow: 1, mscale: 0.707, mscaleAllDim: 0.707 },
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

export type ModelId = 'gpt2' | 'gpt2-xl' | 'llama-3-8b' | 'llama-3-70b' | 'mixtral-8x7b' | 'deepseek-v2-lite';

export const MODELS: Record<ModelId, ModelConfig> = {
  'gpt2': GPT2,
  'gpt2-xl': GPT2_XL,
  'llama-3-8b': LLAMA_3_8B,
  'llama-3-70b': LLAMA_3_70B,
  'mixtral-8x7b': MIXTRAL_8X7B,
  'deepseek-v2-lite': DEEPSEEK_V2_LITE,
};
