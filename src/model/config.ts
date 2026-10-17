export interface ModelConfig {
  name: string;
  vocabSize: number;
  dModel: number;
  nLayers: number;
  nHeads: number;
  nKvHeads: number;
  headDim: number;
  dFfn: number;
  maxContext: number;
  /** RoPE base: frequencies are ropeTheta^(−2i/headDim). */
  ropeTheta: number;
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
};
