import type { ModelConfig } from './config';
import { DTYPES, type DType } from './dtypes';

/**
 * weight: parameters. activation: transient intermediates.
 * kv: activations that are written to the KV cache and persist.
 * scores: attention scores, never materialized by fused attention kernels.
 */
export type TensorKind = 'weight' | 'activation' | 'kv' | 'scores';

export interface Dim {
  name: string;
  size: number;
}

export interface TensorSpec {
  id: string;
  label: string;
  kind: TensorKind;
  /** 0 = embedding, 1..nLayers = decoder layers, nLayers + 1 = LM head. */
  floor: number;
  /** Position in data-flow order on the floor. */
  station: number;
  /** Footprint of one tile, in elements: x along X, z along Z. */
  x: number;
  z: number;
  /** Identical tiles, e.g. one per attention head. */
  tiles: number;
  dims: Dim[];
  dtype: DType;
  note?: string;
}

export interface BuildOptions {
  T: number;
  weightDtype: DType;
  actDtype: DType;
  kvDtype: DType;
  showScores: boolean;
}

const TRANSIENT = 'Transient: during inference it only exists while this layer runs.';
const KV = 'Written to the KV cache: persists for the whole generation.';

export const floorCount = (cfg: ModelConfig) => cfg.nLayers + 2;

export function floorName(cfg: ModelConfig, floor: number): string {
  if (floor === 0) return 'Embedding';
  if (floor === cfg.nLayers + 1) return 'LM head';
  return `Layer ${floor - 1}`;
}

export const elementCount = (s: TensorSpec) => s.x * s.z * s.tiles;
export const byteCount = (s: TensorSpec) => elementCount(s) * DTYPES[s.dtype].bytes;
export const isTransient = (s: TensorSpec) => s.kind === 'activation' || s.kind === 'scores';

export function buildTensors(cfg: ModelConfig, o: BuildOptions): TensorSpec[] {
  const specs: TensorSpec[] = [];
  const { T } = o;
  const tok: Dim = { name: 'T', size: T };
  const dModel: Dim = { name: 'd_model', size: cfg.dModel };
  const qDim: Dim = { name: `q_dim (${cfg.nHeads}×${cfg.headDim})`, size: cfg.nHeads * cfg.headDim };
  const kvDim: Dim = { name: `kv_dim (${cfg.nKvHeads}×${cfg.headDim})`, size: cfg.nKvHeads * cfg.headDim };
  const ffn: Dim = { name: 'd_ffn', size: cfg.dFfn };
  const vocab: Dim = { name: 'vocab', size: cfg.vocabSize };
  const gqaNote = `Grouped-query attention: ${cfg.nKvHeads} KV heads shared by ${cfg.nHeads} query heads.`;

  let floor = 0;
  let station = 0;
  let prefix = '';

  const push = (s: Omit<TensorSpec, 'floor' | 'station' | 'tiles'> & { tiles?: number }) =>
    specs.push({ tiles: 1, ...s, id: `${floor}:${s.id}`, floor, station });

  /** Weights lie with their input dim along X and output dim along Z. */
  const weight = (name: string, inDim: Dim, outDim: Dim, note?: string) =>
    push({ id: name, label: prefix + name, kind: 'weight', x: inDim.size, z: outDim.size, dims: [inDim, outDim], dtype: o.weightDtype, note });
  const vector = (name: string, dim: Dim, note?: string) =>
    push({ id: name, label: prefix + name, kind: 'weight', x: dim.size, z: 1, dims: [dim], dtype: o.weightDtype, note });
  /** Activations lie with their feature dim along X and tokens along Z. */
  const act = (id: string, label: string, feat: Dim, kind: 'activation' | 'kv' = 'activation', note?: string) =>
    push({
      id, label, kind, x: feat.size, z: T, dims: [tok, feat],
      dtype: kind === 'kv' ? o.kvDtype : o.actDtype,
      note: note ?? (kind === 'kv' ? KV : TRANSIENT),
    });
  const next = () => station++;

  // Embedding floor.
  prefix = 'model.';
  weight('embed_tokens.weight', vocab, dModel, 'Lookup table: each token id selects one row.');
  push({ id: 'input_ids', label: 'input_ids', kind: 'activation', x: 1, z: T, dims: [tok], dtype: 'int32', note: 'Token ids of the prompt.' });

  for (let l = 0; l < cfg.nLayers; l++) {
    floor = l + 1;
    station = 0;
    prefix = `model.layers.${l}.`;
    act('x', 'x: residual stream in', dModel, 'activation', `${TRANSIENT} Embeddings for layer 0, the previous layer's output otherwise.`);
    next();
    vector('input_layernorm.weight', dModel);
    act('x_norm', 'x_norm = RMSNorm(x)', dModel);
    next();
    weight('self_attn.q_proj.weight', dModel, qDim);
    act('q', 'Q = x_norm · W_q (+RoPE)', qDim);
    next();
    weight('self_attn.k_proj.weight', dModel, kvDim, gqaNote);
    act('k', 'K = x_norm · W_k (+RoPE)', kvDim, 'kv');
    next();
    weight('self_attn.v_proj.weight', dModel, kvDim, gqaNote);
    act('v', 'V = x_norm · W_v', kvDim, 'kv');
    next();
    if (o.showScores) {
      push({
        id: 'scores', label: 'softmax(QKᵀ/√d): one T×T tile per head', kind: 'scores', x: T, z: T, tiles: cfg.nHeads,
        dims: [{ name: 'heads', size: cfg.nHeads }, tok, tok], dtype: o.actDtype,
        note: 'Never materialized by FlashAttention-style kernels; shown for scale. The causal mask zeroes the upper half.',
      });
      next();
    }
    act('attn', 'attn = softmax(QKᵀ/√d) · V', qDim);
    next();
    weight('self_attn.o_proj.weight', qDim, dModel);
    act('o', 'o = attn · W_o', dModel);
    next();
    act('h', 'h = x + o: residual stream', dModel);
    next();
    vector('post_attention_layernorm.weight', dModel);
    act('h_norm', 'h_norm = RMSNorm(h)', dModel);
    next();
    weight('mlp.gate_proj.weight', dModel, ffn);
    act('g', 'g = h_norm · W_gate', ffn);
    next();
    weight('mlp.up_proj.weight', dModel, ffn);
    act('u', 'u = h_norm · W_up', ffn);
    next();
    act('a', 'a = SiLU(g) ⊙ u', ffn);
    next();
    weight('mlp.down_proj.weight', ffn, dModel);
    act('m', 'm = a · W_down (layer output is h + m)', dModel);
  }

  // LM head floor.
  floor = cfg.nLayers + 1;
  station = 0;
  prefix = 'model.';
  act('x', 'x: residual stream out of the last layer', dModel);
  next();
  vector('norm.weight', dModel);
  act('x_norm', 'x_norm = RMSNorm(x)', dModel);
  next();
  prefix = '';
  weight('lm_head.weight', dModel, vocab, 'Not tied to the embedding in Llama 3: a second 525M-parameter matrix.');
  push({
    id: 'logits', label: 'logits: last position only', kind: 'activation', x: cfg.vocabSize, z: 1,
    dims: [{ name: 'last token', size: 1 }, vocab], dtype: o.actDtype,
    note: 'Only the last position is needed to sample the next token.',
  });

  return specs;
}
