import type { ModelConfig } from './config';
import { DTYPES, type DType } from './dtypes';

/**
 * weight: parameters. activation: transient intermediates.
 * kv: activations that are written to the KV cache and persist.
 * scores: attention scores, never materialized by fused attention kernels.
 * buffer: fixed, non-learned constants stored with the model (not parameters).
 */
export type TensorKind = 'weight' | 'activation' | 'kv' | 'scores' | 'buffer';

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
  /** 0 = attention (front), 1 = MLP (behind it). */
  section: number;
  /** Position in data-flow order within the section. */
  station: number;
  /** Footprint of one tile, in elements: x along X, z along Z. */
  x: number;
  z: number;
  /** Identical tiles, e.g. one per attention head. */
  tiles: number;
  dims: Dim[];
  dtype: DType;
  note?: string;
  heads?: HeadSplit;
}

/**
 * How a tensor divides into attention heads: equal bands along X or Z, or one tile per head.
 * Query-indexed tensors have `perGroup` query heads per KV group; KV-indexed ones have 1.
 */
export interface HeadSplit {
  axis: 'x' | 'z' | 'tiles';
  count: number;
  perGroup: number;
}

/** A selected query head (and its KV group), or a whole KV group when `head` is null. */
export interface HeadSelection {
  group: number;
  head: number | null;
}

export const kvGroup = (split: HeadSplit, head: number) => Math.floor(head / split.perGroup);

export function selectionFor(split: HeadSplit, head: number): HeadSelection {
  return split.perGroup === 1 ? { group: head, head: null } : { group: kvGroup(split, head), head };
}

/** Whether band `head` of a tensor with `split` takes part in the selected head's computation. */
export function onPath(split: HeadSplit, head: number, sel: HeadSelection): boolean {
  if (split.perGroup === 1) return head === sel.group;
  return sel.head === null ? kvGroup(split, head) === sel.group : head === sel.head;
}

/** An operation: no parameters of its own, it combines input tensors into an output tensor. */
export interface OpSpec {
  id: string;
  label: string;
  floor: number;
  inputs: string[];
  output: string;
  note: string;
}

export interface Model {
  tensors: TensorSpec[];
  ops: OpSpec[];
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

export function buildModel(cfg: ModelConfig, o: BuildOptions): Model {
  const tensors: TensorSpec[] = [];
  const ops: OpSpec[] = [];
  const { T } = o;
  const tok: Dim = { name: 'T', size: T };
  const dModel: Dim = { name: 'd_model', size: cfg.dModel };
  const qDim: Dim = { name: `q_dim (${cfg.nHeads}×${cfg.headDim})`, size: cfg.nHeads * cfg.headDim };
  const kvDim: Dim = { name: `kv_dim (${cfg.nKvHeads}×${cfg.headDim})`, size: cfg.nKvHeads * cfg.headDim };
  const ffn: Dim = { name: 'd_ffn', size: cfg.dFfn };
  const vocab: Dim = { name: 'vocab', size: cfg.vocabSize };
  const gqaNote = `Grouped-query attention: ${cfg.nKvHeads} KV heads shared by ${cfg.nHeads} query heads.`;

  let floor = 0;
  let section = 0;
  let station = 0;
  let prefix = '';

  const push = (s: Omit<TensorSpec, 'floor' | 'section' | 'station' | 'tiles'> & { tiles?: number }) => {
    const id = `${floor}:${s.id}`;
    tensors.push({ tiles: 1, ...s, id, floor, section, station });
    return id;
  };

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
  const qHeads: HeadSplit = { axis: 'x', count: cfg.nHeads, perGroup: cfg.nHeads / cfg.nKvHeads };
  const kvHeads: HeadSplit = { axis: 'x', count: cfg.nKvHeads, perGroup: 1 };
  /** Marks a tensor as split into heads; `axis` overrides the split's default. */
  const split = (id: string, heads: HeadSplit, axis = heads.axis) => {
    tensors.find((t) => t.id === id)!.heads = { ...heads, axis };
    return id;
  };
  const op = (label: string, inputs: string[], output: string, note: string) =>
    ops.push({ id: `${floor}:op:${ops.length}`, label, floor, inputs, output, note });
  const matmul = (input: string, w: string, output: string, inDim: Dim, outDim: Dim, extra = '') =>
    op('@', [input, w], output,
      `Matrix multiply${extra}: [T × ${inDim.size}] @ [${inDim.size} × ${outDim.size}] → [T × ${outDim.size}]. ` +
      `${formatFlops(2 * T * inDim.size * outDim.size)} FLOPs.`);
  const next = () => station++;
  const nextSection = () => {
    section++;
    station = 0;
  };
  const nextFloorX = () => `${floor + 1}:x`;

  // Embedding floor: rotary tables first (they feed every layer), then the token embedding.
  prefix = 'model.';
  const half: Dim = { name: 'head_dim / 2', size: cfg.headDim / 2 };
  const headDim: Dim = { name: 'head_dim', size: cfg.headDim };
  const ropeNote = `Computed once per forward pass and shared by all ${cfg.nLayers} layers and all heads.`;
  const invFreq = push({
    id: 'rotary_emb.inv_freq', label: 'model.rotary_emb.inv_freq', kind: 'buffer', x: half.size, z: 1, dims: [half], dtype: 'fp32',
    note: `RoPE frequencies θ_i = ${cfg.ropeTheta.toLocaleString('en-US')}^(−2i/${cfg.headDim}): a fixed buffer, not a parameter. ` +
      `Wavelengths 2π/θ_i run from ~6 tokens to ~${formatTokens(2 * Math.PI * cfg.ropeTheta ** ((cfg.headDim - 2) / cfg.headDim))} tokens.`,
  });
  const pos = push({ id: 'position_ids', label: 'position_ids', kind: 'activation', x: 1, z: T, dims: [tok], dtype: 'int64', note: 'Position of each prompt token: 0 … T−1.' });
  next();
  const cos = act('cos', 'cos(pos · θ): RoPE table', headDim, 'activation', ropeNote);
  op('cos(pos·θ)', [pos, invFreq], cos, 'Outer product of positions and frequencies, then cosine; the 64 angles are repeated to fill head_dim.');
  next();
  const sin = act('sin', 'sin(pos · θ): RoPE table', headDim, 'activation', ropeNote);
  op('sin(pos·θ)', [pos, invFreq], sin, 'Outer product of positions and frequencies, then sine; the 64 angles are repeated to fill head_dim.');
  next();
  const embed = weight('embed_tokens.weight', vocab, dModel, 'Lookup table: each token id selects one row.');
  const ids = push({ id: 'input_ids', label: 'input_ids', kind: 'activation', x: 1, z: T, dims: [tok], dtype: 'int64', note: 'Token ids of the prompt.' });
  op('lookup', [ids, embed], nextFloorX(), 'Embedding lookup: copies one row per token. No arithmetic.');
  const ropeOp = (input: string, output: string, heads: number) =>
    op('RoPE', [input, cos, sin], output,
      `Rotary position embedding: in each of the ${heads} heads, rotates dimension pairs (i, i+${half.size}) by angle pos·θ_i. ` +
      'No parameters; after it, Q·Kᵀ depends on the distance between positions.');

  for (let l = 0; l < cfg.nLayers; l++) {
    floor = l + 1;
    section = 0;
    station = 0;
    prefix = `model.layers.${l}.`;

    // Attention section.
    const x = act('x', 'x: residual stream in', dModel, 'activation', `${TRANSIENT} Embeddings for layer 0, the previous layer's output otherwise.`);
    next();
    const n1 = vector('input_layernorm.weight', dModel);
    const xNorm = act('x_norm', 'x_norm = RMSNorm(x)', dModel);
    op('RMSNorm', [x, n1], xNorm, 'Normalize each token vector by its RMS, then scale by the norm weight.');
    next();
    const wq = split(weight('self_attn.q_proj.weight', dModel, qDim), qHeads, 'z');
    const qRaw = split(act('q_raw', 'x_norm · W_q (before RoPE)', qDim), qHeads);
    matmul(xNorm, wq, qRaw, dModel, qDim);
    next();
    const q = split(act('q', 'Q = RoPE(x_norm · W_q)', qDim), qHeads);
    ropeOp(qRaw, q, cfg.nHeads);
    next();
    const wk = split(weight('self_attn.k_proj.weight', dModel, kvDim, gqaNote), kvHeads, 'z');
    const kRaw = split(act('k_raw', 'x_norm · W_k (before RoPE)', kvDim), kvHeads);
    matmul(xNorm, wk, kRaw, dModel, kvDim);
    next();
    const k = split(act('k', 'K = RoPE(x_norm · W_k)', kvDim, 'kv', `${KV} Cached after rotation, so the cache already encodes position.`), kvHeads);
    ropeOp(kRaw, k, cfg.nKvHeads);
    next();
    const wv = split(weight('self_attn.v_proj.weight', dModel, kvDim, gqaNote), kvHeads, 'z');
    const v = split(act('v', 'V = x_norm · W_v', kvDim, 'kv'), kvHeads);
    matmul(xNorm, wv, v, dModel, kvDim);
    next();
    // QKᵀ and ·V are 2 FLOPs per multiply-add each; the causal mask lets fused kernels skip key > query.
    const attnFlops = 4 * T * T * cfg.nHeads * cfg.headDim;
    const causalFlops = (attnFlops * (T + 1)) / (2 * T);
    let attn: string;
    if (o.showScores) {
      const scores = split(push({
        id: 'scores', label: 'softmax(QKᵀ/√d): one T×T tile per head', kind: 'scores', x: T, z: T, tiles: cfg.nHeads,
        dims: [{ name: 'heads', size: cfg.nHeads }, tok, tok], dtype: o.actDtype,
        note: 'Never materialized by FlashAttention-style kernels; shown for scale. Eager attention allocates every tile in full, masked half included.',
      }), qHeads, 'tiles');
      op('softmax(QKᵀ)', [q, k], scores, 'Per head: softmax(Q Kᵀ / √d + mask). The causal mask sets entries with key > query to −∞, so they become 0.');
      next();
      attn = split(act('attn', 'attn = softmax(QKᵀ/√d) · V', qDim), qHeads);
      op('· V', [scores, v], attn, 'Per head: weighted sum of V rows.');
    } else {
      attn = split(act('attn', 'attn = softmax(QKᵀ/√d) · V', qDim), qHeads);
      op('attention', [q, k, v], attn,
        `softmax(QKᵀ/√d)·V per head, fused (FlashAttention): the T×T scores never reach memory. Causal: query i only attends to keys 0..i, so ~${formatFlops(causalFlops)} FLOPs instead of ${formatFlops(attnFlops)}.`);
    }
    next();
    const wo = split(weight('self_attn.o_proj.weight', qDim, dModel), qHeads);
    const oOut = act('o', 'o = attn · W_o', dModel);
    matmul(attn, wo, oOut, qDim, dModel);
    next();
    const h = act('h', 'h = x + o: residual stream', dModel);
    op('+', [x, oOut], h, 'Residual add: the attention output is added to the residual stream.');

    // MLP section, behind the attention section.
    nextSection();
    const n2 = vector('post_attention_layernorm.weight', dModel);
    const hNorm = act('h_norm', 'h_norm = RMSNorm(h)', dModel);
    op('RMSNorm', [h, n2], hNorm, 'Normalize each token vector by its RMS, then scale by the norm weight.');
    next();
    const wg = weight('mlp.gate_proj.weight', dModel, ffn);
    const g = act('g', 'g = h_norm · W_gate', ffn);
    matmul(hNorm, wg, g, dModel, ffn);
    next();
    const wu = weight('mlp.up_proj.weight', dModel, ffn);
    const u = act('u', 'u = h_norm · W_up', ffn);
    matmul(hNorm, wu, u, dModel, ffn);
    next();
    const a = act('a', 'a = SiLU(g) ⊙ u', ffn);
    op('SiLU ⊙', [g, u], a, 'SwiGLU: elementwise SiLU(g) times u.');
    next();
    const wd = weight('mlp.down_proj.weight', ffn, dModel);
    const m = act('m', 'm = a · W_down', dModel);
    matmul(a, wd, m, ffn, dModel);
    op('+', [m, h], nextFloorX(), "Residual add: h + m is this layer's output and the next floor's input.");
  }

  // LM head floor.
  floor = cfg.nLayers + 1;
  section = 0;
  station = 0;
  prefix = 'model.';
  const xLast = act('x', 'x: residual stream out of the last layer', dModel);
  next();
  const nf = vector('norm.weight', dModel);
  const xNorm = act('x_norm', 'x_norm = RMSNorm(x)', dModel);
  op('RMSNorm', [xLast, nf], xNorm, 'Final norm.');
  next();
  prefix = '';
  const head = weight('lm_head.weight', dModel, vocab, 'Not tied to the embedding in Llama 3: a second 525M-parameter matrix.');
  const logits = push({
    id: 'logits', label: 'logits: last position only', kind: 'activation', x: cfg.vocabSize, z: 1,
    dims: [{ name: 'last token', size: 1 }, vocab], dtype: o.actDtype,
    note: 'Only the last position is needed to sample the next token.',
  });
  op('@', [xNorm, head], logits,
    `Matrix multiply, last position only: [1 × ${cfg.dModel}] @ [${cfg.dModel} × ${cfg.vocabSize}]. ${formatFlops(2 * cfg.dModel * cfg.vocabSize)} FLOPs.`);

  return { tensors, ops };
}

function formatTokens(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toPrecision(2)}M` : n >= 1e3 ? `${(n / 1e3).toPrecision(2)}K` : n.toFixed(0);
}

function formatFlops(n: number): string {
  const units: [number, string][] = [[1e15, 'P'], [1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'K']];
  for (const [scale, unit] of units) if (n >= scale) return `${(n / scale).toPrecision(3)} ${unit}`;
  return String(n);
}
