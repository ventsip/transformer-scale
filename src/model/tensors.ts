import { isMoeLayer, isSlidingLayer, layerWindow, ropeDim, vHeadDim, type ModelConfig } from './config';
import { DTYPES, type DType } from './dtypes';
import { formatCount, formatFlops, formatInt } from '../format';
import { ropeFrequencies, yarnScoreScale } from './rope';
import { routing } from './routing';

/** Which attention projection a weight (or a part of a fused one) is: queries, keys or values. */
export type Role = 'q' | 'k' | 'v';

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
  /** 0 = embedding, 1..nLayers = decoder layers, nLayers + 1 = LM head, nLayers + 2 = sampling. */
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
  /** Square query × key tiles where the causal mask applies: the half above the diagonal is masked. */
  causal?: boolean;
  /**
   * Causal tiles of a sliding-window layer: a query sees only the last `window` keys, itself included, so the corner
   * below the diagonal band is masked too.
   */
  window?: number;
  /** For preallocated buffers: rows along Z actually written; the rest is reserved. */
  filledZ?: number;
  /** Of the written rows, how many (at the end) were written by this pass: highlighted. */
  freshZ?: number;
  /** A tensor this pass replaces (a dynamic cache before torch.cat): drawn faded, left out of the KV totals. */
  superseded?: boolean;
  /** Space before this tensor when it is stacked behind another in the same row (default: a small gap). */
  rowGap?: number;
  /** Place to the right of the previous tensor in its row instead of in front of it. */
  beside?: boolean;
  /** When stacked in its row, center it on the station instead of aligning it left. */
  centered?: boolean;
  /** Part of the residual stream: the edges that carry it are drawn bold. */
  residual?: boolean;
  /** Attention role of a projection weight: W_q, W_k or W_v. */
  role?: Role;
  /** A fused Q/K/V projection weight or bias (GPT-2's c_attn): the axis of its output dim, which holds Q, K and V in
   * equal thirds (Z for the weight, X for the bias). */
  fusedRoles?: 'x' | 'z';
  /** Mixture of experts: a router tensor with one slice per expert along this axis (gate.weight's output dim, the
   * columns of the router logits and probabilities). */
  expertAxis?: 'x' | 'z';
  /** Mixture of experts: the expert this tensor belongs to. */
  expert?: number;
  /** Mixture of experts: part of the shared experts, which every token runs through. */
  shared?: boolean;
  /** Weights of an expert no token is routed to in this pass: never read, drawn dimmed. */
  idle?: boolean;
  /** Extra space before this tensor's station, e.g. between experts. */
  gapBefore?: number;
  /**
   * Room kept for this station whatever its activations, so the layout does not depend on them:
   * depth in front of the bus, and width.
   */
  reserveFront?: number;
  reserveWidth?: number;
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

/**
 * What one layer's KV cache holds for each token: K and V (MLA: as HF caches them, V padded to the Q and K head width
 * under FlashAttention), or MLA's latent and RoPE key. Values, not bytes.
 */
export function cacheParts(cfg: ModelConfig, o: Pick<BuildOptions, 'attention' | 'mlaCache'>): { name: string; width: number; padding?: number }[] {
  const mla = cfg.mla;
  if (mla && o.mlaCache === 'latent') return [{ name: 'c_kv', width: mla.kvLoraRank }, { name: 'k_pe', width: mla.qkRopeDim }];
  const k = { name: 'K', width: cfg.nKvHeads * cfg.headDim };
  if (!mla) return [k, { name: 'V', width: cfg.nKvHeads * cfg.headDim }];
  const padding = o.attention === 'fused' ? cfg.nHeads * (cfg.headDim - mla.vDim) : 0;
  return [k, { name: 'V', width: cfg.nHeads * mla.vDim + padding, padding }];
}

/** Query × key entries a causal mask keeps in a T × T tile: all of key ≤ query, or only the last `window` keys of each query. */
export function keptEntries(T: number, window = Infinity): number {
  if (window >= T) return (T * (T + 1)) / 2;
  return (window * (window + 1)) / 2 + (T - window) * window;
}

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
  /** Bytes actually read from an input, when that is less than the whole tensor (by input id). */
  inputBytes?: Record<string, number>;
  /** The part of a fused Q/K/V input an op reads (by input id): its pipe attaches there. */
  inputRoles?: Record<string, Role>;
  /** Floating-point operations, for the ops that do meaningful arithmetic (matmuls, attention). */
  flops?: number;
  /** The interactive popup this op opens (see ui/opPopup.ts), if any. */
  viz?: OpViz;
}

/** A popup panel and the stage of it an op is focused on, in the panel's pipeline order. */
export type OpViz =
  | { panel: 'lookup' }
  | { panel: 'sampling'; stage: 'logits' | 'temperature' | 'filter' | 'softmax' | 'sample' }
  | { panel: 'moe'; stage: 'router' | 'dispatch' | 'combine' }
  | { panel: 'attention'; stage: 'scores' | 'softmax' | 'output'; window?: number }
  | { panel: 'rope'; stage: 'freq' | 'rotate'; theta?: number; window?: number }
  | { panel: 'rmsnorm'; stage: 'input' | 'mean' | 'normalize' | 'weight'; width?: number }
  | { panel: 'layernorm' }
  | { panel: 'gelu' }
  | { panel: 'swiglu'; stage: 'silu' | 'gate'; width: number; gelu?: boolean }
  | { panel: 'mla'; stage: 'compress' | 'expand' | 'assemble' | 'pad' }
  | { panel: 'kvcache'; stage: 'dynamic' | 'static' };
export type VizStage<P extends OpViz['panel']> = Extract<OpViz, { panel: P; stage: unknown }>['stage'];

/** Extra fields of an op, given when it is created. */
type OpExtra = Pick<OpSpec, 'flops' | 'viz' | 'inputBytes' | 'inputRoles'>;
type MatmulOptions = Omit<OpExtra, 'flops'> & { rows?: number; bias?: string; describe?: (note: string) => string };
type ActOptions = Omit<Partial<TensorSpec>, 'kind'> & { kind?: 'activation' | 'kv'; rows?: Dim };

export interface Model {
  tensors: TensorSpec[];
  ops: OpSpec[];
  /** The generation loop: the token ids the pass reads, and the next token it produces (fed back in decoding). */
  loop: { input: string; output: string };
  /** Mixture of experts: tokens routed to each expert in this pass, per decoder layer. */
  expertTokens?: number[][];
}

/** fused: FlashAttention / SDPA kernels. eager: HF eager_attention_forward, one tensor per step. */
export type AttentionImpl = 'fused' | 'eager';

/** dynamic: grows with the tokens (HF default). static: preallocated for a fixed capacity. */
export type KvCacheImpl = 'dynamic' | 'static';

/** compact: RMSNorm and SwiGLU as single ops. expanded: every intermediate tensor HF allocates. */
export type ElementwiseDetail = 'compact' | 'expanded';

/** prefill: one pass over a T-token prompt. decode: one generation step, 1 new token against L cached ones. */
export type Mode = 'prefill' | 'decode';

export interface BuildOptions {
  mode: Mode;
  /** Prefill: prompt length. Decode: tokens already in the cache (L). */
  T: number;
  weightDtype: DType;
  actDtype: DType;
  kvDtype: DType;
  attention: AttentionImpl;
  kvCache: KvCacheImpl;
  elementwise: ElementwiseDetail;
  /** Slots per layer in a static cache. */
  cacheCapacity: number;
  /** Multi-head latent attention: what the KV cache holds (default hf). */
  mlaCache?: MlaCache;
}

/** hf: K and V, expanded from the latent (as HF caches them). latent: the latent and RoPE key (vLLM, SGLang). */
export type MlaCache = 'hf' | 'latent';

const TRANSIENT = 'Transient: during inference it only exists while this layer runs.';
const KV = 'Written to the KV cache: persists for the whole generation.';
/** Gap between activations stacked in one station, leaving room for the op between them. */
const STACK_GAP = 2048;
/** The MLA popup's stages of the ops that build K and V and pad V. */
const ASSEMBLE: OpExtra = { viz: { panel: 'mla', stage: 'assemble' } };
const PAD: OpExtra = { viz: { panel: 'mla', stage: 'pad' } };
/** The attention popup's stages that eager steps and the fused kernel belong to. */
const SCORES: OpExtra = { viz: { panel: 'attention', stage: 'scores' } };
const SOFTMAX: OpExtra = { viz: { panel: 'attention', stage: 'softmax' } };
/**
 * The SwiGLU popup's stages, for an MLP `width` wide: SiLU (alone or fused with the product), and the product. `gelu`:
 * GELU in place of SiLU (GeGLU).
 */
const SILU = (width: number, gelu = false): OpExtra => ({ viz: { panel: 'swiglu', stage: 'silu', width, ...(gelu ? { gelu } : {}) } });
const GATE = (width: number, gelu = false): OpExtra => ({ viz: { panel: 'swiglu', stage: 'gate', width, ...(gelu ? { gelu } : {}) } });
/** Extra space between experts of a mixture-of-experts layer, so each reads as a block. */
const EXPERT_GAP = 8192;

/** Sections of a decoder layer's floor: attention in front, the MLP (or a mixture of experts' router) behind it, then rows of experts. */
export const ATTENTION_SECTION = 0;
export const MLP_SECTION = 1;
export const FIRST_EXPERT_SECTION = 2;

/** Floors: the embedding (0), one per decoder layer, the LM head, then sampling. */
export const EMBEDDING_FLOOR = 0;
export const layerFloor = (layer: number) => layer + 1;
export const floorLayer = (floor: number) => floor - 1;
export const lmHeadFloor = (cfg: ModelConfig) => cfg.nLayers + 1;
export const samplingFloor = (cfg: ModelConfig) => cfg.nLayers + 2;
export const floorCount = (cfg: ModelConfig) => samplingFloor(cfg) + 1;
export const isLayerFloor = (cfg: ModelConfig, floor: number) => floor > EMBEDDING_FLOOR && floor < lmHeadFloor(cfg);

export function floorName(cfg: ModelConfig, floor: number): string {
  if (floor === EMBEDDING_FLOOR) return 'Embedding';
  if (floor === lmHeadFloor(cfg)) return 'LM head';
  if (floor === samplingFloor(cfg)) return 'Sampling';
  return `Layer ${floorLayer(floor)}`;
}

export const elementCount = (s: TensorSpec) => s.x * s.z * s.tiles;
export const byteCount = (s: TensorSpec) => elementCount(s) * DTYPES[s.dtype].bytes;
export const isTransient = (s: TensorSpec) => s.kind === 'activation' || s.kind === 'scores';

export function buildModel(cfg: ModelConfig, o: BuildOptions): Model {
  const tensors: TensorSpec[] = [];
  const ops: OpSpec[] = [];
  const decode = o.mode === 'decode';
  /** Tokens this pass processes; tokens already cached; keys attention reads. */
  const n = decode ? 1 : o.T;
  const past = decode ? o.T : 0;
  const keys = past + n;
  const tok: Dim = { name: decode ? 'new token' : 'T', size: n };
  const dModel: Dim = { name: 'd_model', size: cfg.dModel };
  const qDim: Dim = { name: `q_dim (${cfg.nHeads}×${cfg.headDim})`, size: cfg.nHeads * cfg.headDim };
  const kvDim: Dim = { name: `kv_dim (${cfg.nKvHeads}×${cfg.headDim})`, size: cfg.nKvHeads * cfg.headDim };
  const ffn: Dim = { name: 'd_ffn', size: cfg.dFfn };
  /** Attention output: V's width per head, times the heads (MLA's V is narrower than its Q and K). */
  const oDim: Dim = cfg.mla ? { name: `${cfg.nHeads}×v_head_dim ${vHeadDim(cfg)}`, size: cfg.nHeads * vHeadDim(cfg) } : qDim;
  const vocab: Dim = { name: 'vocab', size: cfg.vocabSize };
  const gpt2 = cfg.arch === 'gpt2';
  const gqaNote = cfg.nKvHeads === cfg.nHeads
    ? `Multi-head attention: each of the ${cfg.nHeads} query heads has its own K and V head.`
    : cfg.nKvHeads === 1
    ? `Multi-query attention: one KV head shared by all ${cfg.nHeads} query heads.`
    : `Grouped-query attention: ${cfg.nKvHeads} KV heads shared by ${cfg.nHeads} query heads.`;
  // Sliding-window attention (Gemma 3): most layers local, every few a global one.
  const sw = cfg.slidingWindow;
  const nSliding = Array.from({ length: cfg.nLayers }, (_, l) => isSlidingLayer(cfg, l)).filter(Boolean).length;
  const nGlobal = cfg.nLayers - nSliding;

  let floor = 0;
  let section = ATTENTION_SECTION;
  let station = 0;
  let prefix = '';

  const push = (s: Omit<TensorSpec, 'floor' | 'section' | 'station' | 'tiles'> & { tiles?: number }) => {
    const id = `${floor}:${s.id}`;
    tensors.push({ tiles: 1, ...s, id, floor, section, station });
    return id;
  };

  /** Weights lie with their input dim along X and output dim along Z. `extra`: further fields of the tensor. */
  const weight = (name: string, inDim: Dim, outDim: Dim, note?: string, extra: Partial<TensorSpec> = {}) =>
    push({ id: name, label: prefix + name, kind: 'weight', x: inDim.size, z: outDim.size, dims: [inDim, outDim], dtype: o.weightDtype, note, ...extra });
  const vector = (name: string, dim: Dim, note?: string, extra: Partial<TensorSpec> = {}) =>
    push({ id: name, label: prefix + name, kind: 'weight', x: dim.size, z: 1, dims: [dim], dtype: o.weightDtype, note, ...extra });
  /** Activations lie with their feature dim along X and `rows` (by default the pass's tokens) along Z. */
  const act = (id: string, label: string, feat: Dim, { kind = 'activation', note, rows = tok, ...extra }: ActOptions = {}) =>
    push({
      id, label, kind, x: feat.size, z: rows.size, dims: [rows, feat],
      dtype: kind === 'kv' ? o.kvDtype : o.actDtype,
      note: note ?? (kind === 'kv' ? KV : TRANSIENT),
      ...extra,
    });
  const qHeads: HeadSplit = { axis: 'x', count: cfg.nHeads, perGroup: cfg.nHeads / cfg.nKvHeads };
  const kvHeads: HeadSplit = { axis: 'x', count: cfg.nKvHeads, perGroup: 1 };
  /** A head split along another axis than its default. */
  const along = (heads: HeadSplit, axis: HeadSplit['axis']): HeadSplit => ({ ...heads, axis });
  const op = (label: string, inputs: string[], output: string, note: string, extra: OpExtra = {}) =>
    ops.push({ id: `${floor}:op:${ops.length}`, label, floor, inputs, output, note, ...extra });
  /**
   * `rows`: tokens multiplied, when fewer than the pass's (an expert's share). `bias`: added in the same kernel.
   * `describe`: rewrites the generic note.
   */
  const matmul = (input: string, w: string, output: string, inDim: Dim, outDim: Dim,
    { rows = n, bias, describe = (note) => note, ...extra }: MatmulOptions = {}) => {
    const flops = 2 * rows * inDim.size * outDim.size;
    op('@', bias ? [input, w, bias] : [input, w], output, describe(
      `Matrix multiply${bias ? ' plus bias (one fused addmm)' : ''}: [${rows} × ${inDim.size}] @ [${inDim.size} × ${outDim.size}] → [${rows} × ${outDim.size}]. ` +
      `${formatFlops(flops)} FLOPs, reading all ${formatFlops(inDim.size * outDim.size)} weights` +
      (decode ? ': about 1 FLOP per weight byte, so decode waits on memory.' : '.')), { flops, ...extra });
  };
  const next = () => station++;
  const nextSection = () => {
    section++;
    station = 0;
  };
  const nextFloorX = () => `${floor + 1}:x`;

  // Embedding floor. Llama: rotary tables first (they feed every layer), then the token embedding. GPT-2: a learned
  // position table, looked up and added to the token embedding.
  const top = gpt2 ? 'transformer.' : 'model.';
  prefix = top;
  // RoPE rotates all of each head's dims, or MLA's decoupled RoPE part.
  const ropeName = cfg.mla ? 'qk_rope_head_dim' : 'head_dim';
  const half: Dim = { name: `${ropeName} / 2`, size: ropeDim(cfg) / 2 };
  const headDim: Dim = { name: ropeName, size: ropeDim(cfg) };
  // position_ids gets its own station after the RoPE tables, next to input_ids; they already read it.
  const pos = `${floor}:position_ids`;
  let cos = '';
  let sin = '';
  const freq: OpViz = { panel: 'rope', stage: 'freq' };
  /** RoPE tables by layer type: one set for every layer, or (sliding window) one for the global and one for the sliding layers. */
  const ropeTables: Partial<Record<'full' | 'sliding', { cos: string; sin: string }>> = {};
  if (!gpt2 && !cfg.ropePerLayer) {
    const types = sw
      ? [{ type: 'full', theta: cfg.ropeTheta!, layers: `the ${nGlobal} global layers` }, { type: 'sliding', theta: sw.ropeTheta, layers: `the ${nSliding} sliding-window layers` }] as const
      : [{ type: 'full', theta: cfg.ropeTheta!, layers: `all ${cfg.nLayers} layers` }] as const;
    for (const { type, theta, layers } of types) {
      // HF names Gemma's buffers by layer type: full_attention_inv_freq, sliding_attention_inv_freq.
      const name = sw ? `${type}_attention_` : '';
      const ropeNote = `Computed once per forward pass and shared by ${layers} and all heads.`;
      const freqs = ropeFrequencies(cfg, theta);
      const yarn = cfg.yarn
        ? ` YaRN stretches the ${formatTokens(cfg.yarn.originalContext)}-token training context ${cfg.yarn.factor}×: the slowest frequencies are divided by ${cfg.yarn.factor}.`
        : '';
      const other = !sw ? ''
        : type === 'full' ? ` For the ${nGlobal} global layers; the sliding-window layers use base ${formatInt(sw.ropeTheta)}.`
        : ` For the ${nSliding} sliding-window layers: they see only ${sw.size} tokens back, so a small base suffices.`;
      const viz: OpViz = type === 'sliding' ? { panel: 'rope', stage: 'freq', theta, window: sw!.size } : freq;
      const invFreq = push({
        id: `rotary_emb.${name}inv_freq`, label: `model.rotary_emb.${name}inv_freq`, kind: 'buffer', x: half.size, z: 1, dims: [half], dtype: 'fp32',
        note: `RoPE frequencies θ_i = ${formatInt(theta)}^(−2i/${headDim.size}): a fixed buffer, not a parameter.${yarn}${other} ` +
          `Wavelengths 2π/θ_i run from ~${formatTokens(2 * Math.PI / freqs[0])} tokens to ~${formatTokens(2 * Math.PI / freqs[freqs.length - 1])} tokens.`,
      });
      next();
      const tables = { cos: '', sin: '' };
      for (const fn of ['cos', 'sin'] as const) {
        const table = act(sw ? `${fn}_${type}` : fn, `${fn}(pos · θ): RoPE table${sw ? (type === 'full' ? ', global layers' : ', sliding layers') : ''}`, headDim, { note: ropeNote });
        op(`${fn}(pos·θ)`, [pos, invFreq], table,
          `Outer product of positions and frequencies, then ${fn === 'cos' ? 'cosine' : 'sine'}; the ${half.size} angles are repeated to fill ${ropeName}.`, { viz });
        tables[fn] = table;
        next();
      }
      ropeTables[type] = tables;
    }
    ({ cos, sin } = ropeTables.full!);
  }
  push({
    id: 'position_ids', label: 'position_ids', kind: 'activation', x: 1, z: n, dims: [tok], dtype: 'int64',
    note: decode ? `Position of the new token: ${formatInt(past)}.` : 'Position of each prompt token: 0 … T−1.',
  });
  next();
  /** Lookup of `rows` rows of `table`, which reads only those rows. */
  const lookup = (label: string, inputs: string[], table: string, output: string, note: string, viz?: OpViz) =>
    op(label, inputs, output, note, { inputBytes: { [table]: n * cfg.dModel * DTYPES[o.weightDtype].bytes }, viz });
  let posEmb = '';
  if (gpt2) {
    const positions: Dim = { name: 'positions', size: cfg.maxContext };
    const wpe = weight('wpe.weight', positions, dModel,
      `Learned position embeddings: one row per position, ${formatInt(cfg.maxContext)} of them, so the context ` +
      `cannot grow past ${formatInt(cfg.maxContext)} tokens. Parameters, unlike RoPE.`);
    posEmb = act('pos_emb', 'position embeddings', dModel, { note: `${TRANSIENT} One row of wpe per position.` });
    lookup('position lookup', [pos, wpe], wpe, posEmb, 'Copies the row of each position from wpe. No arithmetic.');
    next();
  }
  // Eager prefill needs an explicit causal mask; a single decode query may attend to every cached key.
  let causalMask: string | undefined;
  let slidingMask: string | undefined;
  if (o.attention === 'eager' && !decode) {
    const maskDims = [{ name: 'T (query)', size: n }, { name: 'T (key)', size: n }];
    causalMask = push({
      id: 'causal_mask', label: 'causal_mask', kind: 'activation', x: n, z: n, causal: true, dtype: o.actDtype, dims: maskDims,
      note: `Additive mask for eager attention: 0 where key ≤ query, −∞ above. Built once and shared by ${sw ? `the ${nGlobal} global` : `all ${cfg.nLayers}`} layers and heads; fused kernels only take an is_causal flag.`,
    });
    op('causal mask', [pos], causalMask, 'Builds the T×T additive causal mask from the positions.', SCORES);
    next();
    if (sw) {
      slidingMask = push({
        id: 'sliding_mask', label: 'sliding_window_mask', kind: 'activation', x: n, z: n, causal: true, ...(sw.size < n ? { window: sw.size } : {}), dtype: o.actDtype, dims: maskDims,
        note: `Additive mask of the ${nSliding} sliding-window layers: also −∞ where the key is ${sw.size} or more positions back` +
          (n <= sw.size ? `, which no pair is at T = ${formatInt(n)}: here it equals the causal mask.` : '.'),
      });
      op('sliding mask', [pos], slidingMask, `Builds the T×T additive mask of the sliding-window layers: causal, and at most ${sw.size} keys per query.`,
        { viz: { panel: 'attention', stage: 'scores', window: sw.size } });
      next();
    }
  }
  const tied = gpt2 || !!cfg.tiedEmbedding;
  const embed = weight(gpt2 ? 'wte.weight' : 'embed_tokens.weight', vocab, dModel,
    gpt2 ? 'Token embeddings: each token id selects one row. The LM head reuses this table (tied weights).'
      : `Lookup table: each token id selects one row.${tied ? ` The LM head reuses this table (tied weights): ${formatCount(cfg.vocabSize * cfg.dModel)} parameters, stored once.` : ''}`);
  const ids = push({
    id: 'input_ids', label: 'input_ids', kind: 'activation', x: 1, z: n, dims: [tok], dtype: 'int64',
    note: decode ? 'The token sampled by the previous step.' : 'Token ids of the prompt.',
  });
  if (gpt2) {
    const tokEmb = act('tok_emb', 'token embeddings', dModel, { note: `${TRANSIENT} One row of wte per token.`, rowGap: STACK_GAP });
    lookup('lookup', [ids, embed], embed, tokEmb, 'Embedding lookup: copies one row per token. No arithmetic.', { panel: 'lookup' });
    op('+', [tokEmb, posEmb], nextFloorX(), "Token plus position embedding: the residual stream's start, layer 0's input.");
  } else if (cfg.embedScale) {
    // Gemma: the looked-up rows are multiplied by √d_model, cast to the weight dtype.
    const tokEmb = act('tok_emb', 'token embeddings', dModel, { note: `${TRANSIENT} One row of embed_tokens per token.`, rowGap: STACK_GAP });
    lookup('lookup', [ids, embed], embed, tokEmb, 'Embedding lookup: copies one row per token. No arithmetic.', { panel: 'lookup' });
    const root = Math.sqrt(cfg.dModel);
    const rounded = o.weightDtype === 'bf16' ? `, ${bf16(root)} in bf16` : '';
    const scale = push({
      id: 'embed_scale', label: 'model.embed_tokens.embed_scale', kind: 'buffer', x: 1, z: 1, dims: [{ name: '1', size: 1 }], dtype: 'fp32', beside: true,
      note: `√d_model = √${cfg.dModel} ≈ ${root.toFixed(2)}: a fixed scalar, not a parameter.`,
    });
    op('× √d', [tokEmb, scale], nextFloorX(),
      `Multiplies the embeddings by √${cfg.dModel} (cast to the weight dtype${rounded}): the residual stream's start, layer 0's input. ` +
      'The table stays small enough to double as the LM head, while the stream starts at the scale the layers expect.');
  } else {
    lookup('lookup', [ids, embed], embed, nextFloorX(), 'Embedding lookup: copies one row per token. No arithmetic.', { panel: 'lookup' });
  }
  /** An activation stacked behind the previous one in its station's front row, with room for its op. */
  const stacked = (id: string, label: string, feat: Dim, dtype: DType, note: string, extra: Partial<TensorSpec> = {}) =>
    push({ id, label, kind: 'activation', x: feat.size, z: n, rowGap: STACK_GAP, dims: [tok, feat], dtype, note, ...extra });
  const one: Dim = { name: '1', size: 1 };

  /**
   * The model's norm. `names`: [llama, gpt2] module names, e.g. ['input_layernorm', 'ln_1']. GPT-2's LayerNorm is
   * torch.nn.LayerNorm: one fused kernel, so even expanded it allocates nothing but its output.
   */
  const norm = (input: string, names: [string, string], id: string, what: string) => {
    if (!gpt2) return rmsNorm(input, `${names[0]}.weight`, id, `${id} = RMSNorm(${what})`);
    const w = vector(`${names[1]}.weight`, dModel);
    const b = vector(`${names[1]}.bias`, dModel);
    const out = act(id, `${id} = LayerNorm(${what})`, dModel);
    op('LayerNorm', [input, w, b], out,
      `Per token: subtract the mean, divide by √(variance + ${cfg.normEps}), then × weight + bias (${2 * cfg.dModel} parameters). ` +
      'One fused kernel (torch.nn.LayerNorm): no intermediate tensors, unlike Llama\'s RMSNorm written in PyTorch.', { viz: { panel: 'layernorm' } });
    return out;
  };

  /**
   * RMSNorm with its weight; expanded, every intermediate HF's LlamaRMSNorm (or Gemma3RMSNorm) allocates. `group`:
   * normalize each head's `group`-wide slice of a `dim`-wide tensor split into `heads` (QK-norm), one weight for all.
   */
  const rmsNorm = (input: string, weightName: string, id: string, label: string, dim: Dim = dModel,
    { heads, group, ...out_ }: { kind?: 'activation' | 'kv'; note?: string; heads?: HeadSplit; group?: Dim } = {}) => {
    const normed = group ?? dim;
    const w = vector(weightName, normed);
    const offset = !!cfg.normOffset;
    const stage = (s: VizStage<'rmsnorm'>): OpExtra => ({ viz: { panel: 'rmsnorm', stage: s, ...(group ? { width: group.size } : {}) } });
    const each = group ? `each head's ${group.size}-dim vector` : 'each token vector';
    const scaleBy = offset ? `(1 + weight), in fp32${group ? ', one weight for all heads' : ''}` : 'the norm weight';
    const split = heads ? { heads } : {};
    if (o.elementwise === 'compact') {
      const out = act(id, label, dim, { ...out_, ...split });
      op('RMSNorm', [input, w], out, `Normalize ${each} by its RMS, then scale by ${scaleBy}.`, stage('input'));
      return out;
    }
    const tmp = (suffix: string, tmpLabel: string, dim: Dim, dtype: DType, note: string, extra: Partial<TensorSpec> = {}) =>
      stacked(`${id}.${suffix}`, tmpLabel, dim, dtype, `${TRANSIENT} ${note}`, { ...split, ...extra });
    const perRow: Dim = heads ? { name: 'heads', size: heads.count } : one;
    const per = heads ? 'One value per head and token.' : 'One value per token.';
    const x32 = tmp('fp32', 'x in fp32', dim, 'fp32', 'Upcast for accuracy: twice the bytes of x.');
    op('to fp32', [input], x32, 'RMSNorm computes in fp32.', stage('input'));
    const sq = tmp('sq', 'x²', dim, 'fp32', 'Squares, elementwise.');
    op('x²', [x32], sq, 'Elementwise square.', stage('mean'));
    const ms = tmp('mean', 'mean(x²)', perRow, 'fp32', per);
    op('mean', [sq], ms, `Mean over the ${normed.size} features of each ${heads ? 'head' : 'token'}.`, stage('mean'));
    const r = tmp('rsqrt', '1/√(mean(x²) + ε)', perRow, 'fp32', heads ? 'One scale per head and token.' : 'One scale per token.', { beside: true });
    op('rsqrt(·+ε)', [ms], r, `Adds ε = ${cfg.normEps} and takes the reciprocal square root.`, stage('mean'));
    const n32 = tmp('n32', 'x / RMS(x), fp32', dim, 'fp32', `Every ${heads ? 'head' : 'token'} vector now has RMS 1.`);
    op('×', [x32, r], n32, `Scales ${each} by its own 1/RMS.`, stage('normalize'));
    const outDtype = out_.kind === 'kv' ? o.kvDtype : o.actDtype;
    const outExtra = { ...split, ...(out_.kind ? { kind: out_.kind } : {}) };
    if (offset) {
      // Gemma3RMSNorm: output * (1.0 + weight.float()), then type_as(x): the weight applies before the cast.
      const w1 = push({
        id: `${id}.w1`, label: '1 + weight, fp32', kind: 'activation', x: normed.size, z: 1, rowGap: STACK_GAP, dims: [normed], dtype: 'fp32',
        note: `${TRANSIENT} The stored weight is an offset from 1: a weight of 0 leaves x / RMS(x) unchanged.`,
      });
      op('1 + w', [w], w1, `Upcasts the ${normed.size} weights to fp32 and adds 1.`, stage('weight'));
      const s32 = tmp('s32', 'x / RMS(x) · (1 + weight), fp32', dim, 'fp32', 'Scaled while still in fp32.');
      op('×', [n32, w1], s32, `Multiplies each feature by its scale, 1 + weight: the norm's only ${normed.size} parameters.`, stage('weight'));
      const out = stacked(id, label, dim, outDtype, out_.note ?? TRANSIENT, outExtra);
      op(`to ${DTYPES[outDtype].label}`, [s32], out, 'Casts back to the activation dtype, after the weight (Llama casts before it).', stage('weight'));
      return out;
    }
    const n = tmp('n', `x / RMS(x), ${DTYPES[o.actDtype].label}`, dim, o.actDtype, 'Cast back before the weight is applied.');
    op(`to ${DTYPES[o.actDtype].label}`, [n32], n, 'Casts back to the activation dtype.', stage('normalize'));
    const out = stacked(id, label, dim, outDtype, out_.note ?? TRANSIENT, outExtra);
    op('× weight', [n, w], out, `Multiplies each feature by its learned scale: the norm's only ${normed.size} parameters.`, stage('weight'));
    return out;
  };

  const pairing = cfg.ropeInterleaved ? 'neighboring dimension pairs (2i, 2i+1)' : `dimension pairs (i, i+${half.size})`;
  /** The RoPE popup's rotate stage; a sliding layer's carries its own base and window. */
  let rotateViz: OpViz = { panel: 'rope', stage: 'rotate' };
  const ropeOp = (input: string, output: string, heads: number, what = `in each of the ${heads} heads, rotates ${pairing}`) =>
    op('RoPE', cfg.ropePerLayer ? [input, cos, sin, pos] : [input, cos, sin], output,
      `Rotary position embedding: ${what} by angle pos·θ_i. ` +
      'No parameters; after it, Q·Kᵀ depends on the distance between positions.', { viz: rotateViz });

  /**
   * Sparse mixture of experts (HF MixtralSparseMoeBlock): a router picks the top-k experts per token; each expert
   * is a SwiGLU MLP run on the rows of its tokens only, and the outputs are summed back, weighted by the router.
   * The router and the summed output `m` share a section; the experts form a grid behind it, one section per row,
   * each expert a block of two stations (gate and up, then down). An expert no token picks keeps its weights (dimmed) but gets no
   * activations. Returns the tokens each expert receives.
   */
  const moe = (l: number, h: string, hNorm: string): number[] => {
    const { nExperts, topK, expertFfn, sharedFfn, renormalize, gridColumns: cols, names } = cfg.moe!;
    const experts: Dim = { name: 'experts', size: nExperts };
    const picks: Dim = { name: `top-${topK}`, size: topK };
    const ffn: Dim = { name: 'd_ffn', size: expertFfn };
    const counts = routing(cfg).counts(l, past, past + n);
    prefix = `model.layers.${l}.${names.block}.`;
    const stage = (s: VizStage<'moe'>): OpExtra => ({ viz: { panel: 'moe', stage: s } });
    const wr = weight('gate.weight', dModel, experts,
      `The router: scores all ${nExperts} experts for each token. ${formatCount(cfg.dModel * nExperts)} parameters, ` +
      `against ${formatCount(3 * cfg.dModel * expertFfn)} in each expert. Column e scores expert e.`, { expertAxis: 'z' });
    const routerLogits = act('router_logits', 'router_logits = h_norm · W_router', experts, { expertAxis: 'x' });
    matmul(hNorm, wr, routerLogits, dModel, experts, stage('router'));
    next();
    const routerProbs = push({
      id: 'router_probs', label: 'softmax(router_logits), fp32', kind: 'activation', x: nExperts, z: n,
      dims: [tok, experts], dtype: 'fp32', expertAxis: 'x', note: `${TRANSIENT} Routing probabilities over the ${nExperts} experts.`,
    });
    op('softmax', [routerLogits], routerProbs, 'Softmax over the experts, computed in fp32.', stage('router'));
    next();
    const selected = push({
      id: 'selected_experts', label: `selected_experts: top-${topK} expert ids`, kind: 'activation', x: topK, z: n,
      dims: [tok, picks], dtype: 'int64', note: `${TRANSIENT} The ${topK} experts each token goes to. Illustrative routing, seeded.`,
    });
    op(`top-${topK}`, [routerProbs], selected, `torch.topk: the ${topK} most probable experts of each token.`, stage('router'));
    const weights = push({
      id: 'routing_weights', label: 'routing_weights', kind: 'activation', x: topK, z: n, beside: true,
      dims: [tok, picks], dtype: o.actDtype,
      note: renormalize
        ? `${TRANSIENT} The top-${topK} probabilities, renormalized to sum to 1: how much each chosen expert's output counts.`
        : `${TRANSIENT} The top-${topK} probabilities as they are (norm_topk_prob false): they sum to less than 1, so the routed experts together count for less than the shared ones.`,
    });
    op(renormalize ? '÷ sum' : `top-${topK} probs`, [routerProbs], weights, renormalize
      ? `The top-${topK} probabilities divided by their sum, then cast to the activation dtype.`
      : `The top-${topK} probabilities, not renormalized, cast to the activation dtype.`, stage('router'));
    next();

    // Shared experts (DeepSeek): one SwiGLU MLP every token runs through, between the router and the sum.
    let shared: string | undefined;
    if (sharedFfn) {
      const sFfn: Dim = { name: 'd_ffn (shared)', size: sharedFfn };
      const sharedWeight = (name: string, inDim: Dim, outDim: Dim, what: string) =>
        weight(`shared_experts.${name}.weight`, inDim, outDim, `${what} Shared experts: every token runs through them, unrouted.`, { shared: true });
      const sharedAct = (id: string, label: string, feat: Dim) =>
        push({ id: `shared.${id}`, label: `${label} (shared)`, kind: 'activation', x: feat.size, z: n, rowGap: STACK_GAP, dims: [tok, feat], dtype: o.actDtype, shared: true, note: TRANSIENT });
      const wg = sharedWeight(names.gate, dModel, sFfn, 'Gate projection (W_gate).');
      const wu = sharedWeight(names.up, dModel, sFfn, 'Up projection (W_up).');
      const g = sharedAct('g', 'g = h_norm · W_gate', sFfn);
      matmul(hNorm, wg, g, dModel, sFfn);
      const u = sharedAct('u', 'u = h_norm · W_up', sFfn);
      matmul(hNorm, wu, u, dModel, sFfn);
      next();
      const wd = sharedWeight(names.down, sFfn, dModel, 'Down projection (W_down).');
      const a = sharedAct('a', 'a = SiLU(g) ⊙ u', sFfn);
      op('SiLU ⊙', [g, u], a, 'SwiGLU: elementwise SiLU(g) times u.', SILU(sharedFfn));
      shared = sharedAct('y', 'y = a · W_down', dModel);
      matmul(a, wd, shared, sFfn, dModel);
      next();
    }
    const sum = push({
      id: shared ? 'm_routed' : 'm', label: `${shared ? '' : 'm = '}Σ routing weight × expert output`, kind: 'activation', x: cfg.dModel, z: n, gapBefore: EXPERT_GAP,
      dims: [tok, dModel], dtype: o.actDtype,
      note: `${TRANSIENT} Starts as zeros; each expert adds its weighted rows into the rows of its tokens, so every row receives ${topK} contributions.`,
    });
    const m = shared
      ? push({ id: 'm', label: 'm = routed experts + shared experts', kind: 'activation', x: cfg.dModel, z: n, rowGap: STACK_GAP, dims: [tok, dModel], dtype: o.actDtype, note: TRANSIENT })
      : sum;

    // Every expert reserves room for the busiest expert's activations, so experts keep their place whichever
    // are active: the gate/up station is as wide as g and u, and the deepest station's stack sets each row's depth.
    const expanded = o.elementwise === 'expanded';
    const stackedRows = expanded ? 4 : 3;
    const reserveFront = stackedRows * Math.max(...counts) + (stackedRows - 1) * STACK_GAP;
    const outputs: string[] = [];
    for (let e = 0; e < nExperts; e++) {
      if (e % cols === 0) nextSection();
      const ne = counts[e];
      const idle = ne === 0;
      const rows: Dim = { name: `tokens routed to expert ${e}`, size: ne };
      const share = idle
        ? `No token picks expert ${e} in this pass, so its weights are not read.`
        : `Expert ${e} of ${nExperts}: ${formatInt(ne)} of ${formatInt(n)} tokens routed here.`;
      // The expert index is part of the id: each expert has its own gate, up and down projections.
      const expertWeight = (name: string, inDim: Dim, outDim: Dim, what: string, extra: Partial<TensorSpec> = {}) =>
        weight(`experts.${e}.${name}`, inDim, outDim, `${what} ${share}`, { expert: e, idle, ...extra });
      /** An activation of this expert: one row per token routed to it, stacked in its station. */
      const expertAct = (id: string, label: string, feat: Dim, dtype: DType, note: string) => push({
        id: `e${e}.${id}`, label: `${label} (expert ${e})`, kind: 'activation', x: feat.size, z: ne, rowGap: STACK_GAP,
        dims: [rows, feat], dtype, expert: e, note: `${TRANSIENT} ${note}`,
      });
      // Gate and up projections read the same rows: their weights lie back to back in one station.
      const w1 = expertWeight(`${names.gate}.weight`, dModel, ffn, 'Gate projection (W_gate).',
        { reserveWidth: expertFfn, reserveFront, ...(e % cols ? { gapBefore: EXPERT_GAP } : {}) });
      const w3 = expertWeight(`${names.up}.weight`, dModel, ffn, 'Up projection (W_up).');
      let g = '', u = '';
      if (!idle) {
        const xe = expertAct('x', 'h_norm rows', dModel, o.actDtype, 'A copy of the h_norm rows of the tokens routed to this expert.');
        op(`gather (E${e})`, [hNorm, selected], xe,
          `Selects the ${formatInt(ne)} tokens whose top-${topK} includes expert ${e} and copies their rows (hidden_states[top_x]).`, stage('dispatch'));
        g = expertAct('g', 'g = x · W_gate', ffn, o.actDtype, 'Gate activations.');
        matmul(xe, w1, g, dModel, ffn, { rows: ne });
        u = expertAct('u', 'u = x · W_up', ffn, o.actDtype, 'Up activations.');
        matmul(xe, w3, u, dModel, ffn, { rows: ne });
      }
      next();
      const w2 = expertWeight(`${names.down}.weight`, ffn, dModel, 'Down projection (W_down).', { reserveFront });
      if (!idle) {
        let a: string;
        if (expanded) {
          const sg = expertAct('silu_g', 'SiLU(g)', ffn, o.actDtype, 'SiLU of the gate, its own tensor.');
          op('SiLU', [g], sg, 'SiLU(g) = g · sigmoid(g), elementwise.', SILU(expertFfn));
          a = expertAct('a', 'a = SiLU(g) ⊙ u', ffn, o.actDtype, 'The SwiGLU gate.');
          op('⊙', [sg, u], a, 'Elementwise product with u.', GATE(expertFfn));
        } else {
          a = expertAct('a', 'a = SiLU(g) ⊙ u', ffn, o.actDtype, 'The SwiGLU gate.');
          op('SiLU ⊙', [g, u], a, 'SwiGLU: elementwise SiLU(g) times u.', SILU(expertFfn));
        }
        const y = expertAct('y', 'y = a · W_down', dModel, o.actDtype, "This expert's output for its tokens.");
        matmul(a, w2, y, ffn, dModel, { rows: ne });
        if (expanded) {
          const ys = expertAct('ys', 'y × routing weight', dModel, o.actDtype, 'Each row scaled by its routing weight for this expert.');
          op('× weight', [y, weights], ys, "Multiplies each token's row by the routing weight it gave this expert.", stage('combine'));
          outputs.push(ys);
        } else {
          outputs.push(y);
        }
      }
      next();
    }

    op(expanded ? 'index_add' : '× weight, index_add', expanded ? outputs : [...outputs, weights], sum,
      `${expanded ? '' : 'Scales each expert output row by its routing weight, then '}index_add_ adds the rows back at their tokens' positions. ` +
      `${outputs.length} of ${nExperts} experts contribute in this pass.`, stage('combine'));
    if (shared) op('+', [sum, shared], m, 'Adds the shared experts\' output to the routed experts\' sum.', stage('combine'));
    op('+', [m, h], nextFloorX(), "Residual add: h + m is this layer's output and the next floor's input.");
    return counts;
  };
  /**
   * DeepSeek-V2 (ropePerLayer): each attention layer's own rotary module, with inv_freq and cos / sin tables for every
   * position seen so far. The tables are buffers kept between passes and rebuilt in full whenever the sequence grows,
   * which in decode is every step; RoPE then gathers the rows of this pass's positions.
   */
  const layerRopeTables = () => {
    const freqs = ropeFrequencies(cfg);
    const positions: Dim = { name: decode ? 'positions so far' : 'T', size: keys };
    const invFreq = push({
      id: 'rotary_emb.inv_freq', label: `${prefix}self_attn.rotary_emb.inv_freq`, kind: 'buffer', x: half.size, z: 1, dims: [half], dtype: 'fp32',
      note: `This layer's own copy of the RoPE frequencies, YaRN-adjusted: wavelengths from ~${formatTokens(2 * Math.PI / freqs[0])} ` +
        `to ~${formatTokens(2 * Math.PI / freqs[freqs.length - 1])} tokens. Each attention layer builds its rotary module, and its tables, separately.`,
    });
    const tables = (['cos', 'sin'] as const).map((fn) => {
      const table = push({
        id: `rotary_emb.${fn}_cached`, label: `${prefix}self_attn.rotary_emb.${fn}_cached`, kind: 'buffer', x: headDim.size, z: keys,
        rowGap: STACK_GAP, dims: [positions, headDim], dtype: o.actDtype,
        note: `${fn}(pos · θ) for positions 0 … ${formatInt(keys - 1)}: a buffer kept between passes, one per layer (${cfg.nLayers} in all). ` +
          'Rebuilt for every position whenever the sequence grows: in decode, every step.',
      });
      op(`${fn}(pos·θ)`, [invFreq], table,
        `Outer product of positions 0 … ${formatInt(keys - 1)} and the frequencies, then ${fn === 'cos' ? 'cosine' : 'sine'}, the ${half.size} angles written twice. ` +
        (decode ? 'The sequence grew by one, so the whole table is computed again.' : 'Computed in full on the first pass.'), { viz: freq });
      return table;
    });
    return { cos: tables[0], sin: tables[1] };
  };
  /** By decoder layer; dense layers have none. */
  const expertTokens: number[][] = [];

  for (let l = 0; l < cfg.nLayers; l++) {
    floor = layerFloor(l);
    section = ATTENTION_SECTION;
    station = 0;
    prefix = gpt2 ? `transformer.h.${l}.` : `model.layers.${l}.`;

    // Attention section.
    const x = act('x', 'x: residual stream in', dModel, { note: `${TRANSIENT} Embeddings for layer 0, the previous layer's output otherwise.`, residual: true });
    next();
    const xNorm = norm(x, ['input_layernorm', 'ln_1'], 'x_norm', 'x');
    next();
    if (cfg.ropePerLayer) {
      ({ cos, sin } = layerRopeTables());
      next();
    }
    // Sliding-window layers (Gemma 3): their own RoPE tables, a banded mask, and a cache that stops at the window.
    const window = layerWindow(cfg, l);
    const sliding = window < Infinity;
    if (sw) {
      ({ cos, sin } = ropeTables[sliding ? 'sliding' : 'full']!);
      rotateViz = sliding ? { panel: 'rope', stage: 'rotate', theta: sw.ropeTheta, window } : { panel: 'rope', stage: 'rotate' };
    }
    // The attention popup's stages; a sliding layer's also show its window.
    const attnViz = (stage: VizStage<'attention'>): OpViz => ({ panel: 'attention', stage, ...(sliding ? { window } : {}) });
    const layerScores: OpExtra = sliding ? { viz: attnViz('scores') } : SCORES;
    const layerSoftmax: OpExtra = sliding ? { viz: attnViz('softmax') } : SOFTMAX;
    // Where K and V live afterwards:
    // - dynamic, prefill: the cache keeps K and V themselves;
    // - dynamic, decode: torch.cat copies the past cache plus the new row into a new tensor;
    // - static: K and V are copied into buffers preallocated for the full capacity.
    // Each cache tensor is stacked in front of the K or V it receives, in the same station.
    const isStatic = o.kvCache === 'static';
    const keepsKv = !isStatic && !decode;
    const kvKind = keepsKv ? 'kv' : 'activation';
    // DynamicSlidingWindowLayer keeps a view of the last window − 1 rows of what it returns; while the view lives, so
    // does the whole tensor.
    const kept = window - 1;
    const slidingKeep = sliding && n > kept
      ? ` Sliding window: the cache keeps a view of only the last ${formatInt(kept)} rows, but the view holds on to all ` +
        `${formatInt(n)}; the first decode step shrinks it to ${formatInt(window)}.`
      : '';
    const kvNote = (extra: string) => (keepsKv
      ? `${KV} Dynamic cache: in prefill the cache keeps this tensor as is, so it grows with the prompt.${extra}${slidingKeep}`
      : `${TRANSIENT} Copied into the cache.`);
    const cap = o.cacheCapacity;
    const fresh = decode ? 1 : undefined;
    /**
     * Caches `src` (`dim` wide, split into `heads` if any) as the HF cache does; `name` says what it holds.
     * Returns what attention reads: the cache, or (a sliding layer's prompt longer than its static window) `src`.
     */
    const cacheWrite = (src: string, id: string, label: string, name: string, dim: Dim = kvDim, heads: HeadSplit | undefined = kvHeads) => {
      const cacheDims = (rows: Dim) => [rows, dim];
      if (keepsKv) return src;
      if (isStatic) {
        // StaticSlidingWindowLayer: a buffer as long as the window. Once full, each step rolls it left by one; a
        // prompt longer than the window leaves only its last rows in it.
        const slots = sliding ? Math.min(window, cap) : cap;
        const rolls = sliding && decode && past >= slots;
        const overflow = sliding && !decode && n > slots;
        const buffer = push({
          id, label, kind: 'kv', x: dim.size, z: slots, filledZ: Math.min(keys, slots), freshZ: fresh, rowGap: STACK_GAP, dtype: o.kvDtype, heads,
          dims: cacheDims({ name: sliding ? 'window' : 'capacity', size: slots }),
          note: sliding
            ? `Static sliding-window cache: ${formatInt(slots)} slots, the window, whatever the capacity (${formatInt(cap)}). ` +
              (rolls || overflow ? 'Full: it holds the last window of tokens.' : `${formatInt(keys)} slots written so far; each generated token fills one more, then the oldest rolls out.`)
            : `Static cache: allocated up front for ${formatInt(cap)} tokens, whatever the prompt length. ` +
              `${formatInt(keys)} slots written so far; each generated token fills one more.`,
        });
        op('cache write', [src], buffer, rolls
          ? `The window is full: roll(−1) copies the ${formatInt(slots)}-row buffer shifted by one, the new ${name} row overwrites the last slot, ` +
            `and copy_ writes it all back: ${formatInt(slots)} rows moved per step, not 1.`
          : overflow
          ? `The prompt is longer than the window: copy_ of the last ${formatInt(slots)} of its ${formatInt(n)} ${name} rows. Attention reads ${name} itself.`
          : decode
          ? `index_copy_ of the new ${name} row into slot ${formatInt(past)}: only ${n} row is written.`
          : `index_copy_ of ${name} into the preallocated buffer, at positions 0 … T−1.`, { viz: { panel: 'kvcache', stage: 'static' } });
        return overflow ? src : buffer;
      }
      // Dynamic cache in decode: the previous cache tensor, then the new one torch.cat builds from it. A sliding layer
      // kept only the last window − 1 rows.
      const before = sliding ? Math.min(past, kept) : past;
      const after = before + n;
      const previous = push({
        id: `${id}_past`, label: `${label} (previous step)`, kind: 'kv', x: dim.size, z: before, superseded: true, heads,
        rowGap: STACK_GAP, dtype: o.kvDtype, dims: cacheDims({ name: 'cached tokens', size: before }),
        note: 'The cache tensor from the previous step. Freed once torch.cat has copied it into the new one.',
      });
      const grown = push({
        id, label, kind: 'kv', x: dim.size, z: after, filledZ: after, freshZ: fresh, rowGap: STACK_GAP, dtype: o.kvDtype, heads,
        dims: cacheDims({ name: 'cached tokens', size: after }),
        note: sliding
          ? `${KV} The window: attention reads all ${formatInt(after)} rows, and the cache keeps a view of the last ${formatInt(Math.min(after, kept))}, so it never grows past ${formatInt(window)}.`
          : `${KV} One row longer than before; the newest row is highlighted.`,
      });
      op('cat', [previous, src], grown, sliding
        ? `DynamicSlidingWindowLayer.update: torch.cat copies the ${formatInt(before)} cached rows plus the new ${name} row into a new ` +
          `${formatInt(after)}-row tensor, every step, then keeps a view of its last ${formatInt(kept)}: the cache stops growing at the window.`
        : `DynamicCache.update: torch.cat allocates a new ${formatInt(keys)}-row tensor and copies the whole ` +
          `cache plus the new ${name} row into it, every step. A static cache writes just the one row.`, { viz: { panel: 'kvcache', stage: 'dynamic' } });
      return grown;
    };
    /** Keys attention reads: a decode query of a sliding layer only the window's. */
    const attnKeys = decode ? Math.min(keys, window) : keys;
    const keyRows: Dim = { name: decode ? 'cached tokens' : 'T', size: attnKeys };
    let q: string;
    let kCached: string;
    let vCached: string;
    /** V zero-padded to the Q and K head width (DeepSeek's FlashAttention path): the attention output is too. */
    let vPadded = false;
    if (gpt2) {
      // One fused projection (c_attn, a Conv1D: x · W + b) yields Q, K and V side by side, as views of one [T × 3d]
      // tensor. Drawn as its three column blocks, each reading a third of the weight; position is already in x.
      const qkvDim: Dim = { name: '3 · d_model (Q, K, V)', size: 3 * cfg.dModel };
      const wqkv = weight('attn.c_attn.weight', dModel, qkvDim,
        `Q, K and V projections fused into one matrix (Conv1D: input dim first): output columns 0–${cfg.dModel - 1} are W_q, then W_k, then W_v. ${gqaNote}`,
        { fusedRoles: 'z', heads: along(qHeads, 'z') });
      const bqkv = vector('attn.c_attn.bias', qkvDim, undefined, { fusedRoles: 'x', heads: along(qHeads, 'x') });
      const third = (out: string, dim: Dim, what: 'Q' | 'K' | 'V') => {
        const role = what === 'Q' ? 'q' : what === 'K' ? 'k' : 'v';
        matmul(xNorm, wqkv, out, dModel, dim, {
          bias: bqkv,
          describe: (note) => `${what}: one third of c_attn, the fused projection. ${note} In HF one matmul yields Q, K and V ` +
            `together, as views of one [${n} × ${3 * cfg.dModel}] tensor.`,
          inputBytes: { [wqkv]: cfg.dModel * dim.size * DTYPES[o.weightDtype].bytes, [bqkv]: dim.size * DTYPES[o.weightDtype].bytes },
          inputRoles: { [wqkv]: role, [bqkv]: role },
        });
      };
      q = act('q', 'Q = x_norm · W_attn[Q] + b', qDim, { heads: qHeads });
      third(q, qDim, 'Q');
      next();
      const k = act('k', 'K = x_norm · W_attn[K] + b', kvDim, { kind: kvKind, note: kvNote(' Learned positions were added at the bottom, so K carries no rotation.'), heads: kvHeads });
      third(k, kvDim, 'K');
      kCached = cacheWrite(k, 'k_cache', `past_key_values.layers[${l}].keys`, 'K');
      next();
      const v = act('v', 'V = x_norm · W_attn[V] + b', kvDim, { kind: kvKind, note: kvNote(''), heads: kvHeads });
      third(v, kvDim, 'V');
      vCached = cacheWrite(v, 'v_cache', `past_key_values.layers[${l}].values`, 'V');
      next();
    } else if (cfg.mla) {
      // Multi-head latent attention (see MlaConfig): Q as usual, with RoPE on part of each head; K and V expanded
      // from a small normalized latent, plus one RoPE key shared by all heads.
      const { kvLoraRank, qkNopeDim, qkRopeDim, vDim: vHead } = cfg.mla;
      const latent: Dim = { name: 'kv_lora_rank', size: kvLoraRank };
      const ropeKey: Dim = { name: 'qk_rope_head_dim', size: qkRopeDim };
      const kvA: Dim = { name: `kv_lora_rank + qk_rope_head_dim (${kvLoraRank} + ${qkRopeDim})`, size: kvLoraRank + qkRopeDim };
      const kvB: Dim = { name: `${cfg.nHeads}×(qk_nope_head_dim ${qkNopeDim} + v_head_dim ${vHead})`, size: cfg.nHeads * (qkNopeDim + vHead) };
      const latentCache = o.mlaCache === 'latent';
      const wq = weight('self_attn.q_proj.weight', dModel, qDim,
        `Each head's query: ${qkNopeDim} dims matched against K from the latent, then ${qkRopeDim} that carry position. ` +
        'No query compression in V2-Lite (q_lora_rank null).', { heads: along(qHeads, 'z'), role: 'q' });
      const qRaw = act('q_raw', 'x_norm · W_q (before RoPE)', qDim, { heads: qHeads });
      matmul(xNorm, wq, qRaw, dModel, qDim);
      next();
      q = act('q', `Q: RoPE on the last ${qkRopeDim} dims of each head`, qDim, { heads: qHeads });
      ropeOp(qRaw, q, cfg.nHeads, `in each of the ${cfg.nHeads} heads, rotates only the last ${qkRopeDim} of its ${cfg.headDim} dims, ` +
        `as ${pairing} (decoupled RoPE: the first ${qkNopeDim} carry no position), then copies the parts back together,`);
      next();
      const wkvA = weight('self_attn.kv_a_proj_with_mqa.weight', dModel, kvA,
        `The compression: ${formatInt(cfg.dModel)} features down to a ${kvLoraRank}-dim latent, plus one ${qkRopeDim}-dim key ` +
        `for RoPE shared by all ${cfg.nHeads} heads (as in multi-query attention: "mqa").`);
      const kvARaw = act('kv_a', 'x_norm · W_kv_a: latent and RoPE key', kvA);
      matmul(xNorm, wkvA, kvARaw, dModel, kvA, { viz: { panel: 'mla', stage: 'compress' } });
      next();
      // The normalized latent and the rotated shared key, each in its own station. With a latent cache each is cached in
      // front of itself, so the caches lie side by side and a static cache's long buffers do not run into each other.
      const latentKind = latentCache ? kvKind : 'activation';
      const latentNote = (what: string) => (latentCache ? kvNote(` The latent cache keeps ${what}: ${kvLoraRank + qkRopeDim} values per token, against ${formatInt(kvDim.size + oDim.size)} for K and V.`) : TRANSIENT);
      const cKv = rmsNorm(kvARaw, 'self_attn.kv_a_layernorm.weight', 'c_kv', `c_kv = RMSNorm(latent): ${kvLoraRank} dims`, latent,
        { kind: latentKind, note: latentNote('this latent') });
      const cKvCached = latentCache ? cacheWrite(cKv, 'c_kv_cache', `latent cache, layer ${l}: c_kv`, 'latent', latent, undefined) : cKv;
      next();
      const kPe = act('k_pe', `k_pe = RoPE(shared key): ${qkRopeDim} dims`, ropeKey,
        { kind: latentKind, note: latentNote('the latent and this key') });
      ropeOp(kvARaw, kPe, 1, `rotates the one ${qkRopeDim}-dim key shared by all heads, as ${pairing},`);
      const kPeCached = latentCache ? cacheWrite(kPe, 'k_pe_cache', `latent cache, layer ${l}: k_pe`, 'RoPE key', ropeKey, undefined) : kPe;
      next();
      // The expansion: this pass's tokens, or with a latent cache every cached token, recomputed each pass.
      const wkvB = weight('self_attn.kv_b_proj.weight', latent, kvB,
        `The expansion: each head's K (${qkNopeDim} dims, no position) and V (${vHead}) from the ${kvLoraRank}-dim latent.`,
        { heads: along(kvHeads, 'z') });
      const rows = latentCache ? keyRows : tok;
      const kv = act('kv', `c_kv · W_kv_b: each head's K (no RoPE) and V${latentCache ? ', all cached tokens' : ''}`, kvB,
        { heads: kvHeads, rows, note: latentCache ? `${TRANSIENT} Recomputed from the whole latent cache every pass.` : TRANSIENT });
      matmul(cKvCached, wkvB, kv, latent, kvB, { rows: rows.size, viz: { panel: 'mla', stage: 'expand' } });
      next();
      // K, then V, each in its own station with (HF) its cache in front of it, so the caches lie side by side.
      // HF's FlashAttention path pads V to the Q and K head width before caching it; the eager path does not.
      vPadded = !latentCache && o.attention === 'fused';
      const vCachedWidth = vPadded ? kvDim : oDim;
      const absorb = 'vLLM and SGLang expand like this in prefill; in decode they fold W_kv_b into the query and output ' +
        'projections instead ("absorption") and attend over the latent directly, never building K and V.';
      const hfNote = ` HF caches K and V expanded: ${formatInt(kvDim.size + vCachedWidth.size)} values per token, against ${kvLoraRank + qkRopeDim} for the latent and RoPE key (see MLA cache).`;
      const k = act('k', `K = [k_nope, k_pe]: ${cfg.headDim} dims per head`, kvDim,
        { heads: kvHeads, rows, ...(latentCache ? { note: `${TRANSIENT} ${absorb}` } : { kind: kvKind, note: kvNote(hfNote) }) });
      op('cat', [kv, kPeCached], k,
        `Joins each head's ${qkNopeDim}-dim K with the shared ${qkRopeDim}-dim RoPE key, broadcast to all ${cfg.nHeads} heads: a new tensor.`, ASSEMBLE);
      kCached = latentCache ? k : cacheWrite(k, 'k_cache', `past_key_values.layers[${l}].keys`, 'K');
      next();
      let v = act('v', `V: the last ${vHead} of each head's kv columns`, oDim,
        { heads: kvHeads, rows, ...(latentCache || vPadded ? {} : { kind: kvKind, note: kvNote('') }) });
      op('split', [kv], v, `The V part of each head's ${qkNopeDim + vHead} kv columns: a view of kv, no copy.`, ASSEMBLE);
      if (vPadded) {
        const unpadded = v;
        v = act('v_pad', `V zero-padded to ${cfg.headDim} per head`, kvDim, {
          heads: kvHeads, rowGap: STACK_GAP, kind: kvKind,
          note: kvNote(` ${cfg.headDim - vHead} of each head's ${cfg.headDim} columns are zeros, cached too: a third of the V cache holds nothing.`),
        });
        op('pad', [unpadded], v, `F.pad appends ${cfg.headDim - vHead} zeros to each head's ${vHead}-dim V: FlashAttention needs one head width for Q, K and V. ` +
          'The cache stores the padding too. Eager attention skips this.', PAD);
      }
      vCached = latentCache ? v : cacheWrite(v, 'v_cache', `past_key_values.layers[${l}].values`, 'V', vCachedWidth);
      next();
    } else {
      const wq = weight('self_attn.q_proj.weight', dModel, qDim, undefined, { heads: along(qHeads, 'z'), role: 'q' });
      const qRaw = act('q_raw', 'x_norm · W_q (before RoPE)', qDim, { heads: qHeads });
      matmul(xNorm, wq, qRaw, dModel, qDim);
      next();
      // QK-norm (Gemma 3): each head's q and k normalized before RoPE, which bounds the scores without soft-capping.
      const headFeat: Dim = { name: 'head_dim', size: cfg.headDim };
      const qkNorm = (raw: string, which: 'q' | 'k', dim: Dim, heads: HeadSplit) => {
        if (!cfg.qkNorm) return raw;
        const normed = rmsNorm(raw, `self_attn.${which}_norm.weight`, `${which}_normed`, `${which}_norm(${which.toUpperCase()}): RMSNorm per head`, dim, { heads, group: headFeat });
        next();
        return normed;
      };
      const qn = cfg.qkNorm ? 'q_norm(x_norm · W_q)' : 'x_norm · W_q';
      const kn = cfg.qkNorm ? 'k_norm(x_norm · W_k)' : 'x_norm · W_k';
      const qIn = qkNorm(qRaw, 'q', qDim, qHeads);
      q = act('q', `Q = RoPE(${qn})`, qDim, { heads: qHeads });
      ropeOp(qIn, q, cfg.nHeads);
      next();
      const wk = weight('self_attn.k_proj.weight', dModel, kvDim, gqaNote, { heads: along(kvHeads, 'z'), role: 'k' });
      const kRaw = act('k_raw', 'x_norm · W_k (before RoPE)', kvDim, { heads: kvHeads });
      matmul(xNorm, wk, kRaw, dModel, kvDim);
      next();
      const kIn = qkNorm(kRaw, 'k', kvDim, kvHeads);
      const k = act('k', `K = RoPE(${kn})`, kvDim, { kind: kvKind, note: kvNote(' Cached after rotation, so the cache already encodes position.'), heads: kvHeads });
      ropeOp(kIn, k, cfg.nKvHeads);
      kCached = cacheWrite(k, 'k_cache', `past_key_values.layers[${l}].keys`, 'K');
      next();
      const wv = weight('self_attn.v_proj.weight', dModel, kvDim, gqaNote, { heads: along(kvHeads, 'z'), role: 'v' });
      const v = act('v', 'V = x_norm · W_v', kvDim, { kind: kvKind, note: kvNote(''), heads: kvHeads });
      matmul(xNorm, wv, v, dModel, kvDim);
      vCached = cacheWrite(v, 'v_cache', `past_key_values.layers[${l}].values`, 'V');
      next();
    }
    // Attention reads K and V from the cache.
    // A sliding layer's static cache has only the window's slots; once full, or past a long prompt, none are empty.
    const slotCount = sliding ? Math.min(window, cap) : cap;
    const staticNote = isStatic && (!sliding || attnKeys < slotCount)
      ? ` With a static cache HF attends over all ${formatInt(slotCount)} slots, masking the empty ones; drawn here for the ${formatInt(attnKeys)} filled slots only.`
      : '';
    // QKᵀ and ·V are 2 FLOPs per multiply-add each. In prefill the causal mask lets fused kernels skip
    // key > query; a decode query attends to every cached key.
    const qkFlops = 2 * n * attnKeys * cfg.nHeads * cfg.headDim;
    const pvFlops = 2 * n * attnKeys * cfg.nHeads * (vPadded ? cfg.headDim : vHeadDim(cfg));
    const attnFlops = qkFlops + pvFlops;
    // YaRN raises the softmax temperature to match the stretched positions.
    const scoreScale = yarnScoreScale(cfg);
    const scaleNote = scoreScale === 1 ? '' : ` YaRN multiplies the scores by ${scoreScale.toFixed(3)} (mscale²).`;
    const usefulFlops = decode ? attnFlops : (attnFlops * keptEntries(n, window)) / (n * n);
    /** A prefill tile of a sliding layer whose window is shorter than the prompt: the mask cuts a band. */
    const banded = sliding && !decode && window < n;
    let attn: string;
    if (o.attention === 'eager') {
      // HF eager_attention_forward: every step below allocates a new tensor.
      // repeat_kv copies K and V to line up with the query heads; with multi-head attention there is nothing to
      // repeat, and HF returns them unchanged.
      const perGroup = cfg.nHeads / cfg.nKvHeads;
      let kRep = kCached;
      let vRep = vCached;
      if (perGroup > 1) {
        const repNote = `${TRANSIENT} Each KV head copied ${perGroup}× to line up with the query heads.`;
        kRep = act('k_rep', 'repeat_kv(K)', qDim, { note: repNote, rows: keyRows, heads: qHeads });
        op('repeat_kv', [kCached], kRep, `Copies each of the ${cfg.nKvHeads} KV heads ${perGroup}×: GQA saves cache memory, but eager attention spends it again on temporaries.`, layerScores);
        next();
        vRep = act('v_rep', 'repeat_kv(V)', qDim, { note: repNote, rows: keyRows, heads: qHeads });
        op('repeat_kv', [vCached], vRep, `Copies each of the ${cfg.nKvHeads} KV heads ${perGroup}×.`, layerScores);
        next();
      }
      // One tile per head: keys along X, queries along Z. Prefill tiles are square and causally masked.
      // The steps from raw scores to probabilities stack in one station, like an expanded RMSNorm.
      const scoreGrid = (id: string, label: string, dtype: DType, note: string, causal: boolean) => push({
        id, label, kind: 'scores', x: attnKeys, z: n, tiles: cfg.nHeads, causal: causal && !decode, dtype, rowGap: STACK_GAP,
        heads: along(qHeads, 'tiles'), ...(causal && banded ? { window } : {}),
        dims: [{ name: 'heads', size: cfg.nHeads }, { name: 'query', size: n }, { name: 'key', size: attnKeys }],
        note: `${note} One ${formatInt(n)} × ${formatInt(attnKeys)} tile per head; keys run along X, queries along Z.`,
      });
      const scores = scoreGrid('scores', 'QKᵀ/√d', o.actDtype,
        decode ? `One row of scores per head: the new token against every ${sliding ? 'key in its window' : 'cached key'}.`
          : `Raw scores, computed in full: including the ${banded ? 'entries' : 'half'} the mask is about to discard.`, false);
      op('QKᵀ/√d', [q, kRep], scores,
        `Per head: Q Kᵀ scaled by 1/√${cfg.headDim}.${scaleNote} ${formatFlops(qkFlops)} FLOPs${decode ? '' : ', half of them for entries the mask discards'}.${staticNote}`,
        { flops: qkFlops, ...layerScores });
      let masked = scores;
      if (!decode) {
        masked = scoreGrid('scores_masked', 'QKᵀ/√d + mask', o.actDtype,
          banded ? `Entries with key > query, or ${window} or more keys back, are now −∞.` : 'Entries with key > query are now −∞.', true);
        op('+ mask', [scores, sliding ? slidingMask! : causalMask!], masked, sliding
          ? `Adds the sliding-window mask: 0 for the last ${window} keys up to the query, −∞ elsewhere.`
          : 'Adds the causal mask: 0 where key ≤ query, −∞ above the diagonal.', layerScores);
      }
      const probs32 = scoreGrid('probs_fp32', 'softmax(…) in fp32', 'fp32', 'Softmax runs in fp32 for stability: twice the bytes of the scores.', true);
      op('softmax', [masked], probs32, decode ? 'Softmax over the cached keys, upcast to fp32.' : 'Row-wise softmax over keys, upcast to fp32. Masked entries become exactly 0.', layerSoftmax);
      const probs = scoreGrid('probs', `softmax(…) cast to ${DTYPES[o.actDtype].label}`, o.actDtype, 'Attention probabilities cast back to the activation dtype for the · V matmul.', true);
      op(`to ${DTYPES[o.actDtype].label}`, [probs32], probs, 'Casts the probabilities back to the activation dtype: another full copy.', layerSoftmax);
      next();
      attn = act('attn', 'attn = softmax(QKᵀ/√d) · V', oDim, { heads: qHeads });
      op('· V', [probs, vRep], attn, `Per head: probability-weighted sum of V rows. ${formatFlops(pvFlops)} FLOPs.`,
        { flops: pvFlops, viz: attnViz('output') });
    } else {
      // With padded V the kernel writes each head's output padded too; a slice then drops the zeros (a copy).
      const out = vPadded
        ? act('attn_pad', `softmax(QKᵀ/√d) · V, padded to ${cfg.headDim} per head`, qDim, { heads: qHeads })
        : act('attn', 'attn = softmax(QKᵀ/√d) · V', oDim, { heads: qHeads });
      attn = out;
      op('attention', [q, kCached, vCached], out,
        'softmax(QKᵀ/√d)·V per head, fused (FlashAttention / SDPA): no repeat_kv copies, no mask tensor, and the scores never reach memory. ' +
        (decode
          ? sliding
            ? (keys > window
              ? `The new token attends to the last ${formatInt(attnKeys)} keys only, its window: ${formatFlops(attnFlops)} FLOPs, however long the sequence.`
              : `The new token attends to all ${formatInt(keys)} keys, fewer than its window of ${formatInt(window)}: ${formatFlops(attnFlops)} FLOPs.`)
            : `The new token attends to all ${formatInt(keys)} cached keys: ${formatFlops(attnFlops)} FLOPs, reading the whole K and V cache.`
          : sliding
          ? `Causal, sliding window: query i attends to keys i−${window - 1} … i only, so ~${formatFlops(usefulFlops)} FLOPs instead of ${formatFlops(attnFlops)} ` +
            `(FlashAttention's window_size; SDPA instead takes a boolean mask once T reaches ${window}).`
          : `Causal: query i only attends to keys 0..i, so ~${formatFlops(usefulFlops)} FLOPs instead of ${formatFlops(attnFlops)}.`) +
        scaleNote + staticNote + (vPadded ? ` V is padded, so a third of the · V work multiplies zeros.` : ''), { flops: usefulFlops, ...layerScores });
      if (vPadded) {
        attn = act('attn', 'attn = softmax(QKᵀ/√d) · V', oDim, { heads: qHeads, rowGap: STACK_GAP });
        op('slice', [out], attn, `Keeps the first ${vHeadDim(cfg)} of each head's ${cfg.headDim} output columns; reshape copies them into one ${oDim.size}-wide tensor.`, PAD);
      }
    }
    next();
    const wo = weight(gpt2 ? 'attn.c_proj.weight' : 'self_attn.o_proj.weight', oDim, dModel, undefined, { heads: qHeads });
    const bo = gpt2 ? vector('attn.c_proj.bias', dModel) : undefined;
    const oOut = act('o', `o = attn · W_o${bo ? ' + b' : ''}`, dModel);
    matmul(attn, wo, oOut, oDim, dModel, { bias: bo });
    next();
    // Gemma normalizes each sublayer's output too, before it joins the residual stream; its post_attention_layernorm
    // is that norm, where Llama's is the norm before the MLP.
    let oAdd = oOut;
    if (cfg.postNorms) {
      oAdd = norm(oOut, ['post_attention_layernorm', ''], 'o_norm', 'o');
      next();
    }
    const h = act('h', `h = x + ${cfg.postNorms ? 'o_norm' : 'o'}: residual stream`, dModel, { residual: true });
    op('+', [x, oAdd], h, 'Residual add: the attention output is added to the residual stream.');

    // MLP section, behind the attention section.
    nextSection();
    const hNorm = norm(h, [cfg.postNorms ? 'pre_feedforward_layernorm' : 'post_attention_layernorm', 'ln_2'], 'h_norm', 'h');
    next();
    if (isMoeLayer(cfg, l)) {
      expertTokens[l] = moe(l, h, hNorm);
      continue;
    }
    if (gpt2) {
      // Two matrices with a GELU between them (4 · d_model wide), each with a bias. Laid out like the SwiGLU MLP:
      // the up projection with its output in front, then the down projection with the GELU output and m.
      const wfc = weight('mlp.c_fc.weight', dModel, ffn);
      const bfc = vector('mlp.c_fc.bias', ffn);
      const hfc = stacked('h_fc', 'h_fc = h_norm · W_fc + b', ffn, o.actDtype, TRANSIENT);
      matmul(hNorm, wfc, hfc, dModel, ffn, { bias: bfc });
      next();
      const wproj = weight('mlp.c_proj.weight', ffn, dModel);
      const bproj = vector('mlp.c_proj.bias', dModel);
      const a = stacked('a', 'a = GELU(h_fc)', ffn, o.actDtype, TRANSIENT);
      op('GELU', [hfc], a,
        'GELU, tanh approximation ("gelu_new"): 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³))). HF writes it in ' +
        'plain PyTorch, so each step (x³, products, tanh, sums) passes a full T × d_ffn temporary through memory; drawn as one op.', { viz: { panel: 'gelu' } });
      const m = stacked('m', 'm = a · W_proj + b', dModel, o.actDtype, TRANSIENT);
      matmul(a, wproj, m, ffn, dModel, { bias: bproj });
      op('+', [m, h], nextFloorX(), "Residual add: h + m is this layer's output and the next floor's input.");
      continue;
    }
    // Laid out like one expert of a mixture of experts (see moe): gate and up projections read the same rows, so their
    // weights lie back to back in one station with g and u stacked in front; then the down projection with a and m.
    const wg = weight('mlp.gate_proj.weight', dModel, ffn);
    const wu = weight('mlp.up_proj.weight', dModel, ffn);
    const g = stacked('g', 'g = h_norm · W_gate', ffn, o.actDtype, TRANSIENT);
    matmul(hNorm, wg, g, dModel, ffn);
    const u = stacked('u', 'u = h_norm · W_up', ffn, o.actDtype, TRANSIENT);
    matmul(hNorm, wu, u, dModel, ffn);
    next();
    const wd = weight('mlp.down_proj.weight', ffn, dModel);
    let a: string;
    // GeGLU (Gemma): gelu_pytorch_tanh, F.gelu(approximate='tanh'), one kernel, in place of SiLU.
    const gelu = cfg.mlpAct === 'gelu_tanh';
    const fn = gelu ? 'GELU' : 'SiLU';
    if (o.elementwise === 'expanded') {
      // HF: act_fn(gate_proj(x)) * up_proj(x): SiLU(g) is its own tensor before the product.
      const sg = stacked(gelu ? 'gelu_g' : 'silu_g', `${fn}(g)`, ffn, o.actDtype, TRANSIENT);
      op(fn, [g], sg, gelu
        ? 'GELU(g), tanh approximation, elementwise: one fused kernel (gelu_pytorch_tanh), still a full T × d_ffn tensor of its own.'
        : 'SiLU(g) = g · sigmoid(g), elementwise: a full T × d_ffn tensor of its own.', SILU(cfg.dFfn, gelu));
      a = stacked('a', `a = ${fn}(g) ⊙ u`, ffn, o.actDtype, TRANSIENT);
      op('⊙', [sg, u], a, `Elementwise product with u: the ${gelu ? 'GeGLU' : 'SwiGLU'} gate.`, GATE(cfg.dFfn, gelu));
    } else {
      a = stacked('a', `a = ${fn}(g) ⊙ u`, ffn, o.actDtype, TRANSIENT);
      op(`${fn} ⊙`, [g, u], a, `${gelu ? 'GeGLU' : 'SwiGLU'}: elementwise ${fn}(g) times u.`, SILU(cfg.dFfn, gelu));
    }
    const m = stacked('m', 'm = a · W_down', dModel, o.actDtype, TRANSIENT);
    matmul(a, wd, m, ffn, dModel);
    let mAdd = m;
    if (cfg.postNorms) {
      next();
      mAdd = norm(m, ['post_feedforward_layernorm', ''], 'm_norm', 'm');
    }
    op('+', [mAdd, h], nextFloorX(), `Residual add: h + ${cfg.postNorms ? 'm_norm' : 'm'} is this layer's output and the next floor's input.`);
  }

  // LM head floor.
  floor = lmHeadFloor(cfg);
  section = ATTENTION_SECTION;
  station = 0;
  prefix = top;
  const xLast = act('x', 'x: residual stream out of the last layer', dModel, { residual: true });
  next();
  const xNorm = norm(xLast, ['norm', 'ln_f'], 'x_norm', 'x');
  next();
  prefix = '';
  // GPT-2 and Gemma tie the LM head to the token embedding: the logits read it on the embedding floor, the same table
  // the lookup at the bottom reads one row of per token. Llama 3 and Mixtral have a second matrix.
  const head = tied ? embed
    : weight('lm_head.weight', dModel, vocab, `Not tied to the embedding: a second ${formatCount(cfg.dModel * cfg.vocabSize)}-parameter matrix.`);
  const logits = push({
    id: 'logits', label: 'logits: last position only', kind: 'activation', x: cfg.vocabSize, z: 1,
    dims: [{ name: 'last token', size: 1 }, vocab], dtype: o.actDtype,
    note: 'Only the last position is needed to sample the next token.',
  });
  op('@', [xNorm, head], logits,
    `Matrix multiply, last position only: [1 × ${cfg.dModel}] @ [${cfg.dModel} × ${cfg.vocabSize}]. ${formatFlops(2 * cfg.dModel * cfg.vocabSize)} FLOPs.` +
    (tied ? ` Tied weights: lm_head.weight is ${gpt2 ? 'transformer.wte' : 'model.embed_tokens'}.weight, read here in full, transposed.` : ''),
    { flops: 2 * cfg.dModel * cfg.vocabSize });

  // Sampling floor (HF generate): not part of the model, it turns the logits into the next token. Vocab-wide strips,
  // stacked, and spaced wide enough that the label of the op under the first strip (to fp32, from the floor below)
  // clears the one above the second in the focus view.
  floor = samplingFloor(cfg);
  section = ATTENTION_SECTION;
  station = 0;
  const SAMPLING_GAP = 12288;
  const strip = (id: string, label: string, note: string) => push({
    id, label, kind: 'activation', x: cfg.vocabSize, z: 1, rowGap: SAMPLING_GAP,
    dims: [{ name: 'last token', size: 1 }, vocab], dtype: 'fp32', note,
  });
  const logits32 = strip('logits_fp32', 'logits in fp32', 'generate() copies the last logits to fp32 before any processing.');
  // The copy belongs to the LM head floor, like a layer's last add: it sends the logits up to this floor.
  const stage = (s: VizStage<'sampling'>): OpExtra => ({ viz: { panel: 'sampling', stage: s } });
  floor = lmHeadFloor(cfg);
  op('to fp32', [logits], logits32, 'Upcast copy of the last position\'s logits, handed to generate() for sampling.', stage('logits'));
  floor = samplingFloor(cfg);
  const scaled = strip('scaled', 'logits / temperature', 'Temperature < 1 sharpens the distribution, > 1 flattens it.');
  op('÷ temperature', [logits32], scaled, 'Divides every logit by the temperature.', stage('temperature'));
  const filtered = strip('filtered', 'after top-k / top-p', 'Tokens outside the kept set are −∞.');
  op('top-k / top-p', [scaled], filtered,
    `Keeps the k most likely tokens, then the smallest set whose probability reaches p; the rest become −∞. ` +
    `Top-p sorts all ${formatInt(cfg.vocabSize)} logits, allocating sorted values and int64 indices as further temporaries.`, stage('filter'));
  const probs = strip('probs', 'probabilities', `softmax over the vocabulary: ${formatInt(cfg.vocabSize)} probabilities summing to 1, mostly exactly 0 after filtering.`);
  op('softmax', [filtered], probs, 'Softmax over the vocabulary.', stage('softmax'));
  const nextToken = push({
    id: 'next_token', label: 'next token id', kind: 'activation', x: 1, z: 1, rowGap: SAMPLING_GAP, centered: true,
    dims: [{ name: 'token', size: 1 }], dtype: 'int64',
    note: 'The output of the whole forward pass: one token id, 8 bytes. In decoding it becomes the next input_ids.',
  });
  op('sample', [probs], nextToken, 'Draws one token id from the probabilities (torch.multinomial); greedy decoding takes the argmax instead.', stage('sample'));

  return { tensors, ops, loop: { input: ids, output: nextToken }, expertTokens: cfg.moe ? expertTokens : undefined };
}

/** `v` rounded to bfloat16: 8 significant bits. */
function bf16(v: number): string {
  const step = 2 ** (Math.floor(Math.log2(Math.abs(v))) - 7);
  return String(Math.round(v / step) * step);
}

function formatTokens(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toPrecision(2)}M` : n >= 1e3 ? `${(n / 1e3).toPrecision(2)}K` : n.toFixed(0);
}
