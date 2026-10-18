import type { ModelConfig } from './config';
import { byteCount, elementCount, type Model, type TensorSpec } from './tensors';

export interface Stats {
  params: number;
  paramBytes: number;
  /** Parameters one token runs through: for a mixture of experts, only its top-k experts' share. */
  activeParams: number;
  kvElements: number;
  kvBytes: number;
  /** Transient activations of one decoder layer, excluding attention scores. */
  layerActBytes: number;
  scoresBytes: number;
  /** Floating-point operations of the whole pass. */
  flops: number;
  /** Bytes of weights, buffers and KV cache the ops read during the pass (activations not counted). */
  bytesRead: number;
}

export function summarize({ tensors, ops }: Model, cfg: ModelConfig): Stats {
  const sum = (pred: (s: TensorSpec) => boolean, f: (s: TensorSpec) => number) =>
    tensors.filter(pred).reduce((acc, s) => acc + f(s), 0);
  const isCache = (s: TensorSpec) => s.kind === 'kv' && !s.superseded;
  const byId = new Map(tensors.map((s) => [s.id, s]));
  let bytesRead = 0;
  for (const op of ops) {
    for (const id of op.inputs) {
      const s = byId.get(id);
      if (s && (s.kind === 'weight' || s.kind === 'buffer' || s.kind === 'kv')) bytesRead += op.inputBytes?.[id] ?? byteCount(s);
    }
  }
  const params = sum((s) => s.kind === 'weight', elementCount);
  const expertParams = sum((s) => s.kind === 'weight' && s.expert !== undefined, elementCount);
  const activeShare = cfg.moe ? cfg.moe.topK / cfg.moe.nExperts : 1;
  return {
    params,
    paramBytes: sum((s) => s.kind === 'weight', byteCount),
    activeParams: params - expertParams * (1 - activeShare),
    kvElements: sum(isCache, elementCount),
    kvBytes: sum(isCache, byteCount),
    layerActBytes: sum((s) => s.floor === 1 && s.kind === 'activation', byteCount),
    scoresBytes: sum((s) => s.floor === 1 && s.kind === 'scores', byteCount),
    flops: ops.reduce((acc, op) => acc + (op.flops ?? 0), 0),
    bytesRead,
  };
}
