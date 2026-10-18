import type { ModelConfig } from './config';
import { byteCount, elementCount, isTransient, type Model, type TensorSpec } from './tensors';

export type PartKind = 'attention' | 'mlp' | 'router' | 'experts' | 'expert';

/** A named part of a decoder-layer floor, with the tensors in it and their totals for the current pass. */
export interface Part {
  key: string;
  name: string;
  kind: PartKind;
  /** For one expert of a mixture of experts. */
  expert?: number;
  tensorIds: string[];
  params: number;
  weightBytes: number;
  /** Transient activations and attention scores. */
  activationBytes: number;
  kvBytes: number;
  /** FLOPs of the ops producing tensors in this part. */
  flops: number;
  /** Bytes of weights and KV cache those ops read. */
  bytesRead: number;
  /** An expert no token is routed to in this pass. */
  idle?: boolean;
}

/**
 * The parts of a decoder-layer floor: attention (section 0), then the MLP (section 1) or, in a mixture of experts,
 * the router (section 1, with the norm before it and the sum after the experts), all experts together (sections 2 and
 * up) and each expert on its own. Other floors have no parts.
 */
export function floorParts(cfg: ModelConfig, model: Model, floor: number): Part[] {
  if (floor < 1 || floor > cfg.nLayers) return [];
  const onFloor = model.tensors.filter((s) => s.floor === floor);
  const byId = new Map(model.tensors.map((s) => [s.id, s]));
  const ops = model.ops.filter((op) => op.floor === floor);

  const part = (key: string, name: string, kind: PartKind, tensors: TensorSpec[], expert?: number): Part => {
    const ids = new Set(tensors.map((s) => s.id));
    const weights = tensors.filter((s) => s.kind === 'weight');
    let flops = 0;
    let bytesRead = 0;
    for (const op of ops.filter((o) => ids.has(o.output))) {
      flops += op.flops ?? 0;
      for (const id of op.inputs) {
        const s = byId.get(id);
        if (s && (s.kind === 'weight' || s.kind === 'kv')) bytesRead += op.inputBytes?.[id] ?? byteCount(s);
      }
    }
    return {
      key, name, kind, expert, tensorIds: [...ids], flops, bytesRead,
      params: weights.reduce((a, s) => a + elementCount(s), 0),
      weightBytes: weights.reduce((a, s) => a + byteCount(s), 0),
      activationBytes: tensors.filter(isTransient).reduce((a, s) => a + byteCount(s), 0),
      kvBytes: tensors.filter((s) => s.kind === 'kv' && !s.superseded).reduce((a, s) => a + byteCount(s), 0),
      idle: expert !== undefined ? weights.every((s) => s.idle) : undefined,
    };
  };

  const parts = [part('attention', 'Attention', 'attention', onFloor.filter((s) => s.section === 0))];
  if (!cfg.moe) return [...parts, part('mlp', 'MLP', 'mlp', onFloor.filter((s) => s.section === 1))];
  parts.push(part('router', 'Router', 'router', onFloor.filter((s) => s.section === 1)));
  parts.push(part('experts', 'Experts', 'experts', onFloor.filter((s) => s.section >= 2)));
  for (let e = 0; e < cfg.moe.nExperts; e++) {
    parts.push(part(`expert ${e}`, `E${e}`, 'expert', onFloor.filter((s) => s.expert === e), e));
  }
  return parts;
}
