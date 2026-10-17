import { byteCount, elementCount, type TensorSpec } from './tensors';

export interface Stats {
  params: number;
  paramBytes: number;
  kvElements: number;
  kvBytes: number;
  /** Transient activations of one decoder layer, excluding attention scores. */
  layerActBytes: number;
  scoresBytes: number;
}

export function summarize(specs: TensorSpec[]): Stats {
  const sum = (pred: (s: TensorSpec) => boolean, f: (s: TensorSpec) => number) =>
    specs.filter(pred).reduce((acc, s) => acc + f(s), 0);
  return {
    params: sum((s) => s.kind === 'weight', elementCount),
    paramBytes: sum((s) => s.kind === 'weight', byteCount),
    kvElements: sum((s) => s.kind === 'kv', elementCount),
    kvBytes: sum((s) => s.kind === 'kv', byteCount),
    layerActBytes: sum((s) => s.floor === 1 && s.kind === 'activation', byteCount),
    scoresBytes: sum((s) => s.floor === 1 && s.kind === 'scores', byteCount),
  };
}
