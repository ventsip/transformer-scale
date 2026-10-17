import { DTYPES } from './model/dtypes';
import type { TensorSpec } from './model/tensors';

/**
 * Floor plan: 1 world unit = 1 element along X and Z.
 * Each floor lays its stations out left to right in data-flow order. Weights sit in the
 * back row (-Z), the activations they produce in the front row (+Z), facing the camera.
 * Thickness (Y) is on its own, exaggerated scale: bytes per element.
 */
export const STATION_GAP = 2048;
const ROW_GAP = 1024;
const ITEM_GAP = 512;
const TILE_COLS = 8;
export const THICKNESS_PER_BYTE = 64;

export type ThicknessMode = 'bytes' | 'elements';

/** Extra lift for floors above the focused one, so the camera can look down onto it. */
const FOCUS_GAP = 30000;

export interface LayoutOptions {
  floorSpacing: number;
  thickness: ThicknessMode;
  focusFloor: number;
}

export interface Box {
  cx: number;
  cy: number;
  cz: number;
  sx: number;
  sy: number;
  sz: number;
}

export interface PlacedTensor {
  spec: TensorSpec;
  boxes: Box[];
}

function tiling(s: TensorSpec) {
  const cols = Math.min(s.tiles, TILE_COLS);
  const rows = Math.ceil(s.tiles / cols);
  const gap = s.tiles > 1 ? Math.max(4, Math.round(s.x / 8)) : 0;
  return { cols, gap, width: cols * s.x + (cols - 1) * gap, depth: rows * s.z + (rows - 1) * gap };
}

function thickness(s: TensorSpec, mode: ThicknessMode): number {
  const bytes = mode === 'bytes' ? DTYPES[s.dtype].bytes : DTYPES.bf16.bytes;
  return bytes * THICKNESS_PER_BYTE;
}

export const floorY = (floor: number, o: LayoutOptions) =>
  floor * o.floorSpacing + (floor > o.focusFloor ? FOCUS_GAP : 0);

export function layoutTensors(specs: TensorSpec[], o: LayoutOptions): PlacedTensor[] {
  const byFloor = new Map<number, Map<number, TensorSpec[]>>();
  for (const s of specs) {
    const stations = byFloor.get(s.floor) ?? new Map<number, TensorSpec[]>();
    byFloor.set(s.floor, stations);
    stations.set(s.station, [...(stations.get(s.station) ?? []), s]);
  }

  const placed: PlacedTensor[] = [];
  for (const [floor, stations] of byFloor) {
    const y = floorY(floor, o);
    let x0 = 0;
    for (const station of [...stations.keys()].sort((a, b) => a - b)) {
      const items = stations.get(station)!;
      let back = ROW_GAP / 2;
      let front = ROW_GAP / 2;
      let width = 0;
      for (const s of items) {
        const t = tiling(s);
        const h = thickness(s, o.thickness);
        width = Math.max(width, t.width);
        let zMin: number;
        if (s.kind === 'weight') {
          zMin = -(back + t.depth);
          back += t.depth + ITEM_GAP;
        } else {
          zMin = front;
          front += t.depth + ITEM_GAP;
        }
        const boxes: Box[] = [];
        for (let i = 0; i < s.tiles; i++) {
          const minX = x0 + (i % t.cols) * (s.x + t.gap);
          const minZ = zMin + Math.floor(i / t.cols) * (s.z + t.gap);
          boxes.push({ cx: minX + s.x / 2, cy: y + h / 2, cz: minZ + s.z / 2, sx: s.x, sy: h, sz: s.z });
        }
        placed.push({ spec: s, boxes });
      }
      x0 += width + STATION_GAP;
    }
  }
  return placed;
}
