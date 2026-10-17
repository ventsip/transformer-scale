import { DTYPES } from './model/dtypes';
import type { TensorSpec } from './model/tensors';

/**
 * Floor plan: 1 world unit = 1 element along X and Z.
 * A floor has one or two sections (attention in front, MLP behind). Each section lays its
 * stations out left to right in data-flow order around a "bus" line: weights sit behind it (-Z),
 * the activations they produce in front of it (+Z). Operations sit on the bus.
 * Thickness (Y) is on its own, exaggerated scale: bytes per element.
 */
export const STATION_GAP = 2048;
const ROW_GAP = 1024;
const ITEM_GAP = 512;
const SECTION_GAP = 3072;
const TILE_COLS = 8;
export const THICKNESS_PER_BYTE = 64;

/** Extra lift for floors above the focused one, so the camera can look down onto it. */
const FOCUS_GAP = 40000;

export type ThicknessMode = 'bytes' | 'elements';

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
  /** Z of the bus line of this tensor's section. */
  busZ: number;
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

const groupBy = <T, K>(items: T[], key: (t: T) => K) => {
  const map = new Map<K, T[]>();
  for (const t of items) map.set(key(t), [...(map.get(key(t)) ?? []), t]);
  return [...map.entries()].sort(([a], [b]) => Number(a) - Number(b));
};

export function layoutTensors(specs: TensorSpec[], o: LayoutOptions): PlacedTensor[] {
  const placed: PlacedTensor[] = [];
  for (const [floor, floorSpecs] of groupBy(specs, (s) => s.floor)) {
    const y = floorY(floor, o);
    // Back edge of the previous section; the next section's front edge goes behind it.
    let behind = 0;
    for (const [section, sectionSpecs] of groupBy(floorSpecs, (s) => s.section)) {
      const local: PlacedTensor[] = [];
      let minZ = 0;
      let maxZ = 0;
      let x0 = 0;
      for (const [, items] of groupBy(sectionSpecs, (s) => s.station)) {
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
          minZ = Math.min(minZ, zMin);
          maxZ = Math.max(maxZ, zMin + t.depth);
          const boxes: Box[] = [];
          for (let i = 0; i < s.tiles; i++) {
            const minX = x0 + (i % t.cols) * (s.x + t.gap);
            const tileZ = zMin + Math.floor(i / t.cols) * (s.z + t.gap);
            boxes.push({ cx: minX + s.x / 2, cy: y + h / 2, cz: tileZ + s.z / 2, sx: s.x, sy: h, sz: s.z });
          }
          local.push({ spec: s, boxes, busZ: 0 });
        }
        x0 += width + STATION_GAP;
      }
      const shift = section === 0 ? 0 : behind - SECTION_GAP - maxZ;
      for (const p of local) {
        p.busZ = shift;
        for (const b of p.boxes) b.cz += shift;
      }
      behind = shift + minZ;
      placed.push(...local);
    }
  }
  return placed;
}
