import { DTYPES } from './model/dtypes';
import type { Role, TensorSpec } from './model/tensors';

/**
 * Floor plan: 1 world unit = 1 element along X and Z.
 * A floor has one or more sections (attention in front, MLP behind, rows of experts further back). Each section lays its
 * stations out left to right in data-flow order around a "bus" line: weights sit behind it (-Z),
 * the activations they produce in front of it (+Z). Operations sit on the bus.
 * Thickness (Y) is on its own, exaggerated scale: bytes per element.
 */
export const STATION_GAP = 2048;
const ROW_GAP = 1024;
const ITEM_GAP = 512;
/** Gap between a tensor and one placed beside it. */
const BESIDE_GAP = 2048;
const SECTION_GAP = 3072;
/**
 * Wider gaps in front of section 1 (the MLP, behind attention) and section 2 (in a mixture of experts, the expert
 * rows behind the router), so those read as separate parts. Expert rows keep SECTION_GAP between them.
 */
const PART_GAP = 10240;
const TILE_COLS = 8;
export const THICKNESS_PER_BYTE = 64;

/**
 * The focus view looks at the focused floor from the front, rising FOCUS_VIEW_SLOPE units per unit toward the viewer
 * (1: at 45°; see frameFocus in main.ts).
 */
export const FOCUS_VIEW_SLOPE = 1;

/**
 * Floors above the focused one are lifted and floors below it lowered, opening space around it: at least
 * MIN_FOCUS_GAP, and enough for the focused floor to stay clear of the one above in the focus view. The line of
 * sight from the back edge of a floor of depth D rises D · FOCUS_VIEW_SLOPE by its front edge, so that is the gap.
 * Only the focused floor's depth counts: the LM head floor is 128K deep (the vocabulary), which would push every
 * layer's neighbors far away.
 */
const MIN_FOCUS_GAP = 24000;

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

/** Rows of a cache tensor: written by earlier passes, written by this pass, or allocated but not yet written. */
export type RowState = 'written' | 'fresh' | 'reserved';

export interface PlacedTensor {
  spec: TensorSpec;
  boxes: Box[];
  /** Attention head of each box (see `spec.heads`), or null. */
  heads: (number | null)[];
  /** For cache tensors, which rows each box covers (see `spec.filledZ` and `spec.freshZ`). */
  rows: RowState[];
  /** The attention role (Q, K, V projection) of each box, for color coding; null for other tensors. */
  roles: (Role | null)[];
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

const floorY = (floor: number, o: LayoutOptions, gap: number) =>
  floor * o.floorSpacing + Math.sign(floor - o.focusFloor) * gap;

export interface Layout {
  placed: PlacedTensor[];
  /** Height of a floor's base. */
  floorY: (floor: number) => number;
}

const groupBy = <T, K>(items: T[], key: (t: T) => K) => {
  const map = new Map<K, T[]>();
  for (const t of items) map.set(key(t), [...(map.get(key(t)) ?? []), t]);
  return [...map.entries()].sort(([a], [b]) => Number(a) - Number(b));
};

export function layoutTensors(specs: TensorSpec[], o: LayoutOptions): Layout {
  const placed: PlacedTensor[] = [];
  for (const [floor, floorSpecs] of groupBy(specs, (s) => s.floor)) {
    // Placed without the focus gap first: it depends on the floors' depths.
    const y = floorY(floor, o, 0);
    // Back edge of the previous section; the next section's front edge goes behind it.
    let behind = 0;
    for (const [section, sectionSpecs] of groupBy(floorSpecs, (s) => s.section)) {
      const local: PlacedTensor[] = [];
      let minZ = 0;
      let maxZ = 0;
      let x0 = 0;
      for (const [, items] of groupBy(sectionSpecs, (s) => s.station)) {
        x0 += Math.max(0, ...items.map((s) => s.gapBefore ?? 0));
        let back = ROW_GAP / 2;
        let front = ROW_GAP / 2;
        let width = 0;
        let stackedFront = false;
        // The previous front-row tensor, for one placed beside it.
        let last = { zMin: 0, xEnd: 0 };
        const stationWidth = Math.max(...items.map((s) => tiling(s).width));
        for (const s of items) {
          const t = tiling(s);
          const h = thickness(s, o.thickness);
          let zMin: number;
          let xOffset = 0;
          if (s.kind === 'weight') {
            zMin = -(back + t.depth);
            back += t.depth + ITEM_GAP;
          } else if (s.beside && stackedFront) {
            // Same row as the previous tensor, to its right.
            zMin = last.zMin;
            xOffset = last.xEnd + BESIDE_GAP;
            front = Math.max(front, zMin + t.depth);
          } else {
            const gap = stackedFront ? (s.rowGap ?? ITEM_GAP) : 0;
            zMin = front + gap;
            front = zMin + t.depth;
            stackedFront = true;
            // Centered tensors line up under the middle of the station (and so do the ops hovering over them).
            if (s.centered) xOffset = (stationWidth - t.width) / 2;
          }
          if (s.kind !== 'weight') last = { zMin, xEnd: xOffset + t.width };
          width = Math.max(width, xOffset + t.width, s.reserveWidth ?? 0);
          minZ = Math.min(minZ, zMin);
          maxZ = Math.max(maxZ, zMin + t.depth, ROW_GAP / 2 + (s.reserveFront ?? 0));
          const boxes: Box[] = [];
          const heads: (number | null)[] = [];
          for (let i = 0; i < s.tiles; i++) {
            const minX = x0 + xOffset + (i % t.cols) * (s.x + t.gap);
            const tileZ = zMin + Math.floor(i / t.cols) * (s.z + t.gap);
            const split = s.heads && s.heads.axis !== 'tiles' ? s.heads : null;
            if (!split) {
              boxes.push({ cx: minX + s.x / 2, cy: y + h / 2, cz: tileZ + s.z / 2, sx: s.x, sy: h, sz: s.z });
              heads.push(s.heads?.axis === 'tiles' ? i : null);
              continue;
            }
            // One band per head, no gaps: the tensor keeps its exact footprint.
            const bx = split.axis === 'x' ? s.x / split.count : s.x;
            const bz = split.axis === 'z' ? s.z / split.count : s.z;
            for (let hd = 0; hd < split.count; hd++) {
              const ox = split.axis === 'x' ? hd * bx : 0;
              const oz = split.axis === 'z' ? hd * bz : 0;
              boxes.push({ cx: minX + ox + bx / 2, cy: y + h / 2, cz: tileZ + oz + bz / 2, sx: bx, sy: h, sz: bz });
              heads.push(hd);
            }
          }
          let rows = boxes.map((): RowState => 'written');
          if (s.filledZ !== undefined) {
            // Cache tensors: split every box along Z into rows written earlier, rows this pass wrote, and rows reserved.
            const fresh = s.freshZ ?? 0;
            const cuts: [RowState, number, number][] = (
              [['written', 0, s.filledZ - fresh], ['fresh', s.filledZ - fresh, s.filledZ], ['reserved', s.filledZ, s.z]] as [RowState, number, number][]
            ).filter(([, a, b]) => b > a);
            const split: Box[] = [];
            const splitHeads: (number | null)[] = [];
            const splitRows: RowState[] = [];
            boxes.forEach((b, i) => {
              const start = b.cz - b.sz / 2;
              for (const [state, a, end] of cuts) {
                split.push({ ...b, cz: start + (a + end) / 2, sz: end - a });
                splitHeads.push(heads[i]);
                splitRows.push(state);
              }
            });
            boxes.splice(0, boxes.length, ...split);
            heads.splice(0, heads.length, ...splitHeads);
            rows = splitRows;
          }
          // Each box's attention role: the projection's own, or for a fused Q/K/V weight or bias, the third it covers.
          let roles = boxes.map(() => s.role ?? null);
          if (s.fusedRoles) {
            const thirds = boxes.flatMap((b) => [0, 1, 2].map((k) => (s.fusedRoles === 'z'
              ? { ...b, cz: b.cz - b.sz / 2 + (k + 0.5) * (b.sz / 3), sz: b.sz / 3 }
              : { ...b, cx: b.cx - b.sx / 2 + (k + 0.5) * (b.sx / 3), sx: b.sx / 3 })));
            roles = boxes.flatMap(() => ['q', 'k', 'v'] as const);
            heads.splice(0, heads.length, ...thirds.map(() => null));
            rows = thirds.map((): RowState => 'written');
            boxes.splice(0, boxes.length, ...thirds);
          }
          local.push({ spec: s, boxes, heads, rows, roles, busZ: 0 });
        }
        x0 += width + STATION_GAP;
      }
      const shift = section === 0 ? 0 : behind - (section <= 2 ? PART_GAP : SECTION_GAP) - maxZ;
      for (const p of local) {
        p.busZ = shift;
        for (const b of p.boxes) b.cz += shift;
      }
      behind = shift + minZ;
      placed.push(...local);
    }
  }
  const extent = new Map<number, { min: number; max: number }>();
  for (const p of placed) {
    const e = extent.get(p.spec.floor) ?? { min: Infinity, max: -Infinity };
    for (const b of p.boxes) {
      e.min = Math.min(e.min, b.cz - b.sz / 2);
      e.max = Math.max(e.max, b.cz + b.sz / 2);
    }
    extent.set(p.spec.floor, e);
  }
  const focused = extent.get(o.focusFloor);
  const depth = focused ? focused.max - focused.min : 0;
  const gap = Math.max(MIN_FOCUS_GAP, FOCUS_VIEW_SLOPE * depth);
  for (const p of placed) {
    const dy = Math.sign(p.spec.floor - o.focusFloor) * gap;
    for (const b of p.boxes) b.cy += dy;
  }
  return { placed, floorY: (floor) => floorY(floor, o, gap) };
}
