import * as THREE from 'three/webgpu';
import { boxBounds, type Box, type PlacedTensor, type RowState } from '../layout';
import { kvGroup, onPath, type HeadSelection, type Role, type TensorKind } from '../model/tensors';
import { PALETTE } from '../palette';

export const KIND_COLORS: Record<TensorKind, { fill: number; edge: number }> = {
  weight: { fill: PALETTE.weight, edge: PALETTE.weightEdge },
  activation: { fill: PALETTE.activation, edge: PALETTE.activationEdge },
  kv: { fill: PALETTE.kv, edge: PALETTE.kvEdge },
  scores: { fill: PALETTE.scores, edge: PALETTE.scoresEdge },
  buffer: { fill: PALETTE.buffer, edge: PALETTE.bufferEdge },
};

/** Color of the causally masked half of attention-score tiles. */
export const MASKED_COLORS = { fill: PALETTE.neutral500, edge: PALETTE.neutral600 };

/** Colors of the Q, K and V projection weights (tint and outline). Hues used for nothing else. */
export const ROLE_COLORS = { q: PALETTE.q, k: PALETTE.k, v: PALETTE.v } as const;

/**
 * Which part of a box an instance covers. Causal tiles split along the diagonal: keys run along X and
 * queries along Z, so the causal mask keeps the half where key ≤ query and masks the rest. A sliding window keeps
 * only a band along the diagonal: the corner below it, keys too far back, is `outside`.
 */
export type Part = 'box' | 'kept' | 'masked' | 'outside';

export interface Pick {
  item: PlacedTensor;
  box: Box;
  head: number | null;
  /** For cache tensors: rows written before, by this pass, or not yet. */
  rows: RowState;
  /** For Q, K and V projection weights (or a third of a fused one): which. */
  role: Role | null;
  /** Mixture of experts: the expert this box belongs to (a router column, or an expert's tensor). */
  expert: number | null;
  part: Part;
}

/** Mix of a group hue into a kind color: enough to match heads across tensors, not to hide the kind. */
const GROUP_TINT = 0.3;
/** Mix of the role color into the Q, K and V projection weights (before the group tint). */
const ROLE_TINT = 0.5;
/** Mix of an expert's hue into its tensors and its router column. */
const EXPERT_TINT = 0.35;
/** Brightness of head bands that are not on the selected head's path. */
const OFF_PATH = 0.2;
/** Brightness of the reserved, not yet written part of a preallocated buffer. */
const RESERVED = 0.3;
/** Mix toward white for rows written by this pass (the new token's K or V). */
const FRESH = 0.6;
/** Brightness of a tensor this pass replaces, e.g. a dynamic cache before torch.cat. */
const SUPERSEDED = 0.45;
/** Brightness of the weights of experts no token is routed to. */
const IDLE = 0.3;
/** Brightness of floors other than the focused one. */
const OTHER_FLOOR = 0.15;

// Outlines in unit-box coordinates, as pairs of [x, y, z] endpoints.
type Segment = [number, number, number];
const BOX_CORNERS = [0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 1, 3, 4, 6, 5, 7, 0, 4, 1, 5, 2, 6, 3, 7]; // bits: 1 = +x, 2 = +y, 4 = +z
const BOX_OUTLINE: Segment[] = BOX_CORNERS.map((c) => [c & 1 ? 0.5 : -0.5, c & 2 ? 0.5 : -0.5, c & 4 ? 0.5 : -0.5]);

/** Triangle corners in XZ: key ≤ query is the kept half, key > query the masked half. */
const TRIANGLES: Record<'kept' | 'masked', [number, number][]> = {
  kept: [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]],
  masked: [[-0.5, -0.5], [0.5, 0.5], [0.5, -0.5]],
};

/**
 * A sliding window keeping the last `w` (a fraction of the keys) keys of each query: the band along the diagonal it
 * keeps, and the corner below it.
 */
function bandPolygons(w: number): { kept: [number, number][]; outside: [number, number][] } {
  return {
    kept: [[-0.5, -0.5], [-0.5, w - 0.5], [0.5 - w, 0.5], [0.5, 0.5]],
    outside: [[-0.5, w - 0.5], [-0.5, 0.5], [0.5 - w, 0.5]],
  };
}

/** Outline of a prism over a convex polygon in XZ. */
function prismOutline(poly: [number, number][]): Segment[] {
  const out: Segment[] = [];
  for (let i = 0; i < poly.length; i++) {
    const [ax, az] = poly[i];
    const [bx, bz] = poly[(i + 1) % poly.length];
    out.push([ax, -0.5, az], [bx, -0.5, bz], [ax, 0.5, az], [bx, 0.5, bz], [ax, -0.5, az], [ax, 0.5, az]);
  }
  return out;
}

/** A prism over a convex polygon in XZ: both caps as fans, and a quad per side. */
function prismGeometry(poly: [number, number][]): THREE.BufferGeometry {
  const corners = poly.map(([x, z]) => [new THREE.Vector3(x, -0.5, z), new THREE.Vector3(x, 0.5, z)]);
  const caps = corners.slice(2).flatMap((c, i) => {
    const [a, b] = [corners[0], corners[i + 1]];
    return [a[1], b[1], c[1], a[0], c[0], b[0]];
  });
  const quads = corners.flatMap((p, i) => {
    const q = corners[(i + 1) % corners.length];
    return [p[0], q[0], q[1], p[0], q[1], p[1]];
  });
  const geometry = new THREE.BufferGeometry().setFromPoints([...caps, ...quads]);
  geometry.computeVertexNormals();
  return geometry;
}

const OUTLINES: Record<Exclude<Part, 'outside'>, Segment[]> = {
  box: BOX_OUTLINE,
  kept: prismOutline(TRIANGLES.kept),
  masked: prismOutline(TRIANGLES.masked),
};

/** The window of a causal tile as a fraction of its keys, when it is shorter than them; else null. */
const windowFraction = ({ spec }: PlacedTensor) => (spec.window !== undefined && spec.window < spec.x ? spec.window / spec.x : null);

/** Renders placed tensors: instanced boxes, prism halves for causally masked tiles, and 1px outlines for all. */
export class TensorScene {
  readonly group = new THREE.Group();
  private readonly unitBox = new THREE.BoxGeometry(1, 1, 1);
  private readonly prisms = { kept: prismGeometry(TRIANGLES.kept), masked: prismGeometry(TRIANGLES.masked) };
  /** Sliding-window bands and corners, by window fraction: one size per pass, built on demand. */
  private readonly bands = new Map<number, { kept: THREE.BufferGeometry; outside: THREE.BufferGeometry; outlines: Record<'kept' | 'outside', Segment[]> }>();
  private readonly solidMaterial = new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0 });
  // Prism faces are wound inconsistently; drawing both sides keeps them solid from every angle.
  private readonly prismMaterial = new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0, side: THREE.DoubleSide });
  private readonly edgeMaterial = new THREE.LineBasicMaterial({ vertexColors: true });
  private meshes: { mesh: THREE.InstancedMesh; picks: Pick[] }[] = [];
  private edges?: THREE.LineSegments;
  /** Outlines of hovered boxes: one for a block, the tensor at the end of a pipe. Grown on demand. */
  private readonly highlights: THREE.LineSegments[] = [];
  private readonly highlightGeometry: THREE.EdgesGeometry;
  private readonly highlightMaterial = new THREE.LineBasicMaterial({ color: PALETTE.highlight, depthTest: false, transparent: true });
  private focusFloor = -1;
  private selection: HeadSelection | null = null;
  private expertSelection: number | null = null;
  private readonly groupColor = new THREE.Color();

  constructor() {
    this.highlightGeometry = new THREE.EdgesGeometry(this.unitBox);
  }

  /** Floors other than `focusFloor` are drawn dimmed, and so are head bands off the selected head's path. */
  update(items: PlacedTensor[], focusFloor: number, selection: HeadSelection | null, expertSelection: number | null = null): void {
    this.clear();
    this.focusFloor = focusFloor;
    this.selection = selection;
    this.expertSelection = expertSelection;
    const solid: Pick[] = [];
    const kept: Pick[] = [];
    const masked: Pick[] = [];
    /** Sliding-window tiles, by window fraction: their bands and the corners outside them. */
    const banded = new Map<number, { kept: Pick[]; outside: Pick[] }>();
    for (const item of items) {
      const w = item.spec.causal ? windowFraction(item) : null;
      item.boxes.forEach((box, i) => {
        const head = item.heads[i];
        const rows = item.rows[i];
        const role = item.roles[i];
        const expert = item.experts[i];
        const pick = (part: Part): Pick => ({ item, box, head, rows, role, expert, part });
        if (!item.spec.causal) return solid.push(pick('box'));
        masked.push(pick('masked'));
        if (w === null) return kept.push(pick('kept'));
        const band = banded.get(w) ?? { kept: [], outside: [] };
        banded.set(w, band);
        band.kept.push(pick('kept'));
        band.outside.push(pick('outside'));
      });
    }
    this.addInstanced(solid, this.unitBox, this.solidMaterial);
    this.addInstanced(kept, this.prisms.kept, this.prismMaterial);
    this.addInstanced(masked, this.prisms.masked, this.prismMaterial);
    for (const [w, band] of banded) {
      const shapes = this.band(w);
      this.addInstanced(band.kept, shapes.kept, this.prismMaterial);
      this.addInstanced(band.outside, shapes.outside, this.prismMaterial);
    }
    this.addEdges([...solid, ...kept, ...masked, ...[...banded.values()].flatMap((b) => [...b.kept, ...b.outside])]);
    // Bands of earlier passes (another prompt length) are no longer drawn.
    for (const [w, shapes] of this.bands) {
      if (banded.has(w)) continue;
      shapes.kept.dispose();
      shapes.outside.dispose();
      this.bands.delete(w);
    }
  }

  /** The nearest box under the ray, with its distance from the ray's origin. */
  pick(raycaster: THREE.Raycaster): (Pick & { distance: number }) | null {
    const hits = raycaster.intersectObjects(this.meshes.map((m) => m.mesh), false);
    for (const hit of hits) {
      const entry = this.meshes.find((m) => m.mesh === hit.object);
      if (entry && hit.instanceId !== undefined) return { ...entry.picks[hit.instanceId], distance: hit.distance };
    }
    return null;
  }

  /** Outlines `boxes` in white, drawn over everything; an empty list clears the highlight. */
  setHighlight(boxes: Box[]): void {
    while (this.highlights.length < boxes.length) {
      const outline = new THREE.LineSegments(this.highlightGeometry, this.highlightMaterial);
      outline.renderOrder = 10;
      this.group.add(outline);
      this.highlights.push(outline);
    }
    this.highlights.forEach((outline, i) => {
      const box = boxes[i];
      outline.visible = box !== undefined;
      if (!box) return;
      outline.position.set(box.cx, box.cy, box.cz);
      outline.scale.set(box.sx, Math.max(box.sy, 1), box.sz);
    });
  }

  /** The geometries and outlines of a sliding window keeping fraction `w` of the keys. */
  private band(w: number) {
    let shapes = this.bands.get(w);
    if (!shapes) {
      const poly = bandPolygons(w);
      shapes = {
        kept: prismGeometry(poly.kept), outside: prismGeometry(poly.outside),
        outlines: { kept: prismOutline(poly.kept), outside: prismOutline(poly.outside) },
      };
      this.bands.set(w, shapes);
    }
    return shapes;
  }

  private outlineOf({ item, part }: Pick): Segment[] {
    const w = part === 'kept' || part === 'outside' ? windowFraction(item) : null;
    return w === null ? OUTLINES[part as Exclude<Part, 'outside'>] : this.band(w).outlines[part as 'kept' | 'outside'];
  }

  private colorOf({ item, head, part, rows, role, expert }: Pick, which: 'fill' | 'edge', out: THREE.Color): THREE.Color {
    const isMasked = part === 'masked' || part === 'outside';
    out.setHex(isMasked ? MASKED_COLORS[which] : KIND_COLORS[item.spec.kind][which]);
    // Q, K and V projection weights: blended toward their role color, under the KV-group tint.
    if (role) out.lerp(this.groupColor.setHex(ROLE_COLORS[role]), ROLE_TINT);
    // Mixture of experts: each expert's tensors and its router column share a light tint; a selected expert dims the rest.
    if (expert !== null) {
      out.lerp(this.groupColor.setHSL(expertHue(expert), 0.7, 0.55), EXPERT_TINT);
      if (this.expertSelection !== null && expert !== this.expertSelection) out.multiplyScalar(OFF_PATH);
    }
    const split = item.spec.heads;
    if (split && head !== null) {
      const groups = split.count / split.perGroup;
      // Step hues by 3 slots around the wheel so neighboring groups contrast instead of forming a gradient.
      const hue = ((kvGroup(split, head) * 3) % groups) / groups;
      if (!isMasked) out.lerp(this.groupColor.setHSL(hue, 0.7, 0.5), GROUP_TINT);
      if (this.selection && !onPath(split, head, this.selection)) out.multiplyScalar(OFF_PATH);
    }
    if (rows === 'reserved') out.multiplyScalar(RESERVED);
    if (rows === 'fresh') out.lerp(this.groupColor.setHex(PALETTE.highlight), FRESH);
    if (item.spec.superseded) out.multiplyScalar(SUPERSEDED);
    if (item.spec.idle) out.multiplyScalar(IDLE);
    return out.multiplyScalar(item.spec.floor === this.focusFloor ? 1 : OTHER_FLOOR);
  }

  private addInstanced(picks: Pick[], geometry: THREE.BufferGeometry, material: THREE.Material): void {
    if (picks.length === 0) return;
    const mesh = new THREE.InstancedMesh(geometry, material, picks.length);
    const m = new THREE.Matrix4();
    const color = new THREE.Color();
    picks.forEach((pick, i) => {
      const { box } = pick;
      m.makeScale(box.sx, box.sy, box.sz).setPosition(box.cx, box.cy, box.cz);
      mesh.setMatrixAt(i, m);
      mesh.setColorAt(i, this.colorOf(pick, 'fill', color));
    });
    mesh.computeBoundingBox();
    mesh.computeBoundingSphere();
    this.group.add(mesh);
    this.meshes.push({ mesh, picks });
  }

  private addEdges(picks: Pick[]): void {
    const positions: number[] = [];
    const colors: number[] = [];
    const color = new THREE.Color();
    for (const pick of picks) {
      const { box, head } = pick;
      // Head bands share a tensor: outline them in their fill color so band borders stay quiet.
      this.colorOf(pick, head === null || pick.part !== 'box' ? 'edge' : 'fill', color);
      if (head !== null && pick.part === 'box') color.multiplyScalar(0.75);
      for (const [x, y, z] of this.outlineOf(pick)) {
        positions.push(box.cx + x * box.sx, box.cy + y * box.sy, box.cz + z * box.sz);
        colors.push(color.r, color.g, color.b);
      }
    }
    // Q, K and V projection weights: one outline in the role color around each role's part (the whole matrix, or a
    // third of a fused one), over its band borders.
    for (const item of new Set(picks.map((p) => p.item))) {
      for (const role of new Set(item.roles)) {
        if (!role) continue;
        const boxes = item.boxes.filter((_, i) => item.roles[i] === role);
        color.setHex(ROLE_COLORS[role]).multiplyScalar(item.spec.floor === this.focusFloor ? 1 : OTHER_FLOOR);
        const b = boxBounds(boxes);
        for (const [x, y, z] of BOX_OUTLINE) {
          positions.push(x < 0 ? b.minX : b.maxX, y < 0 ? b.minY : b.maxY, z < 0 ? b.minZ : b.maxZ);
          colors.push(color.r, color.g, color.b);
        }
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    this.edges = new THREE.LineSegments(geometry, this.edgeMaterial);
    this.group.add(this.edges);
  }

  private clear(): void {
    for (const { mesh } of this.meshes) {
      this.group.remove(mesh);
      mesh.dispose();
    }
    this.meshes = [];
    if (this.edges) {
      this.group.remove(this.edges);
      this.edges.geometry.dispose();
      this.edges = undefined;
    }
    this.setHighlight([]);
  }
}

/** A hue per expert, spread by the golden ratio so neighbors contrast however many experts there are. */
export const expertHue = (expert: number) => (expert * 0.618034) % 1;
