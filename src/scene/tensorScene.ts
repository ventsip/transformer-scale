import * as THREE from 'three/webgpu';
import type { Box, PlacedTensor, RowState } from '../layout';
import { kvGroup, onPath, type HeadSelection, type Role, type TensorKind } from '../model/tensors';

export const KIND_COLORS: Record<TensorKind, { fill: number; edge: number }> = {
  weight: { fill: 0x4f8ff7, edge: 0x9cc2ff },
  activation: { fill: 0xf5a524, edge: 0xffd27a },
  kv: { fill: 0x22c55e, edge: 0x86efac },
  scores: { fill: 0xef4444, edge: 0xf87171 },
  buffer: { fill: 0xa78bfa, edge: 0xc4b5fd },
};

/** Color of the causally masked half of attention-score tiles. */
export const MASKED_COLORS = { fill: 0x64748b, edge: 0x475569 };

/** Colors of the Q, K and V projection weights (tint and outline). Hues used for nothing else. */
export const ROLE_COLORS = { q: 0xe879f9, k: 0xfb923c, v: 0x2dd4bf } as const;

/**
 * Which part of a box an instance covers. Causal tiles split along the diagonal: keys run along X and
 * queries along Z, so the causal mask keeps the half where key ≤ query and masks the rest.
 */
export type Part = 'box' | 'kept' | 'masked';

export interface Pick {
  item: PlacedTensor;
  box: Box;
  head: number | null;
  /** For cache tensors: rows written before, by this pass, or not yet. */
  rows: RowState;
  /** For Q, K and V projection weights (or a third of a fused one): which. */
  role: Role | null;
  part: Part;
}

/** Mix of a group hue into a kind color: enough to match heads across tensors, not to hide the kind. */
const GROUP_TINT = 0.3;
/** Mix of the role color into the Q, K and V projection weights (before the group tint). */
const ROLE_TINT = 0.5;
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

function prismOutline(tri: [number, number][]): Segment[] {
  const out: Segment[] = [];
  for (let i = 0; i < 3; i++) {
    const [ax, az] = tri[i];
    const [bx, bz] = tri[(i + 1) % 3];
    out.push([ax, -0.5, az], [bx, -0.5, bz], [ax, 0.5, az], [bx, 0.5, bz], [ax, -0.5, az], [ax, 0.5, az]);
  }
  return out;
}

function prismGeometry(tri: [number, number][]): THREE.BufferGeometry {
  const [a, b, c] = tri.map(([x, z]) => [new THREE.Vector3(x, -0.5, z), new THREE.Vector3(x, 0.5, z)]);
  const quads = [[a, b], [b, c], [c, a]].flatMap(([p, q]) => [p[0], q[0], q[1], p[0], q[1], p[1]]);
  const vertices = [a[1], b[1], c[1], a[0], c[0], b[0], ...quads];
  const geometry = new THREE.BufferGeometry().setFromPoints(vertices);
  geometry.computeVertexNormals();
  return geometry;
}

const OUTLINES: Record<Part, Segment[]> = {
  box: BOX_OUTLINE,
  kept: prismOutline(TRIANGLES.kept),
  masked: prismOutline(TRIANGLES.masked),
};

/** Renders placed tensors: instanced boxes, prism halves for causally masked tiles, and 1px outlines for all. */
export class TensorScene {
  readonly group = new THREE.Group();
  private readonly unitBox = new THREE.BoxGeometry(1, 1, 1);
  private readonly prisms = { kept: prismGeometry(TRIANGLES.kept), masked: prismGeometry(TRIANGLES.masked) };
  private readonly solidMaterial = new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0 });
  // Prism faces are wound inconsistently; drawing both sides keeps them solid from every angle.
  private readonly prismMaterial = new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0, side: THREE.DoubleSide });
  private readonly edgeMaterial = new THREE.LineBasicMaterial({ vertexColors: true });
  private meshes: { mesh: THREE.InstancedMesh; picks: Pick[] }[] = [];
  private edges?: THREE.LineSegments;
  /** Outlines of hovered boxes: one for a block, the tensor at the end of a pipe. Grown on demand. */
  private readonly highlights: THREE.LineSegments[] = [];
  private readonly highlightGeometry: THREE.EdgesGeometry;
  private readonly highlightMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true });
  private focusFloor = -1;
  private selection: HeadSelection | null = null;
  private readonly groupColor = new THREE.Color();

  constructor() {
    this.highlightGeometry = new THREE.EdgesGeometry(this.unitBox);
  }

  /** Floors other than `focusFloor` are drawn dimmed, and so are head bands off the selected head's path. */
  update(items: PlacedTensor[], focusFloor: number, selection: HeadSelection | null): void {
    this.clear();
    this.focusFloor = focusFloor;
    this.selection = selection;
    const solid: Pick[] = [];
    const kept: Pick[] = [];
    const masked: Pick[] = [];
    for (const item of items) {
      item.boxes.forEach((box, i) => {
        const head = item.heads[i];
        const rows = item.rows[i];
        const role = item.roles[i];
        if (!item.spec.causal) return solid.push({ item, box, head, rows, role, part: 'box' });
        kept.push({ item, box, head, rows, role, part: 'kept' });
        masked.push({ item, box, head, rows, role, part: 'masked' });
      });
    }
    this.addInstanced(solid, this.unitBox, this.solidMaterial);
    this.addInstanced(kept, this.prisms.kept, this.prismMaterial);
    this.addInstanced(masked, this.prisms.masked, this.prismMaterial);
    this.addEdges([...solid, ...kept, ...masked]);
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

  private colorOf({ item, head, part, rows, role }: Pick, which: 'fill' | 'edge', out: THREE.Color): THREE.Color {
    out.setHex(part === 'masked' ? MASKED_COLORS[which] : KIND_COLORS[item.spec.kind][which]);
    // Q, K and V projection weights: blended toward their role color, under the KV-group tint.
    if (role) out.lerp(this.groupColor.setHex(ROLE_COLORS[role]), ROLE_TINT);
    const split = item.spec.heads;
    if (split && head !== null) {
      const groups = split.count / split.perGroup;
      // Step hues by 3 slots around the wheel so neighboring groups contrast instead of forming a gradient.
      const hue = ((kvGroup(split, head) * 3) % groups) / groups;
      if (part !== 'masked') out.lerp(this.groupColor.setHSL(hue, 0.7, 0.5), GROUP_TINT);
      if (this.selection && !onPath(split, head, this.selection)) out.multiplyScalar(OFF_PATH);
    }
    if (rows === 'reserved') out.multiplyScalar(RESERVED);
    if (rows === 'fresh') out.lerp(this.groupColor.setHex(0xffffff), FRESH);
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
      for (const [x, y, z] of OUTLINES[pick.part]) {
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
        const b = {
          minX: Math.min(...boxes.map((x) => x.cx - x.sx / 2)), maxX: Math.max(...boxes.map((x) => x.cx + x.sx / 2)),
          minY: Math.min(...boxes.map((x) => x.cy - x.sy / 2)), maxY: Math.max(...boxes.map((x) => x.cy + x.sy / 2)),
          minZ: Math.min(...boxes.map((x) => x.cz - x.sz / 2)), maxZ: Math.max(...boxes.map((x) => x.cz + x.sz / 2)),
        };
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
