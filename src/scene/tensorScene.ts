import * as THREE from 'three/webgpu';
import type { Box, PlacedTensor } from '../layout';
import type { TensorKind } from '../model/tensors';

export const KIND_COLORS: Record<TensorKind, { fill: number; edge: number }> = {
  weight: { fill: 0x4f8ff7, edge: 0x9cc2ff },
  activation: { fill: 0xf5a524, edge: 0xffd27a },
  kv: { fill: 0x22c55e, edge: 0x86efac },
  scores: { fill: 0xef4444, edge: 0xf87171 },
};

// Corner index bits: 1 = +x, 2 = +y, 4 = +z.
const BOX_EDGES = [0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 1, 3, 4, 6, 5, 7, 0, 4, 1, 5, 2, 6, 3, 7];

export interface Pick {
  item: PlacedTensor;
  box: Box;
}

/** Renders placed tensors: instanced solid boxes, ghost boxes for scores, and 1px edges for all. */
export class TensorScene {
  readonly group = new THREE.Group();
  private readonly unitBox = new THREE.BoxGeometry(1, 1, 1);
  private readonly solidMaterial = new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0 });
  private readonly ghostMaterial = new THREE.MeshBasicMaterial({
    color: KIND_COLORS.scores.fill, transparent: true, opacity: 0.08, depthWrite: false,
  });
  private readonly edgeMaterial = new THREE.LineBasicMaterial({ vertexColors: true });
  private meshes: { mesh: THREE.InstancedMesh; picks: Pick[] }[] = [];
  private edges?: THREE.LineSegments;
  private readonly highlight: THREE.LineSegments;

  constructor() {
    this.highlight = new THREE.LineSegments(
      new THREE.EdgesGeometry(this.unitBox),
      new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true }),
    );
    this.highlight.renderOrder = 10;
    this.highlight.visible = false;
    this.group.add(this.highlight);
  }

  update(items: PlacedTensor[]): void {
    this.clear();
    const solid: Pick[] = [];
    const ghost: Pick[] = [];
    for (const item of items) {
      for (const box of item.boxes) (item.spec.kind === 'scores' ? ghost : solid).push({ item, box });
    }
    this.addInstanced(solid, this.solidMaterial);
    this.addInstanced(ghost, this.ghostMaterial);
    this.addEdges([...solid, ...ghost]);
  }

  pick(raycaster: THREE.Raycaster): Pick | null {
    const hits = raycaster.intersectObjects(this.meshes.map((m) => m.mesh), false);
    for (const hit of hits) {
      const entry = this.meshes.find((m) => m.mesh === hit.object);
      if (entry && hit.instanceId !== undefined) return entry.picks[hit.instanceId];
    }
    return null;
  }

  setHighlight(box: Box | null): void {
    this.highlight.visible = box !== null;
    if (!box) return;
    this.highlight.position.set(box.cx, box.cy, box.cz);
    this.highlight.scale.set(box.sx, Math.max(box.sy, 1), box.sz);
  }

  private addInstanced(picks: Pick[], material: THREE.Material): void {
    if (picks.length === 0) return;
    const mesh = new THREE.InstancedMesh(this.unitBox, material, picks.length);
    const m = new THREE.Matrix4();
    const color = new THREE.Color();
    picks.forEach(({ item, box }, i) => {
      m.makeScale(box.sx, box.sy, box.sz).setPosition(box.cx, box.cy, box.cz);
      mesh.setMatrixAt(i, m);
      mesh.setColorAt(i, color.setHex(KIND_COLORS[item.spec.kind].fill));
    });
    mesh.computeBoundingBox();
    mesh.computeBoundingSphere();
    this.group.add(mesh);
    this.meshes.push({ mesh, picks });
  }

  private addEdges(picks: Pick[]): void {
    const positions = new Float32Array(picks.length * BOX_EDGES.length * 3);
    const colors = new Float32Array(positions.length);
    const color = new THREE.Color();
    let o = 0;
    for (const { item, box } of picks) {
      color.setHex(KIND_COLORS[item.spec.kind].edge);
      for (const corner of BOX_EDGES) {
        positions[o] = box.cx + (corner & 1 ? 0.5 : -0.5) * box.sx;
        positions[o + 1] = box.cy + (corner & 2 ? 0.5 : -0.5) * box.sy;
        positions[o + 2] = box.cz + (corner & 4 ? 0.5 : -0.5) * box.sz;
        colors[o] = color.r;
        colors[o + 1] = color.g;
        colors[o + 2] = color.b;
        o += 3;
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
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
    this.highlight.visible = false;
  }
}
