import * as THREE from 'three/webgpu';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { boxBounds, type PlacedTensor } from '../layout';
import type { Part, PartKind } from '../model/sections';
import { PALETTE } from '../palette';

/**
 * Muted tints per part: structure, not data, so far quieter than the tensor colors. Attention and the router lean
 * blue and teal, the MLP and experts warm, so the two halves of a layer tell apart at a glance.
 */
const TINTS: Record<PartKind, number> = {
  attention: PALETTE.trayAttention,
  mlp: PALETTE.trayMlp,
  router: PALETTE.trayRouter,
  shared: PALETTE.trayShared,
  experts: PALETTE.trayMlp,
  expert: PALETTE.trayExpert,
};
/** How far trays reach past their blocks, and how far below the floor they sit (experts' trays nest above theirs). */
const MARGIN: Record<PartKind, number> = { attention: 1536, mlp: 1536, router: 1536, shared: 512, experts: 1536, expert: 512 };
const DEPTH: Record<PartKind, number> = { attention: 48, mlp: 48, router: 48, shared: 24, experts: 48, expert: 24 };
const THICKNESS = 16;
const FILL_OPACITY = 0.14;
/** Trays of experts no token picks, like their weights, are drawn fainter. */
const IDLE_OPACITY = 0.05;
const HOVER_OPACITY = 0.3;

interface Tray {
  part: Part;
  mesh: THREE.Mesh;
  outline: THREE.LineSegments;
  fill: THREE.MeshBasicMaterial;
  edge: THREE.LineBasicMaterial;
  opacity: number;
}

/** Thin translucent trays under the parts of the focused floor, with labels; hovering one is for its totals. */
export class SectionScene {
  readonly group = new THREE.Group();
  private readonly unitBox = new THREE.BoxGeometry(1, 1, 1);
  private readonly unitEdges = new THREE.EdgesGeometry(this.unitBox);
  private trays: Tray[] = [];
  private labels: CSS2DObject[] = [];
  private hovered: Tray | null = null;

  update(parts: Part[], placed: PlacedTensor[]): void {
    this.clear();
    const byId = new Map(placed.map((p) => [p.spec.id, p]));
    for (const part of parts) {
      const boxes = part.tensorIds.flatMap((id) => byId.get(id)?.boxes ?? []);
      if (boxes.length === 0) continue;
      const m = MARGIN[part.kind];
      const b = boxBounds(boxes);
      const [minX, maxX, minZ, maxZ] = [b.minX - m, b.maxX + m, b.minZ - m, b.maxZ + m];
      const top = b.minY - DEPTH[part.kind];
      const opacity = part.idle ? IDLE_OPACITY : FILL_OPACITY;
      const fill = new THREE.MeshBasicMaterial({ color: TINTS[part.kind], transparent: true, opacity, depthWrite: false });
      const edge = new THREE.LineBasicMaterial({ color: TINTS[part.kind], transparent: true, opacity: part.idle ? 0.3 : 0.7 });
      const mesh = new THREE.Mesh(this.unitBox, fill);
      const outline = new THREE.LineSegments(this.unitEdges, edge);
      for (const obj of [mesh, outline]) {
        obj.position.set((minX + maxX) / 2, top - THICKNESS / 2, (minZ + maxZ) / 2);
        obj.scale.set(maxX - minX, THICKNESS, maxZ - minZ);
        this.group.add(obj);
      }
      this.trays.push({ part, mesh, outline, fill, edge, opacity });

      const div = document.createElement('div');
      div.className = `section-label${part.kind === 'expert' || part.kind === 'shared' ? ' expert' : ''}${part.idle ? ' idle' : ''}`;
      div.textContent = part.name;
      const label = new CSS2DObject(div);
      // Anchored by its bottom-left corner on the tray's front-left corner.
      label.center.set(0, 1);
      label.position.set(minX, top, maxZ);
      this.group.add(label);
      this.labels.push(label);
    }
  }

  /** The tray under the ray (an expert's or the shared experts' before the one around it), or null. */
  pick(raycaster: THREE.Raycaster): Part | null {
    const hits = raycaster.intersectObjects(this.trays.map((t) => t.mesh), false);
    const trays = hits.map((h) => this.trays.find((t) => t.mesh === h.object)!);
    return (trays.find((t) => t.part.kind === 'expert' || t.part.kind === 'shared') ?? trays[0])?.part ?? null;
  }

  /** Brightens the tray of `part`, or none. */
  setHover(part: Part | null): void {
    const tray = part ? this.trays.find((t) => t.part === part) ?? null : null;
    if (tray === this.hovered) return;
    if (this.hovered) {
      this.hovered.fill.opacity = this.hovered.opacity;
      this.hovered.edge.color.setHex(TINTS[this.hovered.part.kind]);
    }
    this.hovered = tray;
    if (tray) {
      tray.fill.opacity = HOVER_OPACITY;
      tray.edge.color.setHex(PALETTE.highlight);
    }
  }

  private clear(): void {
    for (const t of this.trays) {
      this.group.remove(t.mesh, t.outline);
      t.fill.dispose();
      t.edge.dispose();
    }
    this.trays = [];
    for (const label of this.labels) this.group.remove(label);
    this.labels = [];
    this.hovered = null;
  }
}
