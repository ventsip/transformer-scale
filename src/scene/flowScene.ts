import * as THREE from 'three/webgpu';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { STATION_GAP, type PlacedTensor } from '../layout';
import { onPath, type HeadSelection, type OpSpec } from '../model/tensors';
import { KIND_COLORS } from './tensorScene';

/** Height of operation nodes above the floor. */
const OP_LIFT = 400;
const DOTS_PER_EDGE = 3;
/** Seconds for a dot to travel one edge. */
const PERIOD = 3;
/** Dot radius as a fraction of its distance to the camera: a constant size on screen. */
const DOT_SCALE = 0.0015;
/** Radius of residual-stream tubes as a fraction of their distance to the camera: a steady width on screen. */
const TUBE_SCALE = 0.0025;
/** Color of the residual stream: distinct from every tensor kind. */
export const RESIDUAL_COLOR = 0x67e8f9;
/** Rebuild tubes when the camera distance has changed by more than this factor since they were built. */
const TUBE_REBUILD = 1.15;

/** Bounds of a tensor's boxes; with a head selected, of only the bands on its path (when it has any). */
function bounds(p: PlacedTensor, sel: HeadSelection | null = null) {
  const split = p.spec.heads;
  const boxes = sel && split ? p.boxes.filter((_, i) => onPath(split, p.heads[i]!, sel)) : p.boxes;
  const b = { minX: Infinity, maxX: -Infinity, minZ: Infinity, maxZ: -Infinity, base: Infinity, top: -Infinity };
  for (const box of boxes.length ? boxes : p.boxes) {
    b.minX = Math.min(b.minX, box.cx - box.sx / 2);
    b.maxX = Math.max(b.maxX, box.cx + box.sx / 2);
    b.minZ = Math.min(b.minZ, box.cz - box.sz / 2);
    b.maxZ = Math.max(b.maxZ, box.cz + box.sz / 2);
    b.base = Math.min(b.base, box.cy - box.sy / 2);
    b.top = Math.max(b.top, box.cy + box.sy / 2);
  }
  return b;
}

/** Where edges attach to a tensor: the center of its top (or bottom) face, all tiles together. */
function anchor(p: PlacedTensor, sel: HeadSelection | null, face: 'top' | 'bottom' = 'top'): THREE.Vector3 {
  const b = bounds(p, sel);
  return new THREE.Vector3((b.minX + b.maxX) / 2, face === 'top' ? b.top : b.base, (b.minZ + b.maxZ) / 2);
}

/** On the bus right behind the output; beside the first input when the output is on another floor. */
function opPosition(op: OpSpec, out: PlacedTensor, first: PlacedTensor, sel: HeadSelection | null): THREE.Vector3 {
  if (out.spec.floor === op.floor) {
    const b = bounds(out, sel);
    return new THREE.Vector3(out.opX ?? (b.minX + b.maxX) / 2, b.base + OP_LIFT, out.opZ);
  }
  const b = bounds(first);
  return new THREE.Vector3(b.maxX + STATION_GAP / 2, b.base + OP_LIFT, first.busZ);
}

/**
 * Tensors on other floors that the focused floor's flow links to: the result its last op sends up,
 * the sources of the op that produces its input, and inputs its ops take from other floors (RoPE tables).
 * They are drawn even when otherwise hidden.
 */
export function linkedTensors(ops: OpSpec[], floorOf: (id: string) => number, focusFloor: number): Set<string> {
  const ids = new Set<string>();
  for (const op of ops) {
    const outFloor = floorOf(op.output);
    if (op.floor === focusFloor && outFloor !== focusFloor) ids.add(op.output);
    if (op.floor === focusFloor) op.inputs.filter((id) => floorOf(id) !== focusFloor).forEach((id) => ids.add(id));
    if (outFloor === focusFloor && op.floor !== focusFloor) op.inputs.forEach((id) => ids.add(id));
  }
  return ids;
}

/** Draws operations of the focused floor as labels, with arcs from inputs to op to output and dots moving along them. */
export class FlowScene {
  readonly group = new THREE.Group();
  private readonly lineMaterial = new THREE.LineBasicMaterial({ vertexColors: true });
  private readonly dotGeometry = new THREE.SphereGeometry(1, 12, 8);
  private readonly dotMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
  private curves: THREE.Curve<THREE.Vector3>[] = [];
  private lines?: THREE.LineSegments;
  /** Edges that carry the residual stream, drawn as tubes sized for the camera distance they were built at. */
  private bold: { curve: THREE.Curve<THREE.Vector3>; mid: THREE.Vector3; material: THREE.Material; tube?: THREE.Mesh; builtAt: number }[] = [];
  private dots?: THREE.InstancedMesh;
  private labels: CSS2DObject[] = [];
  private readonly matrix = new THREE.Matrix4();
  private readonly point = new THREE.Vector3();

  /** `canvas` receives wheel events that land on labels, so zooming works over them too. */
  constructor(
    private readonly floorName: (floor: number) => string,
    private readonly canvas: HTMLElement,
  ) {}

  update(ops: OpSpec[], placed: PlacedTensor[], focusFloor: number | null, sel: HeadSelection | null): void {
    this.clear();
    if (focusFloor === null) return;
    const byId = new Map(placed.map((p) => [p.spec.id, p]));
    const positions: number[] = [];
    const colors: number[] = [];
    const color = new THREE.Color();

    const draw = (curve: THREE.Curve<THREE.Vector3>, hex: number, bold: boolean) => {
      this.curves.push(curve);
      if (bold) {
        this.bold.push({ curve, mid: curve.getPoint(0.5), material: new THREE.MeshBasicMaterial({ color: RESIDUAL_COLOR }), builtAt: 0 });
        return;
      }
      const pts = curve.getPoints(32);
      color.setHex(hex);
      for (let i = 0; i < pts.length - 1; i++) {
        positions.push(pts[i].x, pts[i].y, pts[i].z, pts[i + 1].x, pts[i + 1].y, pts[i + 1].z);
        colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
      }
    };

    /** Within a floor: an arc lifted in proportion to its length. */
    const arc = (a: THREE.Vector3, b: THREE.Vector3, hex: number, bold = false) => {
      const mid = a.clone().add(b).multiplyScalar(0.5);
      mid.y = Math.max(a.y, b.y) + Math.min(a.distanceTo(b) * 0.25, 12000) + 200;
      draw(new THREE.QuadraticBezierCurve3(a, mid, b), hex, bold);
    };

    /** Between floors: leaves `a` heading straight up and enters `b` from below, heading up. */
    const riser = (a: THREE.Vector3, b: THREE.Vector3, hex: number, bold = false) => {
      const lift = Math.max((b.y - a.y) / 2, 2000);
      draw(new THREE.CubicBezierCurve3(a, a.clone().setY(a.y + lift), b.clone().setY(b.y - lift), b), hex, bold);
    };

    const label = (text: string, pos: THREE.Vector3, className: string, title = '') => {
      const div = document.createElement('div');
      div.className = className;
      div.textContent = text;
      div.title = title;
      div.addEventListener('wheel', (e) => {
        e.preventDefault();
        this.canvas.dispatchEvent(new WheelEvent('wheel', e));
      }, { passive: false });
      const obj = new CSS2DObject(div);
      obj.position.copy(pos);
      this.group.add(obj);
      this.labels.push(obj);
    };

    // Ops on the focused floor, plus the op on another floor that produces this floor's input.
    // Ops whose output is on another floor link to that block directly, with a low arc.
    for (const op of ops) {
      const out = byId.get(op.output);
      const inputs = op.inputs.map((id) => byId.get(id));
      if (!out || inputs.some((p) => !p)) continue;
      if (op.floor !== focusFloor && out.spec.floor !== focusFloor) continue;
      const pos = opPosition(op, out, inputs[0]!, sel);
      // Bold: edges that carry the residual stream (into an op that produces it, and out of it).
      // Reads that branch off it (x into RMSNorm) and branch outputs joining it (o, m) stay thin.
      const carriesStream = !!out.spec.residual;
      for (const p of inputs) {
        const bold = carriesStream && !!p!.spec.residual;
        // Inputs from a lower floor (RoPE tables) rise into the op; same-floor inputs arc.
        if (p!.spec.floor < op.floor) riser(anchor(p!, sel), pos, KIND_COLORS[p!.spec.kind].edge, bold);
        else arc(anchor(p!, sel), pos, KIND_COLORS[p!.spec.kind].edge, bold);
      }
      const outColor = KIND_COLORS[out.spec.kind].edge;
      if (out.spec.floor === op.floor) arc(pos, anchor(out, sel), outColor, carriesStream);
      else riser(pos, anchor(out, sel, 'bottom'), outColor, carriesStream);
      const text = op.floor === focusFloor ? op.label : `${op.label} (${this.floorName(op.floor)})`;
      label(text, pos, 'op-label', op.note);
    }

    if (this.curves.length === 0) return;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    this.lines = new THREE.LineSegments(geometry, this.lineMaterial);
    this.group.add(this.lines);

    this.dots = new THREE.InstancedMesh(this.dotGeometry, this.dotMaterial, this.curves.length * DOTS_PER_EDGE);
    this.dots.frustumCulled = false;
    this.group.add(this.dots);
  }

  /** Moves the dots along their edges; `seconds` is the elapsed time. */
  tick(seconds: number, camera: THREE.Camera): void {
    for (const b of this.bold) {
      const dist = b.mid.distanceTo(camera.position);
      if (b.tube && dist < b.builtAt * TUBE_REBUILD && dist > b.builtAt / TUBE_REBUILD) continue;
      if (b.tube) {
        this.group.remove(b.tube);
        b.tube.geometry.dispose();
      }
      b.tube = new THREE.Mesh(new THREE.TubeGeometry(b.curve, 48, dist * TUBE_SCALE, 6), b.material);
      b.builtAt = dist;
      this.group.add(b.tube);
    }
    if (!this.dots) return;
    let i = 0;
    for (const curve of this.curves) {
      for (let d = 0; d < DOTS_PER_EDGE; d++) {
        const t = (seconds / PERIOD + d / DOTS_PER_EDGE) % 1;
        curve.getPoint(t, this.point);
        const r = this.point.distanceTo(camera.position) * DOT_SCALE;
        this.matrix.makeScale(r, r, r).setPosition(this.point);
        this.dots.setMatrixAt(i++, this.matrix);
      }
    }
    this.dots.instanceMatrix.needsUpdate = true;
  }

  private clear(): void {
    for (const label of this.labels) this.group.remove(label);
    this.labels = [];
    this.curves = [];
    for (const b of this.bold) {
      if (b.tube) {
        this.group.remove(b.tube);
        b.tube.geometry.dispose();
      }
      b.material.dispose();
    }
    this.bold = [];
    if (this.lines) {
      this.group.remove(this.lines);
      this.lines.geometry.dispose();
      this.lines = undefined;
    }
    if (this.dots) {
      this.group.remove(this.dots);
      this.dots.dispose();
      this.dots = undefined;
    }
  }
}
