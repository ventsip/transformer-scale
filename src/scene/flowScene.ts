import * as THREE from 'three/webgpu';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import type { PlacedTensor } from '../layout';
import { byteCount, onPath, type HeadSelection, type OpSpec } from '../model/tensors';
import { KIND_COLORS } from './tensorScene';

/** Height of operation nodes above the floor: they hover over the tensor they produce. */
const OP_HOVER = 2500;
const DOTS_PER_EDGE = 3;
/** Seconds for a dot to travel one edge. */
const PERIOD = 3;
/** Smallest radius of a flow dot, in screen pixels. */
const DOT_MIN_PX = 1.5;
/** Edge width: 1 px at 8 KiB, plus this much per doubling of bytes, up to MAX_WIDTH_PX. */
export const PX_PER_DOUBLING = 0.36;
const MAX_WIDTH_PX = 9;
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

/** Hovering over the center of its output; when the output is on a higher floor, just below it instead. */
function opPosition(op: OpSpec, out: PlacedTensor, sel: HeadSelection | null): THREE.Vector3 {
  const b = bounds(out, sel);
  const y = out.spec.floor === op.floor ? b.top + OP_HOVER : b.base - OP_HOVER;
  return new THREE.Vector3((b.minX + b.maxX) / 2, y, (b.minZ + b.maxZ) / 2);
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

/** Bytes an edge carries from tensor `p`: the op's override, else the tensor, else with a head selected its share. */
function flowBytes(p: PlacedTensor, sel: HeadSelection | null, override?: number): number {
  if (override !== undefined) return override;
  const total = byteCount(p.spec);
  const split = p.spec.heads;
  if (!sel || !split) return total;
  let on = 0;
  let all = 0;
  p.boxes.forEach((b, i) => {
    const area = b.sx * b.sz;
    all += area;
    if (onPath(split, p.heads[i]!, sel)) on += area;
  });
  return (total * on) / all;
}

/** On-screen width in pixels of an edge carrying `bytes`: logarithmic, 1 px at 8 KiB. */
export function edgeWidthPx(bytes: number): number {
  return Math.min(MAX_WIDTH_PX, Math.max(1, 1 + (Math.log2(Math.max(bytes, 1)) - 13) * PX_PER_DOUBLING));
}

interface Edge {
  curve: THREE.Curve<THREE.Vector3>;
  /** Start, middle and end: their camera distances decide when the tube needs rebuilding. */
  probes: THREE.Vector3[];
  color: number;
  widthPx: number;
  tube?: THREE.Mesh;
  /** Camera distance to each probe when the tube was built. */
  builtAt: number[];
}

const TUBE_SEGMENTS = 48;
const TUBE_SIDES = 6;

/** Like TubeGeometry, but with a radius that can change along the curve. */
function tubeGeometry(curve: THREE.Curve<THREE.Vector3>, radiusAt: (p: THREE.Vector3) => number): THREE.BufferGeometry {
  const frames = curve.computeFrenetFrames(TUBE_SEGMENTS, false);
  const positions: number[] = [];
  const indices: number[] = [];
  const p = new THREE.Vector3();
  const offset = new THREE.Vector3();
  for (let i = 0; i <= TUBE_SEGMENTS; i++) {
    curve.getPointAt(i / TUBE_SEGMENTS, p);
    const r = radiusAt(p);
    for (let j = 0; j <= TUBE_SIDES; j++) {
      const a = (j / TUBE_SIDES) * Math.PI * 2;
      offset.copy(frames.normals[i]).multiplyScalar(-Math.cos(a)).addScaledVector(frames.binormals[i], Math.sin(a));
      positions.push(p.x + r * offset.x, p.y + r * offset.y, p.z + r * offset.z);
    }
  }
  const ring = TUBE_SIDES + 1;
  for (let i = 0; i < TUBE_SEGMENTS; i++) {
    for (let j = 0; j < TUBE_SIDES; j++) {
      const a = i * ring + j;
      const b = (i + 1) * ring + j;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  return geometry;
}

/**
 * Draws operations of the focused floor as labels, with tubes from inputs to op to output whose width
 * grows with the log of the bytes they carry, and dots moving along them.
 */
export class FlowScene {
  readonly group = new THREE.Group();
  private readonly dotGeometry = new THREE.SphereGeometry(1, 12, 8);
  private readonly dotMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
  private readonly materials = new Map<number, THREE.Material>();
  private edges: Edge[] = [];
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

    const draw = (curve: THREE.Curve<THREE.Vector3>, color: number, bytes: number) =>
      this.edges.push({ curve, probes: [0, 0.5, 1].map((t) => curve.getPoint(t)), color, widthPx: edgeWidthPx(bytes), builtAt: [] });

    /** Within a floor: an arc lifted in proportion to its length. */
    const arc = (a: THREE.Vector3, b: THREE.Vector3, color: number, bytes: number) => {
      const mid = a.clone().add(b).multiplyScalar(0.5);
      mid.y = Math.max(a.y, b.y) + Math.min(a.distanceTo(b) * 0.25, 12000) + 200;
      draw(new THREE.QuadraticBezierCurve3(a, mid, b), color, bytes);
    };

    /** Between floors: leaves `a` heading straight up and enters `b` from below, heading up. */
    const riser = (a: THREE.Vector3, b: THREE.Vector3, color: number, bytes: number) => {
      const lift = Math.max((b.y - a.y) / 2, 2000);
      draw(new THREE.CubicBezierCurve3(a, a.clone().setY(a.y + lift), b.clone().setY(b.y - lift), b), color, bytes);
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
    for (const op of ops) {
      const out = byId.get(op.output);
      const inputs = op.inputs.map((id) => byId.get(id));
      if (!out || inputs.some((p) => !p)) continue;
      if (op.floor !== focusFloor && out.spec.floor !== focusFloor) continue;
      const pos = opPosition(op, out, sel);
      const crossFloor = out.spec.floor !== op.floor;
      // The residual stream (into an op that produces it, and out of it) gets its own color.
      // Reads that branch off it (x into RMSNorm) and branch outputs joining it (o, m) keep their kind's color.
      const carriesStream = !!out.spec.residual;
      for (const p of inputs) {
        const color = carriesStream && p!.spec.residual ? RESIDUAL_COLOR : KIND_COLORS[p!.spec.kind].edge;
        const bytes = flowBytes(p!, sel, op.inputBytes?.[p!.spec.id]);
        // Inputs from a lower floor (RoPE tables), and inputs of an op that sits under a higher floor, rise into it.
        if (p!.spec.floor < op.floor || crossFloor) riser(anchor(p!, sel), pos, color, bytes);
        else arc(anchor(p!, sel), pos, color, bytes);
      }
      const outColor = carriesStream ? RESIDUAL_COLOR : KIND_COLORS[out.spec.kind].edge;
      // The op sits right above its output (or right below it, across floors): a straight line into it.
      draw(new THREE.LineCurve3(pos, anchor(out, sel, crossFloor ? 'bottom' : 'top')), outColor, flowBytes(out, sel));
      const text = op.floor === focusFloor ? op.label : `${op.label} (${this.floorName(op.floor)})`;
      label(text, pos, 'op-label', op.note);
    }

    if (this.edges.length === 0) return;
    this.dots = new THREE.InstancedMesh(this.dotGeometry, this.dotMaterial, this.edges.length * DOTS_PER_EDGE);
    this.dots.frustumCulled = false;
    this.group.add(this.dots);
  }

  /** Keeps tubes at their pixel widths for the current camera distance and moves the dots along the edges. */
  tick(seconds: number, camera: THREE.PerspectiveCamera): void {
    // World size of one screen pixel per unit of distance from the camera.
    const pxPerDist = (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / window.innerHeight;
    for (const e of this.edges) {
      // Rebuild when the camera distance to the start, middle or end has changed noticeably.
      const dists = e.probes.map((p) => p.distanceTo(camera.position));
      const stale = !e.tube || dists.some((d, i) => d > e.builtAt[i] * TUBE_REBUILD || d < e.builtAt[i] / TUBE_REBUILD);
      if (!stale) continue;
      if (e.tube) {
        this.group.remove(e.tube);
        e.tube.geometry.dispose();
      }
      // Radius varies along the tube so every point shows the same pixel width, near or far.
      const geometry = tubeGeometry(e.curve, (p) => (p.distanceTo(camera.position) * pxPerDist * e.widthPx) / 2);
      e.tube = new THREE.Mesh(geometry, this.material(e.color));
      e.builtAt = dists;
      this.group.add(e.tube);
    }
    if (!this.dots) return;
    let i = 0;
    for (const e of this.edges) {
      for (let d = 0; d < DOTS_PER_EDGE; d++) {
        const t = (seconds / PERIOD + d / DOTS_PER_EDGE) % 1;
        e.curve.getPoint(t, this.point);
        const r = this.point.distanceTo(camera.position) * pxPerDist * Math.max(DOT_MIN_PX, e.widthPx * 0.6);
        this.matrix.makeScale(r, r, r).setPosition(this.point);
        this.dots.setMatrixAt(i++, this.matrix);
      }
    }
    this.dots.instanceMatrix.needsUpdate = true;
  }

  private material(color: number): THREE.Material {
    let m = this.materials.get(color);
    if (!m) this.materials.set(color, (m = new THREE.MeshBasicMaterial({ color })));
    return m;
  }

  private clear(): void {
    for (const label of this.labels) this.group.remove(label);
    this.labels = [];
    for (const e of this.edges) {
      if (!e.tube) continue;
      this.group.remove(e.tube);
      e.tube.geometry.dispose();
    }
    this.edges = [];
    if (this.dots) {
      this.group.remove(this.dots);
      this.dots.dispose();
      this.dots = undefined;
    }
  }
}
