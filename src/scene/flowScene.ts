import * as THREE from 'three/webgpu';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { STATION_GAP, type PlacedTensor } from '../layout';
import type { OpSpec } from '../model/tensors';
import { KIND_COLORS } from './tensorScene';

/** Height of operation nodes above the floor. */
const OP_LIFT = 400;
const DOTS_PER_EDGE = 3;
/** Seconds for a dot to travel one edge. */
const PERIOD = 3;
const DOT_RADIUS = 220;

function bounds(p: PlacedTensor) {
  const b = { minX: Infinity, maxX: -Infinity, minZ: Infinity, maxZ: -Infinity, base: Infinity, top: -Infinity };
  for (const box of p.boxes) {
    b.minX = Math.min(b.minX, box.cx - box.sx / 2);
    b.maxX = Math.max(b.maxX, box.cx + box.sx / 2);
    b.minZ = Math.min(b.minZ, box.cz - box.sz / 2);
    b.maxZ = Math.max(b.maxZ, box.cz + box.sz / 2);
    b.base = Math.min(b.base, box.cy - box.sy / 2);
    b.top = Math.max(b.top, box.cy + box.sy / 2);
  }
  return b;
}

/** Where edges attach to a tensor: the middle of the edge that faces its section's bus. */
function anchor(p: PlacedTensor): THREE.Vector3 {
  const b = bounds(p);
  return new THREE.Vector3((b.minX + b.maxX) / 2, b.top, p.spec.kind === 'weight' ? b.maxZ : b.minZ);
}

/** On the bus right behind the output; beside the first input when the output is on another floor. */
function opPosition(op: OpSpec, out: PlacedTensor, first: PlacedTensor): THREE.Vector3 {
  if (out.spec.floor === op.floor) {
    const b = bounds(out);
    return new THREE.Vector3((b.minX + b.maxX) / 2, b.base + OP_LIFT, out.busZ);
  }
  const b = bounds(first);
  return new THREE.Vector3(b.maxX + STATION_GAP / 2, b.base + OP_LIFT, first.busZ);
}

/** Point on the bus a station to the left of a tensor: where data from another floor enters. */
function entryPoint(p: PlacedTensor): THREE.Vector3 {
  const b = bounds(p);
  return new THREE.Vector3(b.minX - STATION_GAP, b.base + OP_LIFT, p.busZ);
}

/** Draws operations of the focused floor as labels, with arcs from inputs to op to output and dots moving along them. */
export class FlowScene {
  readonly group = new THREE.Group();
  private readonly lineMaterial = new THREE.LineBasicMaterial({ vertexColors: true });
  private readonly dotGeometry = new THREE.SphereGeometry(1, 12, 8);
  private readonly dotMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
  private curves: THREE.QuadraticBezierCurve3[] = [];
  private lines?: THREE.LineSegments;
  private dots?: THREE.InstancedMesh;
  private labels: CSS2DObject[] = [];
  private readonly matrix = new THREE.Matrix4();
  private readonly point = new THREE.Vector3();

  constructor(private readonly floorName: (floor: number) => string) {}

  update(ops: OpSpec[], placed: PlacedTensor[], focusFloor: number | null): void {
    this.clear();
    if (focusFloor === null) return;
    const byId = new Map(placed.map((p) => [p.spec.id, p]));
    const positions: number[] = [];
    const colors: number[] = [];
    const color = new THREE.Color();

    const arc = (a: THREE.Vector3, b: THREE.Vector3, hex: number) => {
      const mid = a.clone().add(b).multiplyScalar(0.5);
      mid.y = Math.max(a.y, b.y) + Math.min(a.distanceTo(b) * 0.25, 12000) + 200;
      const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
      this.curves.push(curve);
      const pts = curve.getPoints(24);
      color.setHex(hex);
      for (let i = 0; i < pts.length - 1; i++) {
        positions.push(pts[i].x, pts[i].y, pts[i].z, pts[i + 1].x, pts[i + 1].y, pts[i + 1].z);
        colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
      }
    };

    const label = (text: string, pos: THREE.Vector3, className: string, title = '') => {
      const div = document.createElement('div');
      div.className = className;
      div.textContent = text;
      div.title = title;
      const obj = new CSS2DObject(div);
      obj.position.copy(pos);
      this.group.add(obj);
      this.labels.push(obj);
    };

    for (const op of ops) {
      const out = byId.get(op.output);
      const inputs = op.inputs.map((id) => byId.get(id));
      if (!out || inputs.some((p) => !p)) continue;
      const outColor = KIND_COLORS[out.spec.kind].edge;
      if (op.floor === focusFloor) {
        const pos = opPosition(op, out, inputs[0]!);
        for (const p of inputs) arc(anchor(p!), pos, KIND_COLORS[p!.spec.kind].edge);
        label(op.label, pos, 'op-label', op.note);
        if (out.spec.floor === focusFloor) {
          arc(pos, anchor(out), outColor);
        } else {
          // Leaves the floor: a short stub instead of an arc to the (possibly far) next floor.
          const exit = pos.clone().add(new THREE.Vector3(STATION_GAP * 1.5, 0, 0));
          arc(pos, exit, outColor);
          label(`→ ${this.floorName(out.spec.floor)}`, exit, 'flow-exit');
        }
      } else if (out.spec.floor === focusFloor) {
        // Arrives from another floor.
        const entry = entryPoint(out);
        arc(entry, anchor(out), outColor);
        label(`from ${this.floorName(op.floor)}`, entry, 'flow-exit', op.note);
      }
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
    this.tick(0);
  }

  /** Moves the dots along their edges; `seconds` is the elapsed time. */
  tick(seconds: number): void {
    if (!this.dots) return;
    let i = 0;
    for (const curve of this.curves) {
      for (let d = 0; d < DOTS_PER_EDGE; d++) {
        const t = (seconds / PERIOD + d / DOTS_PER_EDGE) % 1;
        curve.getPoint(t, this.point);
        this.matrix.makeScale(DOT_RADIUS, DOT_RADIUS, DOT_RADIUS).setPosition(this.point);
        this.dots.setMatrixAt(i++, this.matrix);
      }
    }
    this.dots.instanceMatrix.needsUpdate = true;
  }

  private clear(): void {
    for (const label of this.labels) this.group.remove(label);
    this.labels = [];
    this.curves = [];
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
