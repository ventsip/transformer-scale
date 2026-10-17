import * as THREE from 'three/webgpu';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import type { PlacedTensor } from '../layout';
import { byteCount, onPath, type HeadSelection, type OpSpec } from '../model/tensors';
import { KIND_COLORS } from './tensorScene';
import { hasViz, openOpPopup } from '../ui/opPopup';

/** Height of operation nodes above the floor: they hover over the tensor they produce. */
const OP_HOVER = 2500;
const DOTS_PER_EDGE = 3;
/** Seconds for a dot to travel one edge. */
const PERIOD = 3;
/**
 * Pipe diameters are in world units, so they have perspective like the blocks. A width "step" is
 * WIDTH_UNIT world units: about one screen pixel at the focused-floor view.
 */
const WIDTH_UNIT = 65;
/** Pipe width in steps: 1 at 8 KiB, plus this much per doubling of bytes, up to MAX_STEPS. */
const STEPS_PER_DOUBLING = 0.36;
const MAX_STEPS = 9;
/** Smallest flow-dot radius, in width steps. */
const DOT_MIN_STEPS = 1.5;
/** Color of the residual stream: distinct from every tensor kind. */
export const RESIDUAL_COLOR = 0x67e8f9;
/** Color of the generation loop from the sampled token back to the input. */
export const LOOP_COLOR = 0xf8fafc;
/** How far left of the building the generation loop runs. */
const LOOP_MARGIN = 16384;
/** The loop flashes this color on each decode step, fading over PULSE_MS. */
const PULSE_COLOR = 0xfbbf24;
const PULSE_MS = 400;
/** Tube segments: one per UNITS_PER_SEGMENT of length, within these bounds. */
const MIN_SEGMENTS = 48;
const MAX_SEGMENTS = 1024;
const UNITS_PER_SEGMENT = 1000;

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
  // Below outputs on a higher floor and outputs floating above their floor (the sampled token).
  const below = out.spec.floor !== op.floor || out.spec.above !== undefined;
  const y = below ? b.base - OP_HOVER : b.top + OP_HOVER;
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

/** Pipe diameter in world units for `bytes`: logarithmic, one width step at 8 KiB. */
export function pipeDiameter(bytes: number): number {
  const steps = Math.min(MAX_STEPS, Math.max(1, 1 + (Math.log2(Math.max(bytes, 1)) - 13) * STEPS_PER_DOUBLING));
  return steps * WIDTH_UNIT;
}

interface Edge {
  curve: THREE.Curve<THREE.Vector3>;
  diameter: number;
}

/**
 * Draws operations of the focused floor as labels, with pipes from inputs to op to output whose thickness
 * grows with the log of the bytes they carry, and dots moving along them.
 */
export class FlowScene {
  readonly group = new THREE.Group();
  private readonly dotGeometry = new THREE.SphereGeometry(1, 12, 8);
  private readonly dotMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
  private readonly materials = new Map<number, THREE.Material>();
  private readonly loopMaterial = new THREE.MeshBasicMaterial({ color: LOOP_COLOR });
  private readonly pulseColor = new THREE.Color(PULSE_COLOR);
  private pulseStart = -Infinity;
  private edges: Edge[] = [];
  private tubes: THREE.Mesh[] = [];
  private dots?: THREE.InstancedMesh;
  private labels: CSS2DObject[] = [];
  private readonly matrix = new THREE.Matrix4();
  private readonly point = new THREE.Vector3();

  /** `canvas` receives wheel events that land on labels, so zooming works over them too. */
  constructor(
    private readonly floorName: (floor: number) => string,
    private readonly canvas: HTMLElement,
  ) {}

  /** `loop`: ids of the sampled token and the input it feeds next step, joined by a pipe around the building. */
  update(ops: OpSpec[], placed: PlacedTensor[], focusFloor: number | null, sel: HeadSelection | null, loop?: [string, string]): void {
    this.clear();
    if (focusFloor === null) return;
    const byId = new Map(placed.map((p) => [p.spec.id, p]));

    const draw = (curve: THREE.Curve<THREE.Vector3>, color: number, bytes: number, material = this.material(color)) => {
      const diameter = pipeDiameter(bytes);
      this.edges.push({ curve, diameter });
      // Long pipes (the generation loop, risers) need more segments to stay smooth through their bends.
      const segments = Math.min(MAX_SEGMENTS, Math.max(MIN_SEGMENTS, Math.ceil(curve.getLength() / UNITS_PER_SEGMENT)));
      const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, segments, diameter / 2, 8), material);
      this.tubes.push(tube);
      this.group.add(tube);
    };

    /** Within a floor: an arc lifted in proportion to its length. */
    const arc = (a: THREE.Vector3, b: THREE.Vector3, color: number, bytes: number) => {
      // Straight up or down when there is nowhere to arc to.
      if (Math.hypot(a.x - b.x, a.z - b.z) < 1) return draw(new THREE.LineCurve3(a, b), color, bytes);
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
      return div;
    };

    // Ops on the focused floor, plus the op on another floor that produces this floor's input.
    for (const op of ops) {
      const out = byId.get(op.output);
      const inputs = op.inputs.map((id) => byId.get(id));
      if (!out || inputs.some((p) => !p)) continue;
      if (op.floor !== focusFloor && out.spec.floor !== focusFloor) continue;
      const pos = opPosition(op, out, sel);
      const crossFloor = out.spec.floor !== op.floor;
      const fromBelow = crossFloor || out.spec.above !== undefined;
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
      // The op sits right above its output (or right below it, across floors or under a lifted output): a straight line into it.
      draw(new THREE.LineCurve3(pos, anchor(out, sel, fromBelow ? 'bottom' : 'top')), outColor, flowBytes(out, sel));
      const text = op.floor === focusFloor ? op.label : `${op.label} (${this.floorName(op.floor)})`;
      const div = label(text, pos, 'op-label', op.note);
      if (hasViz(op)) {
        // Opens an interactive explanation; pointerdown is stopped so the click does not reach the scene.
        div.classList.add('has-viz');
        div.title = `${op.note}

Click for an interactive view.`;
        div.addEventListener('pointerdown', (e) => e.stopPropagation());
        div.addEventListener('click', () => openOpPopup(op));
      }
    }

    // The generation loop: from the sampled token out past the floor labels, down the outside of the
    // building, and into the next step's input_ids.
    const from = loop && byId.get(loop[0]);
    const to = loop && byId.get(loop[1]);
    if (from && to) {
      const a = anchor(from, null);
      // Enters input_ids from below: down past the embedding floor, under it, then up into the block.
      const b = anchor(to, null, 'bottom');
      const left = Math.min(...placed.flatMap((p) => p.boxes.map((box) => box.cx - box.sx / 2))) - LOOP_MARGIN;
      const top = a.y + OP_HOVER;
      const entry = b.y - OP_HOVER;
      const path = new THREE.CatmullRomCurve3([
        a, new THREE.Vector3(a.x, top, a.z), new THREE.Vector3(left, top, a.z),
        new THREE.Vector3(left, entry, b.z), new THREE.Vector3(b.x, entry, b.z), b,
      ], false, 'centripetal');
      draw(path, LOOP_COLOR, byteCount(from.spec), this.loopMaterial);
      label('next step: token → input_ids', new THREE.Vector3(left, (top + entry) / 2, b.z), 'op-label',
        'The sampled token id is the whole input of the next decode step: 8 bytes go around the loop, while every weight is read again.');
    }

    if (this.edges.length === 0) return;
    this.dots = new THREE.InstancedMesh(this.dotGeometry, this.dotMaterial, this.edges.length * DOTS_PER_EDGE);
    this.dots.frustumCulled = false;
    this.group.add(this.dots);
  }

  /** Flashes the generation loop: called once per decode step while generating. */
  pulse(): void {
    this.pulseStart = performance.now();
  }

  /** Moves the dots along the edges and fades the loop flash; `seconds` is the elapsed time. */
  tick(seconds: number): void {
    const flash = Math.max(0, 1 - (performance.now() - this.pulseStart) / PULSE_MS);
    this.loopMaterial.color.setHex(LOOP_COLOR).lerp(this.pulseColor, flash);
    if (!this.dots) return;
    let i = 0;
    for (const e of this.edges) {
      // Slightly wider than its pipe, so it shows as a bead moving along it.
      const r = Math.max(DOT_MIN_STEPS * WIDTH_UNIT, e.diameter * 0.6);
      for (let d = 0; d < DOTS_PER_EDGE; d++) {
        const t = (seconds / PERIOD + d / DOTS_PER_EDGE) % 1;
        e.curve.getPoint(t, this.point);
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
    for (const tube of this.tubes) {
      this.group.remove(tube);
      tube.geometry.dispose();
    }
    this.tubes = [];
    this.edges = [];
    if (this.dots) {
      this.group.remove(this.dots);
      this.dots.dispose();
      this.dots = undefined;
    }
  }
}
