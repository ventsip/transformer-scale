import * as THREE from 'three/webgpu';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { boxBounds, type PlacedTensor } from '../layout';
import { byteCount, onPath, type HeadSelection, type OpSpec, type Role } from '../model/tensors';
import { KIND_COLORS } from './tensorScene';
import { PALETTE } from '../palette';
import { hasViz, openOpPopup } from '../ui/opPopup';

/** Height of operation nodes above the floor: they hover over the tensor they produce. */
const OP_HOVER = 2500;
/** Pipes climbing from a lower floor: how far in front of the focused floor they rise. */
const CLIMB_MARGIN = 3000;
/** How far apart, left and right of their op, several climbing inputs of one op rise. */
const CLIMB_SPREAD = 1200;
/** Arcs within a floor rise ARC_LIFT of their length, at most ARC_MAX_LIFT, plus ARC_CLEARANCE over their higher end. */
const ARC_LIFT = 0.25;
const ARC_MAX_LIFT = 12000;
const ARC_CLEARANCE = 200;
/** How far pipes between floors run straight up out of their source (at least) before they bend. */
const RISE = 2000;
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
export const RESIDUAL_COLOR = PALETTE.residual;
/** Color of the generation loop from the sampled token back to the input. */
export const LOOP_COLOR = PALETTE.loop;
/** How far left of the building the generation loop runs. */
const LOOP_MARGIN = 16384;
/** The loop flashes this color on each decode step, fading over PULSE_MS. */
const PULSE_COLOR = PALETTE.pulse;
const PULSE_MS = 400;
/** Tube segments: one per UNITS_PER_SEGMENT of length, within these bounds. */
const MIN_SEGMENTS = 48;
const MAX_SEGMENTS = 1024;
const UNITS_PER_SEGMENT = 1000;

/** Bounds of a tensor's boxes; with a head selected, of only the bands on its path (when it has any). */
function bounds(p: PlacedTensor, sel: HeadSelection | null = null, role?: Role) {
  const split = p.spec.heads;
  const onHead = sel && split ? p.boxes.filter((_, i) => onPath(split, p.heads[i]!, sel)) : p.boxes;
  // Only the part of a fused Q/K/V tensor an op reads.
  const boxes = role ? onHead.filter((b) => p.roles[p.boxes.indexOf(b)] === role) : onHead;
  return boxBounds(boxes.length ? boxes : p.boxes);
}

/** Where edges attach to a tensor: the center of its top (or bottom) face, all tiles together. */
function anchor(p: PlacedTensor, sel: HeadSelection | null, face: 'top' | 'bottom' = 'top', role?: Role): THREE.Vector3 {
  const b = bounds(p, sel, role);
  return new THREE.Vector3((b.minX + b.maxX) / 2, face === 'top' ? b.maxY : b.minY, (b.minZ + b.maxZ) / 2);
}

/** Hovering over the center of its output; when the output is on a higher floor, just below it instead. */
function opPosition(op: OpSpec, out: PlacedTensor, sel: HeadSelection | null): THREE.Vector3 {
  const b = bounds(out, sel);
  const y = out.spec.floor === op.floor ? b.maxY + OP_HOVER : b.minY - OP_HOVER;
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

/** What a pipe connects: the tensor(s) at its ends, the op label at the other end, and how to describe it. */
interface Ends {
  tensors: string[];
  label: HTMLElement | null;
  text: string;
}

export interface Edge extends Ends {
  curve: THREE.Curve<THREE.Vector3>;
  /** The pipe's color, which its flow dots share. */
  color: number;
  diameter: number;
  bytes: number;
  tube: THREE.Mesh;
  material: THREE.Material;
  /** Samples along the curve, for picking in screen space. */
  points: THREE.Vector3[];
}

/** A pipe under the pointer, and how far from the camera. */
export interface EdgeHit {
  edge: Edge;
  distance: number;
}

/** Pipes within this many pixels of the pointer count as hovered: thin pipes are about a pixel wide. */
const PICK_PIXELS = 6;
const HOVER_COLOR = PALETTE.highlight;
/**
 * Pipes are see-through, so blocks and other pipes behind them stay visible. They do not write depth, which keeps
 * overlapping pipes from hiding one another by draw order. The hovered pipe is drawn solid.
 */
const PIPE_OPACITY = 0.45;
const PIPE_MATERIAL = { transparent: true, opacity: PIPE_OPACITY, depthWrite: false };

/**
 * Draws operations of the focused floor as labels, with pipes from inputs to op to output whose thickness
 * grows with the log of the bytes they carry, and dots moving along them.
 */
export class FlowScene {
  readonly group = new THREE.Group();
  private readonly dotGeometry = new THREE.SphereGeometry(1, 12, 8);
  private readonly dotMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
  private readonly materials = new Map<number, THREE.Material>();
  private readonly loopMaterial = new THREE.MeshBasicMaterial({ color: LOOP_COLOR, ...PIPE_MATERIAL });
  private readonly pulseColor = new THREE.Color(PULSE_COLOR);
  private pulseStart = -Infinity;
  private edges: Edge[] = [];
  private tubes: THREE.Mesh[] = [];
  private dots?: THREE.InstancedMesh;
  private labels: CSS2DObject[] = [];
  private readonly hoverMaterial = new THREE.MeshBasicMaterial({ color: HOVER_COLOR });
  private hovered: Edge[] = [];
  /** Called when the pointer enters an op label (with the op's pipes) and leaves it (with null). */
  onOpHover?: (pipes: Edge[] | null) => void;
  private readonly projected = new THREE.Vector3();
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

    const draw = (curve: THREE.Curve<THREE.Vector3>, color: number, bytes: number, ends: Ends, material = this.material(color)) => {
      const diameter = pipeDiameter(bytes);
      // Long pipes (the generation loop, risers) need more segments to stay smooth through their bends.
      const segments = Math.min(MAX_SEGMENTS, Math.max(MIN_SEGMENTS, Math.ceil(curve.getLength() / UNITS_PER_SEGMENT)));
      const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, segments, diameter / 2, 8), material);
      this.edges.push({ ...ends, curve, color, diameter, bytes, tube, material, points: curve.getPoints(Math.min(segments, 256)) });
      this.tubes.push(tube);
      this.group.add(tube);
    };

    /** Within a floor: an arc lifted in proportion to its length. */
    const arc = (a: THREE.Vector3, b: THREE.Vector3, color: number, bytes: number, ends: Ends) => {
      // Straight up or down when there is nowhere to arc to.
      if (Math.hypot(a.x - b.x, a.z - b.z) < 1) return draw(new THREE.LineCurve3(a, b), color, bytes, ends);
      const mid = a.clone().add(b).multiplyScalar(0.5);
      mid.y = Math.max(a.y, b.y) + Math.min(a.distanceTo(b) * ARC_LIFT, ARC_MAX_LIFT) + ARC_CLEARANCE;
      draw(new THREE.QuadraticBezierCurve3(a, mid, b), color, bytes, ends);
    };

    /** Between floors: leaves `a` heading straight up and enters `b` from below, heading up. */
    const riser = (a: THREE.Vector3, b: THREE.Vector3, color: number, bytes: number, ends: Ends) => {
      const lift = Math.max((b.y - a.y) / 2, RISE);
      draw(new THREE.CubicBezierCurve3(a, a.clone().setY(a.y + lift), b.clone().setY(b.y - lift), b), color, bytes, ends);
    };

    // Front edge of the focused floor: climbs from lower floors go up in front of it.
    const front = boxBounds(placed.filter((p) => p.spec.floor === focusFloor).flatMap((p) => p.boxes)).maxZ;

    /**
     * From a lower floor into an op above its output (RoPE from the cos / sin tables): leaves `a` heading up, climbs in
     * front of the focused floor to the op's height and runs straight back into `b` from the front, over the blocks.
     * Rising straight into `b` from below would pass through the block the op produces. `side` spreads several such
     * inputs of one op apart (−1 … 1), so they arrive next to each other.
     */
    const climb = (a: THREE.Vector3, b: THREE.Vector3, color: number, bytes: number, ends: Ends, side: number) => {
      const z = front + CLIMB_MARGIN;
      const x = b.x + side * CLIMB_SPREAD;
      const path = new THREE.CatmullRomCurve3([
        a, a.clone().setY(a.y + RISE),
        new THREE.Vector3(x, b.y - OP_HOVER - CLIMB_MARGIN, z), new THREE.Vector3(x, b.y, z),
        new THREE.Vector3(b.x + side * CLIMB_SPREAD * 0.3, b.y, (z + b.z) / 2), b,
      ], false, 'centripetal');
      draw(path, color, bytes, ends);
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
      div.addEventListener('pointerenter', () => this.onOpHover?.(this.pipesOf(div)));
      div.addEventListener('pointerleave', () => this.onOpHover?.(null));
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
      // The residual stream (into an op that produces it, and out of it) gets its own color.
      // Reads that branch off it (x into RMSNorm) and branch outputs joining it (o, m) keep their kind's color.
      const carriesStream = !!out.spec.residual;
      const text = op.floor === focusFloor ? op.label : `${op.label} (${this.floorName(op.floor)})`;
      const div = label(text, pos, 'op-label', op.note);
      // Inputs from lower floors climb side by side: −1 … 1 across them.
      const lower = inputs.filter((p) => p!.spec.floor < op.floor);
      const side = (p: PlacedTensor) => (lower.length > 1 ? (2 * lower.indexOf(p)) / (lower.length - 1) - 1 : 0);
      for (const p of inputs) {
        const color = carriesStream && p!.spec.residual ? RESIDUAL_COLOR : KIND_COLORS[p!.spec.kind].edge;
        const bytes = flowBytes(p!, sel, op.inputBytes?.[p!.spec.id]);
        const ends = { tensors: [p!.spec.id], label: div, text: `${p!.spec.label} → ${text}` };
        // Inputs of an op that sits under a higher floor rise into it; inputs from a lower floor (RoPE tables, the
        // causal mask) climb in front of the floor and come in from the front; the rest arc within the floor.
        if (crossFloor) riser(anchor(p!, sel, 'top', op.inputRoles?.[p!.spec.id]), pos, color, bytes, ends);
        else if (p!.spec.floor < op.floor) climb(anchor(p!, sel, 'top', op.inputRoles?.[p!.spec.id]), pos, color, bytes, ends, side(p!));
        else arc(anchor(p!, sel, 'top', op.inputRoles?.[p!.spec.id]), pos, color, bytes, ends);
      }
      const outColor = carriesStream ? RESIDUAL_COLOR : KIND_COLORS[out.spec.kind].edge;
      // The op sits right above its output (or right below it, across floors): a straight line into it.
      draw(new THREE.LineCurve3(pos, anchor(out, sel, crossFloor ? 'bottom' : 'top')), outColor, flowBytes(out, sel),
        { tensors: [out.spec.id], label: div, text: `${text} → ${out.spec.label}` });
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
      const div = label('next step: token → input_ids', new THREE.Vector3(left, (top + entry) / 2, b.z), 'op-label',
        'The sampled token id is the whole input of the next decode step: 8 bytes go around the loop, while every weight is read again.');
      draw(path, LOOP_COLOR, byteCount(from.spec),
        { tensors: [from.spec.id, to.spec.id], label: div, text: `${from.spec.label} → next step's ${to.spec.label}` }, this.loopMaterial);
    }

    if (this.edges.length === 0) return;
    this.dots = new THREE.InstancedMesh(this.dotGeometry, this.dotMaterial, this.edges.length * DOTS_PER_EDGE);
    this.dots.frustumCulled = false;
    // Each dot takes the color of the pipe it flows along.
    const dotColor = new THREE.Color();
    this.edges.forEach((edge, e) => {
      for (let d = 0; d < DOTS_PER_EDGE; d++) this.dots!.setColorAt(e * DOTS_PER_EDGE + d, dotColor.setHex(edge.color));
    });
    this.group.add(this.dots);
  }

  /**
   * The pipe nearest the pointer, in screen pixels, if within PICK_PIXELS (or its own half-width, for wide pipes).
   * `x`, `y`: pointer position in CSS pixels; the camera's projection covers the whole window.
   */
  pickEdge(camera: THREE.PerspectiveCamera, x: number, y: number): EdgeHit | null {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const toScreen = (p: THREE.Vector3, out: THREE.Vector2) => {
      this.projected.copy(p).project(camera);
      // Behind the camera, or beyond the far plane.
      if (Math.abs(this.projected.z) > 1) return false;
      out.set(((this.projected.x + 1) / 2) * w, ((1 - this.projected.y) / 2) * h);
      return true;
    };
    // Pixels per world unit at distance 1: a pipe of radius r at distance d spans r · scale / d pixels.
    const scale = h / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
    const a = new THREE.Vector2();
    const b = new THREE.Vector2();
    let best: { edge: Edge; pixels: number; point: THREE.Vector3 } | null = null;
    for (const edge of this.edges) {
      for (let i = 1; i < edge.points.length; i++) {
        if (!toScreen(edge.points[i - 1], a) || !toScreen(edge.points[i], b)) continue;
        // Distance from the pointer to the segment a–b, and where along it.
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const len2 = dx * dx + dy * dy;
        const t = len2 > 0 ? Math.max(0, Math.min(1, ((x - a.x) * dx + (y - a.y) * dy) / len2)) : 0;
        const pixels = Math.hypot(x - (a.x + t * dx), y - (a.y + t * dy));
        const radius = ((edge.diameter / 2) * scale) / edge.points[i].distanceTo(camera.position);
        if (pixels > Math.max(PICK_PIXELS, radius) || (best && pixels >= best.pixels)) continue;
        best = { edge, pixels, point: edge.points[i - 1].clone().lerp(edge.points[i], t) };
      }
    }
    return best && { edge: best.edge, distance: best.point.distanceTo(camera.position) };
  }

  /** Draws `edges` (or none, with null) as hovered: bright pipes and lit op labels. */
  setHover(edges: Edge | Edge[] | null): void {
    const next = edges === null ? [] : Array.isArray(edges) ? edges : [edges];
    if (next.length === this.hovered.length && next.every((e, i) => e === this.hovered[i])) return;
    for (const edge of this.hovered) {
      edge.tube.material = edge.material;
      edge.label?.classList.remove('highlight');
    }
    this.hovered = next;
    for (const edge of next) {
      edge.tube.material = this.hoverMaterial;
      edge.label?.classList.add('highlight');
    }
  }

  /** The pipes into and out of an op: every pipe whose op end is the label `div`. */
  pipesOf(div: HTMLElement): Edge[] {
    return this.edges.filter((e) => e.label === div);
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
    if (!m) this.materials.set(color, (m = new THREE.MeshBasicMaterial({ color, ...PIPE_MATERIAL })));
    return m;
  }

  private clear(): void {
    this.hovered = [];
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
