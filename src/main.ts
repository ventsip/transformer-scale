import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { formatBytes, formatCount, formatFlops, formatInt } from './format';
import { FOCUS_VIEW_SLOPE, STATION_GAP, THICKNESS_PER_BYTE, layoutTensors, type Box, type PlacedTensor } from './layout';
import { MODELS, type ModelId } from './model/config';
import { DTYPES } from './model/dtypes';
import { summarize, type Stats } from './model/stats';
import {
  buildModel, byteCount, elementCount, floorCount, floorName, isTransient, kvGroup, selectionFor,
  type HeadSelection, type TensorSpec,
} from './model/tensors';
import { FlowScene, LOOP_COLOR, RESIDUAL_COLOR, linkedTensors, pipeDiameter, type Edge } from './scene/flowScene';
import { KIND_COLORS, MASKED_COLORS, ROLE_COLORS, TensorScene, type Pick } from './scene/tensorScene';
import { SectionScene } from './scene/sectionScene';
import { floorParts, type Part } from './model/sections';
import { createControls, maxTokens, type Settings } from './ui/controls';
import { setPopupContext } from './ui/opPopup';

const params = new URLSearchParams(location.search);
// ?model=mixtral-8x7b opens with that model.
const initialModel = (params.get('model') ?? '') in MODELS ? (params.get('model') as ModelId) : 'llama-3-8b';
let cfg = MODELS[initialModel];

const settings: Settings = {
  model: initialModel,
  mode: 'prefill',
  T: 2048,
  focusFloor: 1,
  activations: 'focused',
  attention: 'fused',
  kvCache: 'dynamic',
  elementwise: 'compact',
  cacheCapacity: cfg.maxContext,
  showFlow: true,
  thickness: 'bytes',
  weightDtype: 'bf16',
  actDtype: 'bf16',
  kvDtype: 'bf16',
  floorSpacing: 8000,
  playSpeed: '4',
};
// The opening model may have a shorter context than the default prompt (GPT-2: 1,024).
settings.T = Math.min(settings.T, maxTokens(settings, cfg.maxContext));

// Renderers.
const app = document.getElementById('app')!;
// ?webgl forces the WebGL2 fallback (for testing, or where WebGPU misbehaves).
const forceWebGL = params.has('webgl');
const renderer = new THREE.WebGPURenderer({ antialias: true, logarithmicDepthBuffer: true, forceWebGL });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
app.appendChild(renderer.domElement);
await renderer.init();
const backend = (renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend ? 'WebGPU' : 'WebGL2';

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(window.innerWidth, window.innerHeight);
labelRenderer.domElement.className = 'labels';
app.appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d12);
scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.8);
sun.position.set(0.4, 1, 0.7);
scene.add(sun);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 1, 1e8);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.zoomToCursor = true;
controls.zoomSpeed = 4;
controls.maxDistance = 5e6;

const tensors = new TensorScene();
scene.add(tensors.group);
// Trays under the sections of the focused floor.
const sections = new SectionScene();
scene.add(sections.group);
const flow = new FlowScene((f) => floorName(cfg, f), renderer.domElement);
scene.add(flow.group);
// For poking at the scene from the console during development.
if (import.meta.env.DEV) Object.assign(window, { camera, controls, flow, placedTensors: () => placed });

// Floor labels, recreated when the model changes.
let floorLabels: CSS2DObject[] = [];

function createFloorLabels() {
  floorLabels.forEach((obj) => {
    scene.remove(obj);
    obj.element.remove();
  });
  floorLabels = Array.from({ length: floorCount(cfg) }, (_, f) => {
    const div = document.createElement('div');
    div.className = 'floor-label';
    div.textContent = floorName(cfg, f);
    const obj = new CSS2DObject(div);
    obj.center.set(1, 0.5);
    scene.add(obj);
    return obj;
  });
}
createFloorLabels();

// State derived from settings.
let specs: TensorSpec[] = [];
let placed: PlacedTensor[] = [];
/** Height of the focused floor's base: the fallback depth for zoom and pan (see pivotUnderCursor). */
let focusY = 0;
let stats: Stats;
/** Mixture of experts: tokens routed to each expert, per decoder layer. */
let expertTokens: number[][] | undefined;
/** Sections of the focused floor (attention, MLP or router and experts), for its trays. */
let parts: Part[] = [];

/** Tensors on other floors that the focused floor's data flow links to. */
let linked = new Set<string>();
/** Selected attention head; it applies to every layer. */
let selection: HeadSelection | null = null;

const isVisible = (s: TensorSpec) =>
  !isTransient(s) || settings.activations === 'all' || s.floor === settings.focusFloor || linked.has(s.id);

function rebuild() {
  const model = buildModel(cfg, settings);
  specs = model.tensors;
  stats = summarize(model, cfg);
  expertTokens = model.expertTokens;
  setPopupContext(cfg, settings);
  const layout = layoutTensors(specs, settings);
  placed = layout.placed;
  focusY = layout.floorY(settings.focusFloor);
  parts = floorParts(cfg, model, settings.focusFloor);
  sections.update(parts, placed);
  const floorOf = new Map(specs.map((s) => [s.id, s.floor]));
  linked = settings.showFlow ? linkedTensors(model.ops, (id) => floorOf.get(id)!, settings.focusFloor) : new Set();
  // The generation loop: the sampled token feeds the next step's input. Both ends stay visible.
  const loop: [string, string] = [`${cfg.nLayers + 2}:next_token`, '0:input_ids'];
  if (settings.showFlow) loop.forEach((id) => linked.add(id));
  tensors.update(placed.filter((p) => isVisible(p.spec)), settings.focusFloor, selection);
  flow.update(model.ops, placed, settings.showFlow ? settings.focusFloor : null, selection, loop);
  floorLabels.forEach((obj, f) => {
    obj.position.set(-4 * STATION_GAP, layout.floorY(f), 0);
    obj.element.classList.toggle('focused', f === settings.focusFloor);
  });
  renderStats();
  document.getElementById('title')!.textContent = `${cfg.name} · ${settings.mode} · ${backend}`;
}

// Camera framing.
function boundsOf(items: PlacedTensor[]): THREE.Box3 {
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  for (const { boxes } of items) {
    for (const b of boxes) {
      box.expandByPoint(v.set(b.cx - b.sx / 2, b.cy - b.sy / 2, b.cz - b.sz / 2));
      box.expandByPoint(v.set(b.cx + b.sx / 2, b.cy + b.sy / 2, b.cz + b.sz / 2));
    }
  }
  return box;
}

/** Width of the screen covered by the settings panel, kept out of the camera's center. */
function panelInset(): number {
  const panel = document.getElementById('panel')!;
  if (panel.hidden || window.innerWidth < 700) return 0;
  return panel.getBoundingClientRect().width + 24;
}

function updateProjection() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  // Shift the center of projection to the middle of the area left of the panel.
  camera.setViewOffset(w, h, panelInset() / 2, 0, w, h);
  camera.updateProjectionMatrix();
}

/** Places the camera along `dir` from the box center, at the distance where every corner fits the visible area. */
function frame(box: THREE.Box3, dir: THREE.Vector3, instant = false) {
  const center = box.getCenter(new THREE.Vector3());
  const back = dir.clone().normalize();
  const right = new THREE.Vector3(0, 1, 0).cross(back).normalize();
  const up = back.clone().cross(right);
  const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  const tanH = (tanV * (window.innerWidth - panelInset())) / window.innerHeight;
  const corner = new THREE.Vector3();
  let dist = 0;
  for (let i = 0; i < 8; i++) {
    corner.set(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z).sub(center);
    const z = corner.dot(back);
    dist = Math.max(dist, Math.abs(corner.dot(right)) / tanH + z, Math.abs(corner.dot(up)) / tanV + z);
  }
  flyTo(center.clone().addScaledVector(back, dist * 1.05), center, instant);
}

// Camera flights: eased interpolation of position and orbit target; any user input cancels them.
const FLIGHT_SECONDS = 0.8;
let flight: { from: THREE.Vector3; fromTarget: THREE.Vector3; to: THREE.Vector3; toTarget: THREE.Vector3; start: number } | null = null;

function flyTo(position: THREE.Vector3, target: THREE.Vector3, instant = false) {
  if (instant) {
    flight = null;
    camera.position.copy(position);
    controls.target.copy(target);
    return;
  }
  flight = {
    from: camera.position.clone(), fromTarget: controls.target.clone(),
    to: position, toTarget: target, start: performance.now(),
  };
}

function updateFlight() {
  if (!flight) return;
  const t = Math.min(1, (performance.now() - flight.start) / (FLIGHT_SECONDS * 1000));
  const e = t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2;
  camera.position.lerpVectors(flight.from, flight.to, e);
  controls.target.lerpVectors(flight.fromTarget, flight.toTarget, e);
  if (t === 1) flight = null;
}

controls.addEventListener('start', () => (flight = null));

/** Frames a single box from the current viewing direction, kept at least slightly above the floor. */
function frameBox(b: Box) {
  const box = new THREE.Box3(
    new THREE.Vector3(b.cx - b.sx / 2, b.cy - b.sy / 2, b.cz - b.sz / 2),
    new THREE.Vector3(b.cx + b.sx / 2, b.cy + b.sy / 2, b.cz + b.sz / 2),
  );
  const dir = camera.position.clone().sub(controls.target).normalize();
  if (dir.y < 0.35) dir.setY(0.35).normalize();
  frame(box, dir);
}

const overview = (instant = false) => frame(boundsOf(placed), new THREE.Vector3(0.45, 0.35, 1), instant);
// Front and above: the focus gap keeps the floors overhead out of the line of sight.
const frameFocus = () =>
  frame(boundsOf(placed.filter((p) => p.spec.floor === settings.focusFloor && isVisible(p.spec))), new THREE.Vector3(0, FOCUS_VIEW_SLOPE, 1));

// Generation playback: each decode step adds one token to the cache.
let playing = false;
let playClock = 0;

function togglePlay() {
  playing = !playing;
  playClock = performance.now();
  if (playing && settings.mode !== 'decode') {
    settings.mode = 'decode';
    settings.T = Math.min(settings.T, maxTokens(settings, cfg.maxContext));
    rebuild();
  }
}

function stepPlayback() {
  if (!playing) return;
  const steps = Math.floor(((performance.now() - playClock) / 1000) * Number(settings.playSpeed));
  if (steps < 1) return;
  playClock += (steps / Number(settings.playSpeed)) * 1000;
  const max = maxTokens(settings, cfg.maxContext);
  settings.T = Math.min(max, settings.T + steps);
  if (settings.T === max) playing = false;
  ui.refresh();
  rebuild();
  flow.pulse();
}

// UI.
const panel = document.getElementById('panel')!;
const statsEl = document.getElementById('stats')!;
const controlsEl = document.getElementById('controls')!;

/** Builds the settings panel for the current model (its context length sets the token range). */
const createUi = () => {
  controlsEl.replaceChildren();
  return createControls(controlsEl, settings, {
    maxT: cfg.maxContext,
    floorCount: floorCount(cfg),
    floorName: (f) => floorName(cfg, f),
    onChange: (key) => {
      if (key === 'model') return switchModel();
      // Switching to decode: the cache plus the new token must still fit the context.
      if (key === 'mode') {
        settings.T = Math.min(settings.T, maxTokens(settings, cfg.maxContext));
        ui.refresh();
      }
      rebuild();
    },
    onOverview: () => overview(),
    onFrameFocus: frameFocus,
    onPlay: togglePlay,
    isPlaying: () => playing,
  });
};
let ui = createUi();

function switchModel() {
  cfg = MODELS[settings.model];
  settings.cacheCapacity = cfg.maxContext;
  settings.T = Math.min(settings.T, maxTokens(settings, cfg.maxContext));
  settings.focusFloor = Math.min(settings.focusFloor, floorCount(cfg) - 1);
  selection = null;
  playing = false;
  createFloorLabels();
  ui = createUi();
  rebuild();
  overview();
}

function describeSelection(sel: HeadSelection): string {
  const perGroup = cfg.nHeads / cfg.nKvHeads;
  if (perGroup === 1) return `Head ${sel.group}, with its own K and V`;
  const first = sel.group * perGroup;
  const group = `KV group ${sel.group}`;
  return sel.head === null ? `${group} · Q heads ${first}–${first + perGroup - 1}` : `Q head ${sel.head} · ${group}`;
}

/** Mixture of experts: active parameters, and how the focused layer routes this pass's tokens. */
function expertRows(): [string, string][] {
  const { nExperts, topK } = cfg.moe!;
  const rows: [string, string][] = [[`Active per token (${topK} of ${nExperts} experts)`, formatCount(stats.activeParams)]];
  const counts = expertTokens?.[settings.focusFloor - 1];
  if (!counts) return rows;
  const layer = floorName(cfg, settings.focusFloor);
  if (settings.mode === 'decode') {
    const used = counts.flatMap((c, e) => (c > 0 ? [e] : []));
    rows.push([`Experts used, ${layer}`, used.join(' and ')]);
  } else {
    rows.push([`Tokens per expert, ${layer}`, counts.map(formatInt).join(' · ')]);
  }
  return rows;
}

function renderStats() {
  const isStatic = settings.kvCache === 'static';
  const decode = settings.mode === 'decode';
  // Tokens in the cache once this pass is done.
  const cached = settings.T + (decode ? 1 : 0);
  const kvPerToken = stats.kvBytes / (isStatic ? settings.cacheCapacity : cached);
  const pass = decode ? 'per generated token' : 'for the prompt';
  const rows: [string, string][] = [
    ['Parameters', `${formatCount(stats.params)} · ${formatBytes(stats.paramBytes)}`],
    ...(cfg.moe ? expertRows() : []),
    [isStatic ? `KV cache, static: ${formatInt(settings.cacheCapacity)} slots` : `KV cache @ ${formatInt(cached)} tokens`,
      `${formatCount(stats.kvElements)} · ${formatBytes(stats.kvBytes)}`],
    ['KV cache per token', formatBytes(kvPerToken)],
    ...(isStatic ? [['KV cache filled', `${formatInt(cached)} of ${formatInt(settings.cacheCapacity)} slots`] as [string, string]] : []),
    ['Transient activations, one layer', formatBytes(stats.layerActBytes)],
    [`FLOPs ${pass}`, formatFlops(stats.flops)],
    [`Weights + KV read ${pass}`, formatBytes(stats.bytesRead)],
    ['FLOPs per byte read', (stats.flops / stats.bytesRead).toPrecision(3)],
  ];
  if (settings.attention === 'eager') rows.push(['Eager score tensors, one layer', formatBytes(stats.scoresBytes)]);
  if (selection) rows.push(['Selected head (Esc clears)', describeSelection(selection)]);
  statsEl.replaceChildren(
    ...rows.map(([k, v]) => {
      const div = document.createElement('div');
      div.className = v.length > 28 ? 'stat wide' : 'stat';
      div.innerHTML = `<span>${k}</span><b>${v}</b>`;
      return div;
    }),
  );
}

const widthExamples = ([['8 KiB', 2 ** 13], ['16 MiB', 2 ** 24], ['1 GiB', 2 ** 30]] as const)
  .map(([name, bytes]) => `${name} ≈ ${formatInt(Math.round(pipeDiameter(bytes)))}`)
  .join(', ');
const swatch = (hex: number, label: string) =>
  `<span><i style="background:#${hex.toString(16).padStart(6, '0')}"></i>${label}</span>`;
document.getElementById('legend')!.innerHTML = [
  swatch(KIND_COLORS.weight.fill, 'Weights'),
  swatch(KIND_COLORS.activation.fill, 'Transient activations'),
  swatch(KIND_COLORS.kv.fill, 'K, V: KV cache'),
  swatch(ROLE_COLORS.q, 'W_q: query projection'),
  swatch(ROLE_COLORS.k, 'W_k: key projection'),
  swatch(ROLE_COLORS.v, 'W_v: value projection'),
  swatch(RESIDUAL_COLOR, 'Residual stream'),
  swatch(LOOP_COLOR, 'Generation loop'),
  swatch(KIND_COLORS.buffer.fill, 'Buffers (fixed, not learned)'),
  swatch(KIND_COLORS.scores.fill, 'Attention scores (eager only)'),
  swatch(MASKED_COLORS.fill, 'Causally masked scores'),
].join('') +
  `<p>1 unit = 1 element along both floor axes. Thickness: 1 byte = ${THICKNESS_PER_BYTE} units (exaggerated).
  Weights: input dim → X, output dim → Z. Activations: features → X, tokens → Z.
  Flow pipe diameter grows with the log of the bytes carried (in world units, so with perspective): ${widthExamples}.</p>`;

// Hover and click.
const tooltip = document.getElementById('tooltip')!;
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function pickAt(e: MouseEvent) {
  pointer.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  return tensors.pick(raycaster);
}

function share(s: TensorSpec): string {
  const b = byteCount(s);
  const pct = (x: number, of: number) => `${((100 * x) / of).toPrecision(3)}%`;
  if (s.kind === 'weight') return `${pct(elementCount(s), stats.params)} of all parameters`;
  if (s.kind === 'kv') return `${pct(b, stats.kvBytes)} of the KV cache`;
  if (s.kind === 'activation' && s.floor >= 1 && s.floor <= cfg.nLayers) return `${pct(b, stats.layerActBytes)} of this layer's transient activations`;
  return '';
}

function maskLine({ part }: Pick): string {
  const T = settings.T;
  if (part === 'kept') return `<div class="tt-head">Kept by the causal mask (key ≤ query): <b>${formatCount((T * (T + 1)) / 2)}</b> entries per T×T tile</div>`;
  if (part === 'masked') return `<div class="tt-head">Causally masked (key > query): <b>${formatCount((T * (T - 1)) / 2)}</b> entries per T×T tile: −∞ before softmax, 0 after</div>`;
  return '';
}

function rowLine({ item, rows }: Pick): string {
  if (item.spec.filledZ === undefined) return '';
  const fresh = item.spec.freshZ ?? 0;
  const line = (text: string) => `<div class="tt-head">${text}</div>`;
  if (rows === 'reserved') return line(`Reserved, not yet written: <b>${formatInt(item.spec.z - item.spec.filledZ)}</b> of ${formatInt(item.spec.z)} slots`);
  if (rows === 'fresh') return line(`Written by this step: <b>${formatInt(fresh)}</b> row (the new token)`);
  return line(`Written by earlier passes: <b>${formatInt(item.spec.filledZ - fresh)}</b> rows`);
}

function headLine({ item, box, head, role }: Pick): string {
  const split = item.spec.heads;
  if (!split || head === null) return '';
  const band = box.sx * box.sz;
  const what = cfg.nHeads === cfg.nKvHeads
    ? `Head ${head} of ${split.count} (multi-head attention: its own K and V)`
    : split.perGroup === 1
    ? `KV head ${head} of ${split.count}, shared by Q heads ${head * (cfg.nHeads / cfg.nKvHeads)}–${(head + 1) * (cfg.nHeads / cfg.nKvHeads) - 1}`
    : `Q head ${head} of ${split.count} · KV group ${kvGroup(split, head)}`;
  // In a fused Q/K/V tensor, also which third the band is in.
  const part = item.spec.fusedRoles && role ? `${role.toUpperCase()} · ` : '';
  return `<div class="tt-head">${part}${what}: <b>${formatCount(band)}</b> elements · <b>${formatBytes(band * DTYPES[item.spec.dtype].bytes)}</b></div>`;
}

function placeTooltip(e: PointerEvent) {
  tooltip.hidden = false;
  const x = Math.min(e.clientX + 16, window.innerWidth - tooltip.offsetWidth - 8);
  const y = Math.min(e.clientY + 16, window.innerHeight - tooltip.offsetHeight - 8);
  tooltip.style.transform = `translate(${x}px, ${y}px)`;
}

/** Totals of a section of the focused floor: parameters and their share, memory, and this pass's work. */
function showPartTooltip(part: Part, e: PointerEvent) {
  // Shares are of the whole layer: its top-level parts (each expert is also inside "Experts").
  const layer = parts.filter((p) => p.kind !== 'expert');
  const sum = (f: (p: Part) => number) => layer.reduce((a, p) => a + f(p), 0);
  const pct = (x: number, of: number) => (of > 0 ? `${((100 * x) / of).toPrecision(3)}%` : '–');
  const pass = settings.mode === 'decode' ? 'this decode step' : 'the prompt';
  const routed = part.expert !== undefined ? expertTokens?.[settings.focusFloor - 1]?.[part.expert] : undefined;
  const rows: string[] = [
    `<div><b>${formatCount(part.params)}</b> parameters · <b>${formatBytes(part.weightBytes)}</b></div>`,
    `<div class="tt-head">${pct(part.params, sum((p) => p.params))} of this layer's parameters · ${pct(part.params, stats.params)} of the model's</div>`,
    `<div>Transient activations: <b>${formatBytes(part.activationBytes)}</b></div>`,
  ];
  if (part.kvBytes > 0) rows.push(`<div>KV cache: <b>${formatBytes(part.kvBytes)}</b></div>`);
  rows.push(
    `<div>FLOPs for ${pass}: <b>${formatFlops(part.flops)}</b> (${pct(part.flops, sum((p) => p.flops))} of the layer)</div>`,
    `<div>Weights + KV read: <b>${formatBytes(part.bytesRead)}</b> (${pct(part.bytesRead, sum((p) => p.bytesRead))} of the layer)</div>`,
  );
  if (routed !== undefined) {
    rows.push(`<div class="tt-note">${part.idle ? 'No token is routed to this expert in this pass: its weights are not read.'
      : `${formatInt(routed)} ${routed === 1 ? 'token is' : 'tokens are'} routed to this expert in this pass.`}</div>`);
  }
  tooltip.innerHTML = `<div class="tt-title">${part.name} · ${floorName(cfg, settings.focusFloor)}</div>
    <div class="tt-kind">section</div>${rows.join('')}`;
  placeTooltip(e);
}

function showEdgeTooltip(edge: Edge, e: PointerEvent) {
  tooltip.innerHTML = `
    <div class="tt-title">${edge.text}</div>
    <div>carries <b>${formatBytes(edge.bytes)}</b></div>
    <div class="tt-note">Pipe width grows with the log of the bytes: ${formatInt(Math.round(edge.diameter))} units across.</div>`;
  placeTooltip(e);
}

/** The whole extent of a tensor: all its tiles and head bands in one box. */
function extentOf(id: string): Box | null {
  const p = placed.find((q) => q.spec.id === id);
  if (!p) return null;
  const box = boundsOf([p]);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  return { cx: center.x, cy: center.y, cz: center.z, sx: size.x, sy: size.y, sz: size.z };
}

function showTooltip(p: Pick, e: PointerEvent) {
  const s = p.item.spec;
  const dims = s.dims.map((d) => `${d.name} ${formatInt(d.size)}`).join(' × ');
  tooltip.innerHTML = `
    <div class="tt-title">${s.label}</div>
    <div class="tt-kind" style="color:#${KIND_COLORS[s.kind].edge.toString(16)}">${s.kind} · ${DTYPES[s.dtype].label} · ${floorName(cfg, s.floor)}</div>
    <div>${dims}</div>
    <div><b>${formatCount(elementCount(s))}</b> elements · <b>${formatBytes(byteCount(s))}</b></div>
    ${headLine(p)}
    ${maskLine(p)}
    ${rowLine(p)}
    <div class="tt-share">${share(s)}</div>
    ${s.note ? `<div class="tt-note">${s.note}</div>` : ''}`;
  placeTooltip(e);
}

/**
 * Before a zoom or pan, moves the orbit pivot along the view direction to the depth of what is under the cursor:
 * a block, or else the focused floor's plane. OrbitControls scales both by the camera–pivot distance, so a pivot
 * left close to the camera (after zooming into empty space, or flying to a small block) made them crawl. The camera
 * keeps its position and orientation, so the view does not move.
 */
function pivotUnderCursor(e: MouseEvent) {
  if (flight) return;
  const block = pickAt(e);
  const point = block
    ? raycaster.ray.at(block.distance, new THREE.Vector3())
    : raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), -focusY), new THREE.Vector3());
  if (!point) return;
  const dir = camera.getWorldDirection(new THREE.Vector3());
  const depth = point.sub(camera.position).dot(dir);
  if (depth > 0) controls.target.copy(camera.position).addScaledVector(dir, depth);
}

// Capture phase: runs before OrbitControls handles the same events. Right or middle button pans, and so does the left
// button with Ctrl, Meta or Shift held.
renderer.domElement.addEventListener('wheel', pivotUnderCursor, { capture: true, passive: true });
renderer.domElement.addEventListener('pointerdown', (e) => {
  if (e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey) pivotUnderCursor(e);
}, { capture: true });

// Hover: a block, or a flow pipe with its endpoints (the tensor and the op), whichever is nearer.
renderer.domElement.addEventListener('pointermove', (e) => {
  const block = pickAt(e);
  const pipe = settings.showFlow ? flow.pickEdge(camera, e.clientX, e.clientY) : null;
  if (pipe && (!block || pipe.distance < block.distance)) {
    flow.setHover(pipe.edge);
    tensors.setHighlight(pipe.edge.tensors.map(extentOf).filter((b): b is Box => b !== null));
    showEdgeTooltip(pipe.edge, e);
    return;
  }
  flow.setHover(null);
  tensors.setHighlight(block ? [block.box] : []);
  // Over empty floor: the tray of the section under the pointer, if any.
  const part = block ? null : sections.pick(raycaster);
  sections.setHover(part);
  if (block) showTooltip(block, e);
  else if (part) showPartTooltip(part, e);
  else tooltip.hidden = true;
});
renderer.domElement.addEventListener('pointerleave', () => {
  sections.setHover(null);
  flow.setHover(null);
  tensors.setHighlight([]);
  tooltip.hidden = true;
});

// Hovering an op label lights all its pipes, in and out, and outlines the tensors at their ends.
flow.onOpHover = (pipes) => {
  flow.setHover(pipes);
  const ids = new Set((pipes ?? []).flatMap((p) => p.tensors));
  tensors.setHighlight([...ids].map(extentOf).filter((b): b is Box => b !== null));
};

// Click focuses a floor; double-click also flies to the element. The first click of a double-click
// may change the focus and move floors, so the double-click reuses that click's pick by id.
let downAt: { x: number; y: number } | null = null;
let lastClick: { id: string; tile: number } | null = null;
renderer.domElement.addEventListener('pointerdown', (e) => (downAt = { x: e.clientX, y: e.clientY }));
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) > 4) return;
  const p = pickAt(e);
  if (e.detail <= 1) lastClick = p ? { id: p.item.spec.id, tile: p.item.boxes.indexOf(p.box) } : null;
  // A head band selects its head (a KV band its whole group); empty space clears the selection.
  const split = p?.item.spec.heads;
  const nextSelection = p && split && p.head !== null ? selectionFor(split, p.head) : p ? selection : null;
  const selectionChanged = JSON.stringify(nextSelection) !== JSON.stringify(selection);
  selection = nextSelection;
  if (p && p.item.spec.floor !== settings.focusFloor) {
    settings.focusFloor = p.item.spec.floor;
    ui.refresh();
    rebuild();
  } else if (selectionChanged) {
    rebuild();
  }
});
renderer.domElement.addEventListener('dblclick', () => {
  const item = lastClick && placed.find((p) => p.spec.id === lastClick!.id);
  if (item) frameBox(item.boxes[lastClick!.tile] ?? item.boxes[0]);
});

window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement && e.target.type === 'text') return;
  if (e.key === 'f' || e.key === 'F') frameFocus();
  if (e.key === 'o' || e.key === 'O') overview();
  if (e.key === 'Escape' && selection) {
    selection = null;
    rebuild();
  }
});

window.addEventListener('resize', () => {
  updateProjection();
  renderer.setSize(window.innerWidth, window.innerHeight);
  labelRenderer.setSize(window.innerWidth, window.innerHeight);
});

panel.hidden = false;
updateProjection();

rebuild();
overview(true);

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  stepPlayback();
  flow.tick(clock.getElapsedTime());
  updateFlight();
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
});
