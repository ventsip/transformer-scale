import * as THREE from 'three/webgpu';
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { CameraRig, box3Of } from './camera';
import { FOCUS_VIEW_SLOPE, STATION_GAP, enclosingBox, layoutTensors, type Box, type PlacedTensor } from './layout';
import { MODELS, type ModelId } from './model/config';
import { PALETTE, applyPalette } from './palette';
import { summarize, type Stats } from './model/stats';
import { buildModel, floorCount, floorName, isTransient, selectionFor, type HeadSelection, type TensorSpec } from './model/tensors';
import { FlowScene, linkedTensors } from './scene/flowScene';
import { TensorScene } from './scene/tensorScene';
import { SectionScene } from './scene/sectionScene';
import { floorParts, type Part } from './model/sections';
import { renderLegend, renderStats as renderStatsPanel } from './statsPanel';
import { Tooltips } from './tooltips';
import { createControls, maxTokens, type Settings } from './ui/controls';
import { setPopupContext } from './ui/opPopup';

applyPalette();
/** Lighting: a white sky over a dark blue ground, and a sun from above, front right. */
const LIGHT = { ambient: 1.6, sun: 1.8, sunDir: new THREE.Vector3(0.4, 1, 0.7) };
/** Floor labels sit this many station gaps left of the floors. */
const FLOOR_LABEL_INSET = 4;
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
  mlaCache: 'hf',
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
scene.background = new THREE.Color(PALETTE.background);
scene.add(new THREE.HemisphereLight(PALETTE.highlight, PALETTE.ground, LIGHT.ambient));
const sun = new THREE.DirectionalLight(PALETTE.highlight, LIGHT.sun);
sun.position.copy(LIGHT.sunDir);
scene.add(sun);

const rig = new CameraRig(renderer.domElement);
const { camera, controls } = rig;

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
/** Selected expert of a mixture of experts; it applies to every layer. */
let expertSelection: number | null = null;

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
  const loop: [string, string] = [model.loop.output, model.loop.input];
  if (settings.showFlow) loop.forEach((id) => linked.add(id));
  tensors.update(placed.filter((p) => isVisible(p.spec)), settings.focusFloor, selection, expertSelection);
  flow.update(model.ops, placed, settings.showFlow ? settings.focusFloor : null, selection, loop);
  floorLabels.forEach((obj, f) => {
    obj.position.set(-FLOOR_LABEL_INSET * STATION_GAP, layout.floorY(f), 0);
    obj.element.classList.toggle('focused', f === settings.focusFloor);
  });
  renderStats();
  document.getElementById('title')!.textContent = `${cfg.name} · ${settings.mode} · ${backend}`;
}

const overview = (instant = false) => rig.overview(placed.flatMap((p) => p.boxes), instant);
// Front and above: the focus gap keeps the floors overhead out of the line of sight.
const frameFocus = () =>
  rig.frame(box3Of(placed.filter((p) => p.spec.floor === settings.focusFloor && isVisible(p.spec)).flatMap((p) => p.boxes)), new THREE.Vector3(0, FOCUS_VIEW_SLOPE, 1));

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
const renderStats = () => renderStatsPanel(statsEl, { cfg, settings, stats, expertTokens, selection, expertSelection });
renderLegend(document.getElementById('legend')!);
const controlsEl = document.getElementById('controls')!;

/** Builds the settings panel for the current model (its context length sets the token range). */
const createUi = () => {
  controlsEl.replaceChildren();
  return createControls(controlsEl, settings, {
    maxT: cfg.maxContext,
    mla: !!cfg.mla,
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
  expertSelection = null;
  playing = false;
  createFloorLabels();
  ui = createUi();
  rebuild();
  overview();
}

// Hover and click.
const tooltip = new Tooltips(document.getElementById('tooltip')!, () => ({
  cfg, mode: settings.mode, T: settings.T, focusFloor: settings.focusFloor, stats, parts, expertTokens,
}));
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function pickAt(e: MouseEvent) {
  pointer.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  return tensors.pick(raycaster);
}

/** The whole extent of a tensor: all its tiles and head bands in one box. */
function extentOf(id: string): Box | null {
  const p = placed.find((q) => q.spec.id === id);
  return p ? enclosingBox(p.boxes) : null;
}

// Capture phase: runs before OrbitControls handles the same events. Right or middle button pans, and so does the left
// button with Ctrl, Meta or Shift held.
/** Moves the orbit pivot to the depth of the block under the cursor, or else of the focused floor. */
const pivotUnderCursor = (e: MouseEvent) => rig.pivotAt(raycaster.ray, pickAt(e)?.distance ?? null, focusY);
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
    tooltip.edge(pipe.edge, e);
    return;
  }
  flow.setHover(null);
  tensors.setHighlight(block ? [block.box] : []);
  // Over empty floor: the tray of the section under the pointer, if any.
  const part = block ? null : sections.pick(raycaster);
  sections.setHover(part);
  if (block) tooltip.tensor(block, e);
  else if (part) tooltip.part(part, e);
  else tooltip.hide();
});
renderer.domElement.addEventListener('pointerleave', () => {
  sections.setHover(null);
  flow.setHover(null);
  tensors.setHighlight([]);
  tooltip.hide();
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
  // An expert's tensor or router column selects that expert; empty space clears it too.
  const nextExpert = p && p.expert !== null ? p.expert : p ? expertSelection : null;
  const selectionChanged = JSON.stringify(nextSelection) !== JSON.stringify(selection) || nextExpert !== expertSelection;
  selection = nextSelection;
  expertSelection = nextExpert;
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
  if (item) rig.frameBox(item.boxes[lastClick!.tile] ?? item.boxes[0]);
});

window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement && e.target.type === 'text') return;
  if (e.key === 'f' || e.key === 'F') frameFocus();
  if (e.key === 'o' || e.key === 'O') overview();
  if (e.key === 'Escape' && (selection || expertSelection !== null)) {
    selection = null;
    expertSelection = null;
    rebuild();
  }
});

window.addEventListener('resize', () => {
  rig.updateProjection();
  renderer.setSize(window.innerWidth, window.innerHeight);
  labelRenderer.setSize(window.innerWidth, window.innerHeight);
});

panel.hidden = false;
rig.updateProjection();

rebuild();
overview(true);

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  stepPlayback();
  flow.tick(clock.getElapsedTime());
  rig.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
});
