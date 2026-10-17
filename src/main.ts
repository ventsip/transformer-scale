import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { formatBytes, formatCount, formatInt } from './format';
import { STATION_GAP, THICKNESS_PER_BYTE, floorY, layoutTensors, type PlacedTensor } from './layout';
import { LLAMA_3_8B } from './model/config';
import { DTYPES } from './model/dtypes';
import { summarize, type Stats } from './model/stats';
import { buildTensors, byteCount, elementCount, floorCount, floorName, isTransient, type TensorSpec } from './model/tensors';
import { KIND_COLORS, TensorScene, type Pick } from './scene/tensorScene';
import { createControls, type Settings } from './ui/controls';

const cfg = LLAMA_3_8B;
const FLOORS = floorCount(cfg);

const settings: Settings = {
  T: 2048,
  focusFloor: 1,
  activations: 'focused',
  showScores: false,
  thickness: 'bytes',
  weightDtype: 'bf16',
  actDtype: 'bf16',
  kvDtype: 'bf16',
  floorSpacing: 8000,
};

// Renderers.
const app = document.getElementById('app')!;
const renderer = new THREE.WebGPURenderer({ antialias: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
app.appendChild(renderer.domElement);
await renderer.init();

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
controls.maxDistance = 5e6;

const tensors = new TensorScene();
scene.add(tensors.group);

// Floor labels.
const floorLabels = Array.from({ length: FLOORS }, (_, f) => {
  const div = document.createElement('div');
  div.className = 'floor-label';
  div.textContent = floorName(cfg, f);
  const obj = new CSS2DObject(div);
  obj.center.set(1, 0.5);
  scene.add(obj);
  return obj;
});

// State derived from settings.
let specs: TensorSpec[] = [];
let placed: PlacedTensor[] = [];
let stats: Stats;

const isVisible = (s: TensorSpec) =>
  !isTransient(s) || settings.activations === 'all' || s.floor === settings.focusFloor;

function rebuild() {
  specs = buildTensors(cfg, settings);
  stats = summarize(specs);
  placed = layoutTensors(specs, settings);
  tensors.update(placed.filter((p) => isVisible(p.spec)), settings.focusFloor);
  floorLabels.forEach((obj, f) => {
    obj.position.set(-STATION_GAP, floorY(f, settings), 0);
    obj.element.classList.toggle('focused', f === settings.focusFloor);
  });
  renderStats();
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
function frame(box: THREE.Box3, dir: THREE.Vector3) {
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
  controls.target.copy(center);
  camera.position.copy(center).addScaledVector(back, dist * 1.05);
  controls.update();
}

const overview = () => frame(boundsOf(placed), new THREE.Vector3(0.45, 0.35, 1));
// Front and above: the focus gap keeps the floors overhead out of the line of sight.
const frameFocus = () =>
  frame(boundsOf(placed.filter((p) => p.spec.floor === settings.focusFloor && isVisible(p.spec))), new THREE.Vector3(0, 1.5, 1));

// UI.
const panel = document.getElementById('panel')!;
const statsEl = document.getElementById('stats')!;
const ui = createControls(document.getElementById('controls')!, settings, {
  maxT: cfg.maxContext,
  floorCount: FLOORS,
  floorName: (f) => floorName(cfg, f),
  onChange: rebuild,
  onOverview: overview,
  onFrameFocus: frameFocus,
});

function renderStats() {
  const kvPerToken = stats.kvBytes / settings.T;
  const rows: [string, string][] = [
    ['Parameters', `${formatCount(stats.params)} · ${formatBytes(stats.paramBytes)}`],
    [`KV cache @ ${formatInt(settings.T)} tokens`, `${formatCount(stats.kvElements)} · ${formatBytes(stats.kvBytes)}`],
    ['KV cache per token', formatBytes(kvPerToken)],
    ['Transient activations, one layer', formatBytes(stats.layerActBytes)],
  ];
  if (settings.showScores) rows.push(['Attention scores, one layer', formatBytes(stats.scoresBytes)]);
  statsEl.replaceChildren(
    ...rows.map(([k, v]) => {
      const div = document.createElement('div');
      div.className = 'stat';
      div.innerHTML = `<span>${k}</span><b>${v}</b>`;
      return div;
    }),
  );
}

document.getElementById('legend')!.innerHTML = [
  ['weight', 'Weights'],
  ['activation', 'Transient activations'],
  ['kv', 'K, V: KV cache'],
  ['scores', 'Attention scores (not stored)'],
]
  .map(([k, label]) => `<span><i style="background:#${KIND_COLORS[k as keyof typeof KIND_COLORS].fill.toString(16).padStart(6, '0')}"></i>${label}</span>`)
  .join('') +
  `<p>1 unit = 1 element along both floor axes. Thickness: 1 byte = ${THICKNESS_PER_BYTE} units (exaggerated).
  Weights: input dim → X, output dim → Z. Activations: features → X, tokens → Z.</p>`;

// Hover and click.
const tooltip = document.getElementById('tooltip')!;
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let hovered: Pick | null = null;

function pickAt(e: PointerEvent): Pick | null {
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

function showTooltip(p: Pick, e: PointerEvent) {
  const s = p.item.spec;
  const dims = s.dims.map((d) => `${d.name} ${formatInt(d.size)}`).join(' × ');
  tooltip.innerHTML = `
    <div class="tt-title">${s.label}</div>
    <div class="tt-kind" style="color:#${KIND_COLORS[s.kind].edge.toString(16)}">${s.kind} · ${DTYPES[s.dtype].label} · ${floorName(cfg, s.floor)}</div>
    <div>${dims}</div>
    <div><b>${formatCount(elementCount(s))}</b> elements · <b>${formatBytes(byteCount(s))}</b></div>
    <div class="tt-share">${share(s)}</div>
    ${s.note ? `<div class="tt-note">${s.note}</div>` : ''}`;
  tooltip.hidden = false;
  const x = Math.min(e.clientX + 16, window.innerWidth - tooltip.offsetWidth - 8);
  const y = Math.min(e.clientY + 16, window.innerHeight - tooltip.offsetHeight - 8);
  tooltip.style.transform = `translate(${x}px, ${y}px)`;
}

renderer.domElement.addEventListener('pointermove', (e) => {
  hovered = pickAt(e);
  tensors.setHighlight(hovered?.box ?? null);
  if (hovered) showTooltip(hovered, e);
  else tooltip.hidden = true;
});
renderer.domElement.addEventListener('pointerleave', () => {
  tensors.setHighlight(null);
  tooltip.hidden = true;
});

let downAt: { x: number; y: number } | null = null;
renderer.domElement.addEventListener('pointerdown', (e) => (downAt = { x: e.clientX, y: e.clientY }));
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) > 4) return;
  const p = pickAt(e);
  if (p && p.item.spec.floor !== settings.focusFloor) {
    settings.focusFloor = p.item.spec.floor;
    ui.refresh();
    rebuild();
  }
});

window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
  if (e.key === 'f' || e.key === 'F') frameFocus();
  if (e.key === 'o' || e.key === 'O') overview();
});

window.addEventListener('resize', () => {
  updateProjection();
  renderer.setSize(window.innerWidth, window.innerHeight);
  labelRenderer.setSize(window.innerWidth, window.innerHeight);
});

const backend = (renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend ? 'WebGPU' : 'WebGL2';
document.getElementById('title')!.textContent = `${cfg.name} · prefill · ${backend}`;
panel.hidden = false;
updateProjection();

rebuild();
overview();

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
});
