import { LLAMA_3_8B, type ModelConfig } from '../model/config';
import type { OpSpec } from '../model/tensors';

/** The current model: popups take its vocabulary, RoPE base, head and context sizes. */
let model: ModelConfig = LLAMA_3_8B;

export function setPopupModel(cfg: ModelConfig): void {
  model = cfg;
}

/** The stages of the sampling panel, in pipeline order. */
type Stage = 'logits' | 'temperature' | 'filter' | 'softmax' | 'sample';

/** Which sampling stage an op belongs to, from the tensor it produces; null for ops without a view. */
function samplingStage(op: OpSpec): Stage | null {
  const out = op.output.split(':')[1];
  if (out === 'logits_fp32') return 'logits';
  if (out === 'scaled') return 'temperature';
  if (out === 'filtered') return 'filter';
  if (out === 'probs' && op.label === 'softmax') return 'softmax';
  if (out === 'next_token') return 'sample';
  return null;
}

/** The stages of the SwiGLU panel. */
type GluStage = 'silu' | 'gate';

/** Which SwiGLU stage an op belongs to: compact `SiLU ⊙`, or expanded `SiLU` and `⊙`. */
function swigluStage(op: OpSpec): GluStage | null {
  if (op.label === 'SiLU ⊙' || op.label === 'SiLU') return 'silu';
  if (op.label === '⊙') return 'gate';
  return null;
}

/** The stages of the RoPE panel. */
type RopeStage = 'freq' | 'rotate';

/** The cos / sin tables on the embedding floor show the frequencies; the RoPE ops in each layer, the rotation. */
function ropeStage(op: OpSpec): RopeStage | null {
  if (op.label === 'RoPE') return 'rotate';
  const out = op.output.split(':')[1];
  return out === 'cos' || out === 'sin' ? 'freq' : null;
}

/** The stages of the RMSNorm panel. */
type NormStage = 'input' | 'mean' | 'normalize' | 'weight';

/**
 * Which RMSNorm stage an op belongs to, from its output: x_norm or h_norm (compact, or the final × weight), or one of
 * the expanded intermediates x_norm.fp32, .sq, .mean, .rsqrt, .n32, .n. Labels alone are ambiguous (`× weight`, `to fp32`).
 */
function normStage(op: OpSpec): NormStage | null {
  const m = /^[xh]_norm(?:\.(\w+))?$/.exec(op.output.split(':')[1]);
  if (!m) return null;
  const step = m[1];
  if (step === undefined) return op.label === 'RMSNorm' ? 'input' : 'weight';
  if (step === 'fp32') return 'input';
  if (step === 'sq' || step === 'mean' || step === 'rsqrt') return 'mean';
  return 'normalize';
}

/** The interactive view of an op: its title and how to fill the popup body; null for ops without one. */
function vizFor(op: OpSpec): { title: string; render: (body: HTMLElement) => void } | null {
  const sampling = samplingStage(op);
  if (sampling) return { title: 'Sampling: from logits to the next token', render: (body) => samplingViz(body, sampling) };
  const rope = ropeStage(op);
  if (rope) return { title: 'RoPE: rotary position embedding', render: (body) => ropeViz(body, rope) };
  const norm = normStage(op);
  if (norm) return { title: 'RMSNorm: x / RMS(x) · weight', render: (body) => rmsNormViz(body, norm) };
  const glu = swigluStage(op);
  if (glu) return { title: 'SwiGLU: a = SiLU(g) ⊙ u', render: (body) => swigluViz(body, glu) };
  return null;
}

export const hasViz = (op: OpSpec) => vizFor(op) !== null;

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, props: object = {}): HTMLElementTagNameMap[K] =>
  Object.assign(document.createElement(tag), props);
const SVG = 'http://www.w3.org/2000/svg';
const svg = <K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}) => {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
};

/** Opens the popup for `op`; Esc, the close button or a click outside closes it. */
export function openOpPopup(op: OpSpec): void {
  const viz = vizFor(op);
  if (!viz) return;
  document.querySelector('.op-popup-backdrop')?.remove();
  const backdrop = el('div', { className: 'op-popup-backdrop' });
  const popup = el('div', { className: 'op-popup' });
  const head = el('div', { className: 'op-popup-head' });
  const close = el('button', { className: 'op-popup-close', textContent: '×', title: 'Close (Esc)' });
  head.append(el('h2', { textContent: viz.title }), close);
  const body = el('div', { className: 'op-popup-body' });
  popup.append(head, body);
  backdrop.append(popup);
  document.body.append(backdrop);

  const onKey = (e: KeyboardEvent) => e.key === 'Escape' && dismiss();
  function dismiss() {
    backdrop.remove();
    window.removeEventListener('keydown', onKey, true);
  }
  close.addEventListener('click', dismiss);
  backdrop.addEventListener('pointerdown', (e) => e.target === backdrop && dismiss());
  window.addEventListener('keydown', onKey, true);
  viz.render(body);
}

/** A titled stage of a popup; the focused one (the clicked op's) is highlighted. */
function stageSection(body: HTMLElement, focused: boolean, title: string, text: string): HTMLElement {
  const s = el('section', { className: `op-stage${focused ? ' focus' : ''}` });
  s.append(el('h3', { textContent: title }), el('p', { className: 'op-popup-note', textContent: text }));
  body.append(s);
  return s;
}

// ---------------------------------------------------------------------------------------------
// Sampling: logits → ÷ temperature → top-k / top-p → softmax → sample

const SHOWN = 40;

/** Deterministic pseudo-random numbers, so the illustration is the same every time. */
function rng(seed: number) {
  return () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
}

/**
 * Illustrative last-position logits: 60 candidates falling from about 18 to about 0, steeply at first and
 * flattening out, with some jitter; then the long tail of the remaining vocabulary well below, around −8.
 * Sorted descending, so index = rank − 1.
 */
function illustrativeLogits(): Float64Array {
  const rand = rng(7);
  const normal = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const logits = new Float64Array(model.vocabSize);
  for (let r = 0; r < model.vocabSize; r++) {
    logits[r] = r < 60 ? 18 - 14 * Math.log(1 + r / 15) + 0.6 * normal() : -8 + 2 * normal();
  }
  return logits.sort().reverse();
}

/** softmax(x[0..n)) for x sorted descending. */
function softmaxPrefix(x: Float64Array, n: number): Float64Array {
  const out = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) sum += out[i] = Math.exp(x[i] - x[0]);
  for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

/** HF order: temperature, then top-k, then top-p on the softmax of what top-k kept. Returns how many tokens survive. */
function keptCount(scaled: Float64Array, k: number, p: number): { afterK: number; kept: number } {
  const afterK = k > 0 ? Math.min(k, scaled.length) : scaled.length;
  if (p >= 1) return { afterK, kept: afterK };
  const probs = softmaxPrefix(scaled, afterK);
  let cum = 0;
  let n = 0;
  while (n < afterK && cum < p) cum += probs[n++];
  return { afterK, kept: Math.max(1, n) };
}

function samplingViz(body: HTMLElement, focus: Stage): void {
  const logits = illustrativeLogits();
  const input = Array.from(logits.subarray(0, SHOWN));
  const sections = new Map<Stage, HTMLElement>();
  const section = (stage: Stage, title: string, text: string) => {
    const s = stageSection(body, stage === focus, title, text);
    sections.set(stage, s);
    return s;
  };
  const slider = (label: string, min: number, max: number, step: number, value: number) => {
    const input = el('input', { type: 'range', min: String(min), max: String(max), step: String(step), value: String(value) });
    const shown = el('b');
    const control = el('label', { className: 'op-popup-control' });
    control.append(el('span', { textContent: label }), input, shown);
    return { control, input, shown };
  };

  // 1. Logits.
  const sLogits = section('logits', 'logits (fp32)', `The LM head scores every token in the vocabulary. Illustrative values; the top ${SHOWN} of ${model.vocabSize.toLocaleString('en-US')} shown, by rank.`);
  const cLogits = barChart();
  sLogits.append(cLogits.root);

  // 2. Temperature.
  const sTemp = section('temperature', '÷ temperature', 'Divides every logit by T. Outlines: the logits before, on the same scale.');
  const temp = slider('Temperature', 0.05, 2, 0.05, 0.6);
  const cTemp = barChart();
  sTemp.append(temp.control, cTemp.root);

  // 3. Top-k / top-p.
  const sFilter = section('filter', 'top-k / top-p',
    'Keeps the k most likely tokens, then the smallest set of them whose probability reaches p. The rest become −∞ (outlines).');
  const topK = slider('Top-k', 0, 100, 1, 50);
  const topP = slider('Top-p', 0.05, 1, 0.01, 0.9);
  const cFilter = barChart();
  const filterFacts = el('div', { className: 'op-popup-facts' });
  sFilter.append(topK.control, topP.control, cFilter.root, filterFacts);

  // 4. Softmax.
  const sSoft = section('softmax', 'softmax', 'Turns the kept logits into probabilities that sum to 1; removed tokens get exactly 0.');
  const cSoft = barChart();
  const softFacts = el('div', { className: 'op-popup-facts' });
  sSoft.append(cSoft.root, softFacts);

  // 5. Sample.
  const sSample = section('sample', 'sample', 'Draws one token id at random, weighted by the probabilities. The pick is marked on the softmax chart above.');
  const sampleRow = el('div', { className: 'op-popup-control' });
  const sampleBtn = el('button', { textContent: 'Sample' });
  const sampleOut = el('span');
  sampleRow.append(sampleBtn, sampleOut);
  sSample.append(sampleRow);

  const rand = rng(Date.now() >>> 0);
  let picked: number | null = null;
  let probs: Float64Array = new Float64Array(0);

  const pct = (x: number) => `${(100 * x).toPrecision(x < 0.1 ? 2 : 3)}%`;
  const int = (n: number) => n.toLocaleString('en-US');

  function update() {
    const t = Number(temp.input.value);
    const k = Number(topK.input.value);
    const p = Number(topP.input.value);
    temp.shown.textContent = `T = ${t.toFixed(2)}`;
    topK.shown.textContent = k > 0 ? `k = ${k}` : 'off';
    topP.shown.textContent = p >= 1 ? 'off' : `p = ${p.toFixed(2)}`;

    const scaled = logits.map((v) => v / t);
    const { afterK, kept } = keptCount(scaled, k, p);
    probs = softmaxPrefix(scaled, kept);
    if (picked !== null && picked >= kept) picked = null;

    const scaledShown = Array.from(scaled.subarray(0, SHOWN));
    cLogits.draw({ values: input, lo: Math.min(0, ...input), hi: Math.max(...input), describe: (i) => `#${i + 1} · logit ${input[i].toFixed(2)}` });
    const sLo = Math.min(0, ...input, ...scaledShown);
    const sHi = Math.max(...input, ...scaledShown);
    cTemp.draw({
      values: scaledShown, ghost: input, lo: sLo, hi: sHi,
      describe: (i) => `#${i + 1} · ${input[i].toFixed(2)} ÷ ${t.toFixed(2)} = ${scaledShown[i].toFixed(2)}`,
    });
    cFilter.draw({
      values: scaledShown, removed: scaledShown.map((_, i) => i >= kept), lo: sLo, hi: sHi,
      describe: (i) => (i < kept ? `#${i + 1} · kept · ${scaledShown[i].toFixed(2)}` : `#${i + 1} · removed → −∞`),
    });
    const shownProbs = scaledShown.map((_, i) => (i < kept ? probs[i] : 0));
    cSoft.draw({
      values: shownProbs, lo: 0, hi: Math.max(...shownProbs), percent: true, picked,
      describe: (i) => `#${i + 1} · p = ${pct(shownProbs[i])}`,
    });

    filterFacts.innerHTML =
      `Top-k keeps <b>${int(afterK)}</b> of ${int(model.vocabSize)} tokens; top-p then keeps <b>${int(kept)}</b>` +
      (kept > SHOWN ? ` (only the first ${SHOWN} are shown).` : '.');
    let cum = 0;
    let for90 = 0;
    while (for90 < kept && cum < 0.9) cum += probs[for90++];
    softFacts.innerHTML =
      `The top token gets <b>${pct(probs[0])}</b>; <b>${int(for90)}</b> ${for90 === 1 ? 'token makes' : 'tokens make'} up 90% of the probability.`;
    sampleOut.innerHTML = picked === null
      ? 'Not sampled yet.'
      : `Sampled token <b>#${picked + 1}</b>, which had probability <b>${pct(probs[picked])}</b>.`;
  }

  sampleBtn.addEventListener('click', () => {
    const u = rand();
    let cum = 0;
    picked = probs.length - 1;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i];
      if (u < cum) {
        picked = i;
        break;
      }
    }
    update();
  });
  for (const s of [temp, topK, topP]) {
    s.input.addEventListener('input', () => {
      picked = null;
      update();
    });
  }
  update();
  sections.get(focus)?.scrollIntoView({ block: 'center' });
}

// ---------------------------------------------------------------------------------------------
// SwiGLU: a = SiLU(g) ⊙ u

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const silu = (x: number) => x * sigmoid(x);
/** SiLU's minimum: x = −1.2785 solves 1 + x(1 − σ(x)) = 0. */
const SILU_MIN_X = -1.2785;
/** Features shown for one illustrative token. */
const FEATURES = 32;

function swigluViz(body: HTMLElement, focus: GluStage): void {
  const sSilu = stageSection(body, focus === 'silu', 'SiLU(x) = x · σ(x)',
    'The activation of the gate. Like ReLU for large |x|, but smooth, and slightly negative for negative inputs. ' +
    'No parameters: elementwise, the same for every feature and token.');
  sSilu.append(siluChart());
  const fSilu = el('div', { className: 'op-popup-facts' });
  fSilu.innerHTML =
    `Its minimum is <b>${silu(SILU_MIN_X).toFixed(3)}</b> at x = <b>${SILU_MIN_X.toFixed(2)}</b>; below that it rises back toward 0. ` +
    `σ(x) is the share of x that passes: <b>${(100 * sigmoid(2)).toFixed(0)}%</b> at x = 2, <b>${(100 * sigmoid(-2)).toFixed(0)}%</b> at x = −2.`;
  sSilu.append(fSilu);

  const sGate = stageSection(body, focus === 'gate', 'a = SiLU(g) ⊙ u',
    `Two projections of the same input: the gate g = x · W_gate and the up projection u = x · W_up. ` +
    `SiLU(g) decides, feature by feature, how much of u passes. Illustrative values for ${FEATURES} of the d_ffn features of one token, all on one scale.`);
  const resample = el('button', { textContent: 'Another token' });
  const row = el('div', { className: 'op-popup-control' });
  row.append(resample);
  const cGate = barChart('SiLU(g)', 'outlines: g');
  const cUp = barChart('u', '');
  const cOut = barChart('a = SiLU(g) ⊙ u', 'outlines: u');
  const facts = el('div', { className: 'op-popup-facts' });
  sGate.append(row, cGate.root, cUp.root, cOut.root, facts);

  let seed = 11;
  function update() {
    const rand = rng(seed);
    const normal = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
    const g = Array.from({ length: FEATURES }, () => 1.6 * normal());
    const u = Array.from({ length: FEATURES }, () => normal());
    const sg = g.map(silu);
    const a = sg.map((v, i) => v * u[i]);
    const all = [...g, ...u, ...a];
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    const ticks = [0, 8, 16, 24, FEATURES - 1];
    const tick = (i: number) => String(i);
    const f = (v: number) => v.toFixed(2);
    cGate.draw({ values: sg, ghost: g, lo, hi, tickLabel: tick, ticks, describe: (i) => `feature ${i} · g = ${f(g[i])} → SiLU(g) = ${f(sg[i])}` });
    cUp.draw({ values: u, lo, hi, tickLabel: tick, ticks, describe: (i) => `feature ${i} · u = ${f(u[i])}` });
    cOut.draw({
      values: a, ghost: u, lo, hi, tickLabel: tick, ticks,
      describe: (i) => `feature ${i} · ${f(sg[i])} × ${f(u[i])} = ${f(a[i])}`,
    });
    const closed = g.filter((v) => v < 0).length;
    facts.innerHTML =
      `<b>${closed}</b> of ${FEATURES} gates are negative: SiLU squeezes them into [${silu(SILU_MIN_X).toFixed(2)}, 0], ` +
      `so those features of u barely pass (and flip sign slightly). Positive gates pass u scaled by about g. ` +
      'Cheap in FLOPs, but g, u and a are each a full T × d_ffn tensor, several times wider than the residual stream.';
  }
  resample.addEventListener('click', () => {
    seed++;
    update();
  });
  update();
  (focus === 'gate' ? sGate : sSilu).scrollIntoView({ block: 'center' });
}

/** SiLU against ReLU over x ∈ [−6, 6], with a hover crosshair and the minimum marked. */
function siluChart(): HTMLElement {
  const minY = silu(SILU_MIN_X);
  return lineChart({
    caption: '<span class="op-legend"><i class="swatch-line"></i>SiLU(x)</span>' +
      '<span class="op-legend"><i class="swatch-ref"></i>ReLU(x) = max(0, x), for reference</span>',
    label: 'SiLU and ReLU from x = −6 to 6',
    x: { lo: -6, hi: 6, ticks: [-6, -4, -2, 0, 2, 4, 6], tickLabel: String },
    y: { lo: -1, hi: 6, ticks: [-1, 0, 1, 2, 3, 4, 5, 6] },
    f: silu,
    ref: (x) => Math.max(0, x),
    marker: { x: SILU_MIN_X, label: `min ${minY.toFixed(3)}` },
    describe: (x) => `x = ${x.toFixed(2)} · σ(x) = ${sigmoid(x).toFixed(3)} · SiLU = ${silu(x).toFixed(3)} · ReLU = ${Math.max(0, x).toFixed(2)}`,
  });
}

interface LineChartOptions {
  /** Legend, as HTML. */
  caption: string;
  label: string;
  /** With `log`, x is spaced logarithmically (lo must be positive). */
  x: { lo: number; hi: number; log?: boolean; ticks: number[]; tickLabel: (x: number) => string };
  y: { lo: number; hi: number; ticks: number[] };
  f: (x: number) => number;
  /** A dashed reference curve. */
  ref?: (x: number) => number;
  /** A labeled point on f. */
  marker?: { x: number; label: string };
  /** Snaps the hovered x, e.g. to whole tokens. */
  snap?: (x: number) => number;
  describe: (x: number) => string;
  /** Points along the curve; more for fast oscillations. */
  samples?: number;
}

/** A single-function line chart in SVG, with an optional dashed reference, a marked point and a hover crosshair. */
function lineChart(o: LineChartOptions): HTMLElement {
  const W = 560;
  const H = 220;
  const pad = { l: 40, r: 12, t: 10, b: 22 };
  const t = (x: number) => (o.x.log ? Math.log(x) : x);
  const [tLo, tHi] = [t(o.x.lo), t(o.x.hi)];
  const px = (x: number) => pad.l + ((t(x) - tLo) / (tHi - tLo)) * (W - pad.l - pad.r);
  const py = (y: number) => pad.t + ((o.y.hi - y) / (o.y.hi - o.y.lo)) * (H - pad.t - pad.b);
  const fromPx = (sx: number) => {
    const u = tLo + ((sx - pad.l) / (W - pad.l - pad.r)) * (tHi - tLo);
    return Math.max(o.x.lo, Math.min(o.x.hi, o.x.log ? Math.exp(u) : u));
  };
  const root = el('figure', { className: 'op-chart' });
  const caption = el('figcaption');
  caption.innerHTML = o.caption;
  const plot = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': o.label });
  const tip = el('div', { className: 'op-chart-tip' });
  tip.hidden = true;
  root.append(caption, plot, tip);

  for (const y of o.y.ticks) {
    plot.append(svg('line', { x1: pad.l, x2: W - pad.r, y1: py(y), y2: py(y), class: y === 0 ? 'axis' : 'grid' }));
    const label = svg('text', { x: pad.l - 6, y: py(y) + 3, class: 'tick', 'text-anchor': 'end' });
    label.textContent = String(y);
    plot.append(label);
  }
  for (const x of o.x.ticks) {
    if (x === 0) plot.append(svg('line', { x1: px(0), x2: px(0), y1: py(o.y.lo), y2: py(o.y.hi), class: 'axis' }));
    const label = svg('text', { x: px(x), y: H - 6, class: 'tick', 'text-anchor': 'middle' });
    label.textContent = o.x.tickLabel(x);
    plot.append(label);
  }
  const samples = o.samples ?? 240;
  const path = (f: (x: number) => number) => {
    let d = '';
    for (let i = 0; i <= samples; i++) {
      const x = fromPx(pad.l + ((W - pad.l - pad.r) * i) / samples);
      d += `${i ? 'L' : 'M'}${px(x).toFixed(1)},${py(f(x)).toFixed(1)}`;
    }
    return d;
  };
  if (o.ref) plot.append(svg('path', { d: path(o.ref), class: 'ref' }));
  plot.append(svg('path', { d: path(o.f), class: 'line' }));
  if (o.marker) {
    const my = o.f(o.marker.x);
    plot.append(svg('circle', { cx: px(o.marker.x), cy: py(my), r: 4, class: 'marker' }));
    const label = svg('text', { x: px(o.marker.x), y: py(my) + 16, class: 'tick', 'text-anchor': 'middle' });
    label.textContent = o.marker.label;
    plot.append(label);
  }

  // Crosshair: follows the pointer along x.
  const cross = svg('line', { y1: py(o.y.hi), y2: py(o.y.lo), class: 'crosshair', visibility: 'hidden' });
  const dot = svg('circle', { r: 4, class: 'marker', visibility: 'hidden' });
  const hit = svg('rect', { x: pad.l, y: pad.t, width: W - pad.l - pad.r, height: H - pad.t - pad.b, class: 'hit-area' });
  plot.append(cross, dot, hit);
  hit.addEventListener('pointermove', (e) => {
    const box = plot.getBoundingClientRect();
    const raw = fromPx(((e.clientX - box.left) / box.width) * W);
    const x = o.snap ? o.snap(raw) : raw;
    for (const node of [cross, dot]) node.setAttribute('visibility', 'visible');
    cross.setAttribute('x1', String(px(x)));
    cross.setAttribute('x2', String(px(x)));
    dot.setAttribute('cx', String(px(x)));
    dot.setAttribute('cy', String(py(o.f(x))));
    tip.textContent = o.describe(x);
    tip.hidden = false;
    tip.style.left = `${(px(x) / W) * 100}%`;
  });
  hit.addEventListener('pointerleave', () => {
    for (const node of [cross, dot]) node.setAttribute('visibility', 'hidden');
    tip.hidden = true;
  });
  return root;
}

// ---------------------------------------------------------------------------------------------
// RoPE: rotate each dimension pair of q and k by position · θ_i

/** Dimension pairs drawn as dials: from the fastest to the slowest. */
const DIAL_PAIRS = [0, 4, 8, 16, 24, 32, 48, 63];

/** Tokens, compactly: 6.3, 1,234, 12K, 3.1M. */
function tokens(n: number): string {
  if (n >= 1e6) return `${+(n / 1e6).toPrecision(2)}M`;
  if (n >= 1e4) return `${+(n / 1e3).toPrecision(3)}K`;
  return n >= 100 || Number.isInteger(n) ? Math.round(n).toLocaleString('en-US') : n.toPrecision(2);
}

function ropeViz(body: HTMLElement, focus: RopeStage): void {
  const pairs = model.headDim / 2;
  const theta = Array.from({ length: pairs }, (_, i) => model.ropeTheta ** ((-2 * i) / model.headDim));
  const wavelength = theta.map((t) => (2 * Math.PI) / t);
  const ctx = model.maxContext;

  // 1. Frequencies.
  const sFreq = stageSection(body, focus === 'freq', 'θ_i: one frequency per dimension pair',
    `Each head's ${model.headDim} dimensions form ${pairs} pairs; pair i rotates by θ_i = ` +
    `${model.ropeTheta.toLocaleString('en-US')}^(−2i/${model.headDim}) radians per token. ` +
    `The cos and sin tables hold pos · θ_i for every position: the ${pairs} angles, written twice to fill head_dim.`);
  const cFreq = barChart('wavelength 2π/θ_i in tokens (log scale), by pair', 'line: context length');
  const fFreq = el('div', { className: 'op-popup-facts' });
  sFreq.append(cFreq.root, fFreq);
  const logLambda = wavelength.map(Math.log10);
  cFreq.draw({
    values: logLambda, lo: 0, hi: Math.max(...logLambda, Math.log10(ctx)) * 1.05,
    ticks: [0, 16, 32, 48, pairs - 1], tickLabel: String, yTick: (v) => tokens(10 ** v),
    refs: [{ value: Math.log10(ctx), label: `context ${ctx.toLocaleString('en-US')}`, start: true }],
    describe: (i) => `pair ${i} · θ = ${theta[i].toPrecision(3)} rad/token · one turn every ${tokens(wavelength[i])} tokens`,
  });
  const slow = wavelength.filter((l) => l > ctx).length;
  fFreq.innerHTML =
    `Wavelengths run from <b>${tokens(wavelength[0])}</b> to <b>${tokens(wavelength[pairs - 1])}</b> tokens. ` +
    `<b>${slow}</b> of ${pairs} pairs turn less than once within the ${ctx.toLocaleString('en-US')}-token context: ` +
    'fast pairs tell neighbors apart, slow pairs carry coarse, long-range position.';

  // 2. Rotation by position.
  const sRot = stageSection(body, focus === 'rotate', 'RoPE: rotate each pair by pos · θ_i',
    'q and k are split into pairs (HF rotate_half pairs dimension i with i + 64, not neighbors), and each pair is ' +
    'rotated like a 2D vector. No parameters, nothing added: the vector keeps its length, only its angles change.');
  const maxLog = Math.round(Math.log2(ctx) * 100);
  const pos = el('input', { type: 'range', min: '0', max: String(maxLog), step: '1', value: '700' });
  const posShown = el('b');
  const posRow = el('label', { className: 'op-popup-control' });
  posRow.append(el('span', { textContent: 'Position' }), pos, posShown);
  const dials = svg('svg', { viewBox: '0 0 560 250', role: 'img', 'aria-label': 'Rotation of selected dimension pairs' });
  const dialFig = el('figure', { className: 'op-chart' });
  dialFig.append(dials);
  const fRot = el('div', { className: 'op-popup-facts' });
  sRot.append(posRow, dialFig, fRot);

  function drawDials() {
    const p = Math.round(2 ** (Number(pos.value) / 100)) - 1;
    posShown.textContent = p.toLocaleString('en-US');
    dials.replaceChildren();
    DIAL_PAIRS.filter((i) => i < pairs).forEach((i, k) => {
      const cx = 70 + (k % 4) * 140;
      const cy = 60 + Math.floor(k / 4) * 125;
      const r = 36;
      const angle = p * theta[i];
      dials.append(svg('circle', { cx, cy, r, class: 'dial' }));
      dials.append(svg('line', { x1: cx, y1: cy, x2: cx + r, y2: cy, class: 'ref' }));
      // Counterclockwise, as angles are drawn in math.
      dials.append(svg('line', { x1: cx, y1: cy, x2: cx + r * Math.cos(angle), y2: cy - r * Math.sin(angle), class: 'hand' }));
      dials.append(svg('circle', { cx: cx + r * Math.cos(angle), cy: cy - r * Math.sin(angle), r: 4, class: 'marker' }));
      const turns = angle / (2 * Math.PI);
      const label = svg('text', { x: cx, y: cy + r + 16, class: 'tick', 'text-anchor': 'middle' });
      label.textContent = `pair ${i} · λ ${tokens(wavelength[i])}`;
      const sub = svg('text', { x: cx, y: cy + r + 28, class: 'tick', 'text-anchor': 'middle' });
      sub.textContent = turns >= 1 ? `${turns >= 100 ? Math.round(turns).toLocaleString('en-US') : turns.toFixed(1)} turns` : `${((angle * 180) / Math.PI).toFixed(1)}°`;
      dials.append(label, sub);
    });
    fRot.innerHTML =
      `At position <b>${p.toLocaleString('en-US')}</b>, pair 0 has turned <b>${tokens(p * theta[0] / (2 * Math.PI))}</b> times ` +
      `and pair ${pairs - 1} by <b>${((p * theta[pairs - 1] * 180) / Math.PI).toFixed(2)}°</b>. ` +
      'K is cached after rotation, so each cached row already carries its position.';
  }
  pos.addEventListener('input', drawDials);
  drawDials();

  // 3. Relative position.
  const sRel = stageSection(body, focus === 'rotate', 'Q · K sees only the distance',
    'Rotating q by m · θ_i and k by n · θ_i turns each pair\'s dot product into one rotated by (m − n) · θ_i: ' +
    'the score depends on the distance m − n, not on where the two tokens are.');
  const score = (d: number) => theta.reduce((a, t) => a + Math.cos(d * t), 0) / pairs;
  sRel.append(lineChart({
    caption: '<span class="op-legend"><i class="swatch-line"></i>q · k for the same vector at distance m − n (1 = no rotation)</span>',
    label: 'Score of identical q and k against their distance',
    x: { lo: 1, hi: ctx, log: true, ticks: [1, 10, 100, 1000, 10000].filter((x) => x <= ctx), tickLabel: tokens },
    y: { lo: -0.2, hi: 1, ticks: [0, 0.5, 1] },
    f: score,
    snap: Math.round,
    samples: 1600,
    describe: (d) => `distance ${d.toLocaleString('en-US')} · score ${score(d).toFixed(3)}`,
  }));
  const fRel = el('div', { className: 'op-popup-facts' });
  fRel.innerHTML =
    `For q = k with equal weight in every pair, the score falls from <b>1</b> at distance 0 to about ` +
    `<b>${score(100).toFixed(2)}</b> at 100 and <b>${score(Math.min(4000, ctx)).toFixed(2)}</b> at ${tokens(Math.min(4000, ctx))}, ` +
    'with ripples from the fast pairs: nearby tokens line up more easily. Real heads learn which pairs to use.';
  sRel.append(fRel);

  (focus === 'freq' ? sFreq : sRot).scrollIntoView({ block: 'center' });
}

// ---------------------------------------------------------------------------------------------
// RMSNorm: x / √(mean(x²) + ε) · weight

/** Features of the illustrative token vector, standing in for d_model. */
const NORM_FEATURES = 64;
const NORM_EPS = 1e-5;
/** Residual streams have a few outlier features far larger than the rest ("massive activations"). */
const OUTLIERS: [number, number][] = [[11, 14], [40, -9]];

function rmsNormViz(body: HTMLElement, focus: NormStage): void {
  const rand = rng(5);
  const normal = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const base = Array.from({ length: NORM_FEATURES }, () => 0.8 * normal());
  for (const [i, v] of OUTLIERS) base[i] = v;
  // Learned scales: illustrative, mostly 0.2–0.6, smaller on the outlier features.
  const gamma = base.map(() => Math.abs(0.4 + 0.12 * normal()));
  for (const [i] of OUTLIERS) gamma[i] = 0.05;

  const sInput = stageSection(body, focus === 'input', 'x: one token',
    `The residual stream of one token, upcast to fp32. Illustrative: ${NORM_FEATURES} features standing in for d_model, ` +
    'including two outliers, as real residual streams have.');
  const scale = el('input', { type: 'range', min: '-2', max: '2', step: '0.01', value: '0' });
  const scaleShown = el('b');
  const scaleRow = el('label', { className: 'op-popup-control' });
  scaleRow.append(el('span', { textContent: 'Scale x by' }), scale, scaleShown);
  const cX = barChart('x');
  const fX = el('div', { className: 'op-popup-facts' });
  sInput.append(scaleRow, cX.root, fX);

  const sMean = stageSection(body, focus === 'mean', 'mean(x²), then 1/√(mean(x²) + ε)',
    'Squares, then the mean over the features: one number per token. Its reciprocal square root is the scale.');
  const cSq = barChart('x²', 'line: mean(x²)');
  const fMean = el('div', { className: 'op-popup-facts' });
  sMean.append(cSq.root, fMean);

  const sNorm = stageSection(body, focus === 'normalize', 'x / RMS(x)',
    'Every feature times the same 1/RMS, then cast back to the activation dtype. The vector keeps its direction; its RMS becomes 1.');
  const cNorm = barChart('x / RMS(x)', 'outlines: x · lines: ±1');
  const fNorm = el('div', { className: 'op-popup-facts' });
  sNorm.append(cNorm.root, fNorm);

  const sWeight = stageSection(body, focus === 'weight', '× weight',
    `A learned scale per feature: the norm's only parameters (d_model of them). Illustrative values.`);
  const cGamma = barChart('weight', '', 'weight');
  const cOut = barChart('x / RMS(x) · weight', 'outlines: x / RMS(x)');
  sWeight.append(cGamma.root, cOut.root);

  const ticks = [0, 16, 32, 48, NORM_FEATURES - 1];
  const tickLabel = (i: number) => String(i);
  const f = (v: number) =>
    Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(1) : v.toFixed(3);
  const extent = (vs: number[]) => ({ lo: Math.min(0, ...vs), hi: Math.max(0, ...vs) });

  function update() {
    const s = 10 ** Number(scale.value);
    scaleShown.textContent = `×${s < 1 ? s.toPrecision(2) : s.toFixed(s < 10 ? 2 : 1)}`;
    const x = base.map((v) => v * s);
    const sq = x.map((v) => v * v);
    const ms = sq.reduce((a, b) => a + b, 0) / NORM_FEATURES;
    const r = 1 / Math.sqrt(ms + NORM_EPS);
    const n = x.map((v) => v * r);
    const out = n.map((v, i) => v * gamma[i]);
    const rmsOf = (vs: number[]) => Math.sqrt(vs.reduce((a, v) => a + v * v, 0) / vs.length);
    const outlierShare = OUTLIERS.reduce((a, [i]) => a + sq[i], 0) / (ms * NORM_FEATURES);

    cX.draw({ values: x, ...extent(x), ticks, tickLabel, describe: (i) => `feature ${i} · x = ${f(x[i])}` });
    fX.innerHTML = `RMS(x) = <b>${f(Math.sqrt(ms))}</b>; largest |x| = <b>${f(Math.max(...x.map(Math.abs)))}</b>.`;

    cSq.draw({
      values: sq, ...extent(sq), ticks, tickLabel, refs: [{ value: ms, label: 'mean' }],
      describe: (i) => `feature ${i} · x² = ${f(sq[i])}`,
    });
    fMean.innerHTML =
      `mean(x²) = <b>${f(ms)}</b>, ε = ${NORM_EPS}, so 1/√(mean + ε) = <b>${f(r)}</b>. ` +
      `The two outliers contribute <b>${(100 * outlierShare).toFixed(0)}%</b> of the mean: they set the scale for every other feature.`;

    const nExt = extent([...n, ...x, 1, -1]);
    cNorm.draw({
      values: n, ghost: x, ...nExt, ticks, tickLabel, refs: [{ value: 1, label: '+1' }, { value: -1, label: '−1' }],
      describe: (i) => `feature ${i} · ${f(x[i])} × ${f(r)} = ${f(n[i])}`,
    });
    fNorm.innerHTML =
      `RMS after: <b>${rmsOf(n).toFixed(3)}</b>. Move the scale slider: the outlines follow x, the bars stay the same, ` +
      `because x and RMS(x) scale together. Only when mean(x²) gets near ε (scale ×0.01) does the RMS drop below 1.`;

    cGamma.draw({ values: gamma, ...extent(gamma), ticks, tickLabel, describe: (i) => `feature ${i} · weight = ${f(gamma[i])}` });
    const oExt = extent([...out, ...n]);
    cOut.draw({
      values: out, ghost: n, ...oExt, ticks, tickLabel,
      describe: (i) => `feature ${i} · ${f(n[i])} × ${f(gamma[i])} = ${f(out[i])}`,
    });
  }
  scale.addEventListener('input', update);
  update();
  ({ input: sInput, mean: sMean, normalize: sNorm, weight: sWeight })[focus].scrollIntoView({ block: 'center' });
}

interface Draw {
  values: number[];
  /** Horizontal reference lines, labeled at the right (or the left, with `start`). */
  refs?: { value: number; label: string; start?: boolean }[];
  /** Label of the column ticks (default: rank, #1). */
  tickLabel?: (i: number) => string;
  /** Label of the value ticks (default: the value, compact). */
  yTick?: (v: number) => string;
  /** Columns that get a tick label (default: ranks 1, 10, 20, …). */
  ticks?: number[];
  lo: number;
  hi: number;
  describe: (i: number) => string;
  /** Outline columns drawn over the bars, e.g. the values before an op. */
  ghost?: number[];
  /** Columns drawn as outlines only: tokens a filter removed. */
  removed?: boolean[];
  /** Index of a column to mark, e.g. the sampled token. */
  picked?: number | null;
  /** Tick labels as percentages. */
  percent?: boolean;
}

/**
 * A single-series column chart in SVG, with optional outlines, removed columns, a marked column, reference lines and
 * hover tooltips. `tone`: weight draws the columns in the weight color.
 */
function barChart(title?: string, subtitle = '', tone: 'activation' | 'weight' = 'activation') {
  const W = 560;
  const H = 150;
  const pad = { l: 40, r: 8, t: 10, b: 18 };
  const root = el('figure', { className: 'op-chart' });
  const plot = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
  const tip = el('div', { className: 'op-chart-tip' });
  tip.hidden = true;
  if (title) {
    const caption = el('figcaption', { textContent: title });
    if (subtitle) caption.append(el('span', { textContent: subtitle }));
    root.append(caption);
  }
  root.append(plot, tip);

  function draw({ values, lo, hi, describe, ghost, removed, picked, percent, refs = [], yTick, tickLabel = (i) => `#${i + 1}`, ticks = [0, 9, 19, 29, 39] }: Draw) {
    plot.replaceChildren();
    const x0 = pad.l;
    const band = (W - pad.l - pad.r) / values.length;
    const barW = Math.min(24, band - 2);
    const span = hi - lo || 1;
    const y = (v: number) => pad.t + ((hi - v) / span) * (H - pad.t - pad.b);

    // Recessive grid with clean ticks.
    const step = niceStep(span / 4);
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
      plot.append(svg('line', { x1: x0, x2: W - pad.r, y1: y(v), y2: y(v), class: Math.abs(v) < 1e-9 ? 'axis' : 'grid' }));
      const label = svg('text', { x: x0 - 6, y: y(v) + 3, class: 'tick', 'text-anchor': 'end' });
      label.textContent = percent ? `${Math.round(v * 1000) / 10}%` : yTick ? yTick(v) : tickText(v, step);
      plot.append(label);
    }

    values.forEach((v, i) => {
      const cx = x0 + band * (i + 0.5);
      const cls = removed?.[i] ? 'ghost' : i === picked ? 'bar picked' : tone === 'weight' ? 'bar weight' : 'bar';
      plot.append(column(cx, barW, y(0), y(v), cls));
      // Outline on top, so it shows whether the bar is taller or shorter.
      if (ghost) plot.append(column(cx, barW, y(0), y(ghost[i]), 'ghost'));
      if (i === picked) {
        const mark = svg('text', { x: cx, y: Math.max(pad.t + 8, y(v) - 4), class: 'pick-mark', 'text-anchor': 'middle' });
        mark.textContent = '▼';
        plot.append(mark);
      }
      // Hit target: the whole band, taller than the bar.
      const hit = svg('rect', { x: cx - band / 2, y: pad.t, width: band, height: H - pad.t - pad.b, class: 'hit' });
      hit.addEventListener('pointerenter', () => {
        tip.textContent = describe(i);
        tip.hidden = false;
        tip.style.left = `${(cx / W) * 100}%`;
      });
      hit.addEventListener('pointerleave', () => (tip.hidden = true));
      plot.append(hit);
    });

    for (const ref of refs) {
      plot.append(svg('line', { x1: x0, x2: W - pad.r, y1: y(ref.value), y2: y(ref.value), class: 'ref' }));
      const label = ref.start
        ? svg('text', { x: x0 + 4, y: y(ref.value) - 4, class: 'tick', 'text-anchor': 'start' })
        : svg('text', { x: W - pad.r - 2, y: y(ref.value) - 4, class: 'tick', 'text-anchor': 'end' });
      label.textContent = ref.label;
      plot.append(label);
    }
    for (const i of ticks.filter((i) => i < values.length)) {
      const label = svg('text', { x: x0 + band * (i + 0.5), y: H - 4, class: 'tick', 'text-anchor': 'middle' });
      label.textContent = tickLabel(i);
      plot.append(label);
    }
  }

  return { root, draw };
}

/** A column from the baseline to `top`, with a 4px rounded data-end and a square base. */
function column(cx: number, w: number, base: number, top: number, cls: string) {
  const r = Math.min(4, w / 2, Math.abs(base - top));
  const x = cx - w / 2;
  const up = top < base;
  const d = up
    ? `M${x},${base} V${top + r} Q${x},${top} ${x + r},${top} H${x + w - r} Q${x + w},${top} ${x + w},${top + r} V${base} Z`
    : `M${x},${base} V${top - r} Q${x},${top} ${x + r},${top} H${x + w - r} Q${x + w},${top} ${x + w},${top - r} V${base} Z`;
  return svg('path', { d, class: cls });
}

function niceStep(raw: number): number {
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}

/** An axis tick: as many decimals as the step needs, and K / M for large values so labels fit the margin. */
function tickText(v: number, step: number): string {
  const a = Math.abs(v);
  if (a >= 1e6) return `${+(v / 1e6).toPrecision(3)}M`;
  if (a >= 1e4) return `${+(v / 1e3).toPrecision(3)}K`;
  return v.toFixed(Math.max(0, -Math.floor(Math.log10(step))));
}
