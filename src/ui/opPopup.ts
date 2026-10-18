import { formatBytes } from '../format';
import { LLAMA_3_8B, type ModelConfig } from '../model/config';
import { DTYPES, type DType } from '../model/dtypes';
import { routing } from '../model/routing';
import type { Mode, OpSpec } from '../model/tensors';

/** The current model: popups take its vocabulary, RoPE base, head and context sizes. */
let model: ModelConfig = LLAMA_3_8B;
/** The pass being drawn: popups quote its real sizes next to their illustrations. */
let pass: { mode: Mode; T: number; weightDtype: DType } = { mode: 'prefill', T: 2048, weightDtype: 'bf16' };

export function setPopupContext(cfg: ModelConfig, current: { mode: Mode; T: number; weightDtype: DType }): void {
  model = cfg;
  pass = { mode: current.mode, T: current.T, weightDtype: current.weightDtype };
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

/** The stages of the mixture-of-experts panel. */
type MoeStage = 'router' | 'dispatch' | 'combine';

/** MoE ops, by output: the router's tensors, each expert's gathered rows, and the weighted sum back into m. */
function moeStage(op: OpSpec): MoeStage | null {
  const out = op.output.split(':')[1];
  if (['router_logits', 'router_probs', 'selected_experts', 'routing_weights'].includes(out)) return 'router';
  if (/^e\d+\.x$/.test(out)) return 'dispatch';
  if (/^e\d+\.ys$/.test(out) || (out === 'm' && op.label.includes('index_add'))) return 'combine';
  return null;
}

/** The stages of the attention panel. */
type AttnStage = 'scores' | 'softmax' | 'output';

/**
 * Attention ops, by output: eager repeat_kv copies, the causal mask, raw and masked scores, softmax and its cast
 * (`probs` only on layer floors: the LM head's sampling probabilities share the id), and attn (fused, or eager · V).
 */
function attnStage(op: OpSpec): AttnStage | null {
  const out = op.output.split(':')[1];
  if (['k_rep', 'v_rep', 'causal_mask', 'scores', 'scores_masked'].includes(out)) return 'scores';
  if (out === 'probs_fp32' || (out === 'probs' && op.floor >= 1 && op.floor <= model.nLayers)) return 'softmax';
  if (out === 'attn') return op.label === 'attention' ? 'scores' : 'output';
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
  if (op.label === 'lookup') return { title: 'Embedding lookup: token ids select rows', render: lookupViz };
  const sampling = samplingStage(op);
  if (sampling) return { title: 'Sampling: from logits to the next token', render: (body) => samplingViz(body, sampling) };
  const moe = moeStage(op);
  if (moe) return { title: 'Mixture of experts: route, dispatch, combine', render: (body) => moeViz(body, moe, op.floor - 1) };
  const attn = attnStage(op);
  if (attn) return { title: 'Attention: softmax(QKᵀ/√d) · V, one head', render: (body) => attentionViz(body, attn) };
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
    placeTip(tip, px(x) / W);
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
// Mixture of experts: router softmax → top-k → renormalize; dispatch; weighted combine

/** Features shown for an expert output. */
const MOE_FEATURES = 24;

function moeViz(body: HTMLElement, focus: MoeStage, layer: number): void {
  const { nExperts: E, topK: K } = model.moe!;
  const expertIds = Array.from({ length: E }, (_, e) => e);
  const expertTick = (e: number) => `E${e}`;
  const pct = (p: number) => `${(100 * p).toFixed(1)}%`;

  // 1. Router.
  const sRouter = stageSection(body, focus === 'router', `Router: softmax, top-${K}, renormalize`,
    `A ${model.dModel.toLocaleString('en-US')} × ${E} weight scores the ${E} experts for each token; softmax (in fp32) turns the ` +
    `scores into probabilities, the top ${K} are kept, and their probabilities are rescaled to sum to 1. Illustrative scores for one token.`);
  const another = el('button', { textContent: 'Another token' });
  const row = el('div', { className: 'op-popup-control' });
  row.append(another);
  const cLogits = barChart('router logits');
  const cProbs = barChart('softmax', `outlines: experts outside the top ${K}`);
  const cWeights = barChart('routing weights', `the top ${K}, rescaled to sum to 1`);
  const fRouter = el('div', { className: 'op-popup-facts' });
  sRouter.append(row, cLogits.root, cProbs.root, cWeights.root, fRouter);

  // 2. Dispatch, from the same seeded routing the scene draws.
  const decode = pass.mode === 'decode';
  const n = decode ? 1 : pass.T;
  const past = decode ? pass.T : 0;
  const counts = routing(model).counts(layer, past, past + n);
  const sDispatch = stageSection(body, focus === 'dispatch', `Dispatch: layer ${layer}, this pass`,
    `Each expert gathers the rows of the tokens that picked it and runs its MLP on those only. ` +
    `These are the counts the scene draws (seeded, illustrative routing).`);
  const cCounts = barChart('tokens per expert', decode ? '' : `line: the even share, ${K}T / ${E}`);
  const fDispatch = el('div', { className: 'op-popup-facts' });
  sDispatch.append(cCounts.root, fDispatch);
  const even = (K * n) / E;
  cCounts.draw({
    values: counts, lo: 0, hi: Math.max(...counts, even) * 1.1, ticks: expertIds, tickLabel: expertTick,
    refs: decode ? [] : [{ value: even, label: 'even' }], removed: counts.map((c) => c === 0),
    describe: (e) => (counts[e] ? `E${e} · ${counts[e].toLocaleString('en-US')} tokens` : `E${e} · idle: its weights are not read`),
  });
  const used = expertIds.filter((e) => counts[e] > 0);
  fDispatch.innerHTML = decode
    ? `The new token goes to experts <b>${used.join('</b> and <b>')}</b>; the other ${E - used.length} stay idle, so this ` +
      `layer reads only <b>${used.length}</b> of ${E} experts' weights. Generate to watch the pair change token by token.`
    : `${K} × ${n.toLocaleString('en-US')} = <b>${(K * n).toLocaleString('en-US')}</b> token slots; the busiest expert gets ` +
      `<b>${Math.max(...counts).toLocaleString('en-US')}</b> (${(Math.max(...counts) / even).toFixed(2)}× its even share) and the quietest ` +
      `<b>${Math.min(...counts).toLocaleString('en-US')}</b>. HF's loop drops no tokens; training adds a load-balancing loss to keep this even.`;

  // 3. Combine.
  const sCombine = stageSection(body, focus === 'combine', 'Combine: m = Σ weight × expert output',
    `Each expert's output rows are scaled by their routing weights and added back at their tokens' positions (index_add_). ` +
    `Every row of m receives exactly ${K} contributions. Illustrative outputs for the token above.`);
  const cParts = Array.from({ length: K }, () => barChart('', ''));
  const cSum = barChart('m: the sum', 'outlines: the first expert alone');
  sCombine.append(...cParts.map((c) => c.root), cSum.root);

  let seed = 21;
  function update() {
    const rand = rng(seed);
    const normal = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
    const logits = expertIds.map(() => 1.2 * normal());
    const m = Math.max(...logits);
    const exp = logits.map((v) => Math.exp(v - m));
    const sum = exp.reduce((a, b) => a + b, 0);
    const probs = exp.map((v) => v / sum);
    const top = [...expertIds].sort((a, b) => probs[b] - probs[a]).slice(0, K);
    const kept = top.reduce((a, e) => a + probs[e], 0);
    const weights = expertIds.map((e) => (top.includes(e) ? probs[e] / kept : 0));

    const lHi = Math.max(0, ...logits);
    const lLo = Math.min(0, ...logits);
    cLogits.draw({ values: logits, lo: lLo, hi: lHi, ticks: expertIds, tickLabel: expertTick, describe: (e) => `E${e} · logit ${logits[e].toFixed(2)}` });
    cProbs.draw({
      values: probs, lo: 0, hi: 1, percent: true, ticks: expertIds, tickLabel: expertTick,
      removed: expertIds.map((e) => !top.includes(e)),
      describe: (e) => `E${e} · p = ${pct(probs[e])}${top.includes(e) ? ' · kept' : ''}`,
    });
    cWeights.draw({
      values: weights, lo: 0, hi: 1, percent: true, ticks: expertIds, tickLabel: expertTick, ghost: probs,
      describe: (e) => (weights[e] ? `E${e} · ${pct(probs[e])} ÷ ${pct(kept)} = ${pct(weights[e])}` : `E${e} · not used`),
    });
    fRouter.innerHTML =
      `This token goes to <b>E${top.join('</b> and <b>E')}</b>. Together they had <b>${pct(kept)}</b> of the probability; ` +
      `after rescaling their weights are <b>${top.map((e) => pct(weights[e])).join('</b> and <b>')}</b>. ` +
      `The router is ${(model.dModel * E).toLocaleString('en-US')} parameters per layer, against ` +
      `${(3 * model.dModel * model.dFfn / 1e6).toFixed(0)}M in each expert.`;

    // Illustrative expert outputs: different experts, different vectors.
    const outputs = top.map(() => Array.from({ length: MOE_FEATURES }, () => normal()));
    const parts = outputs.map((y, k) => y.map((v) => v * weights[top[k]]));
    const total = parts[0].map((_, i) => parts.reduce((a, p) => a + p[i], 0));
    const all = [...outputs.flat(), ...parts.flat(), ...total];
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    const ticks = [0, 8, 16, MOE_FEATURES - 1];
    parts.forEach((p, k) => {
      const e = top[k];
      cParts[k].root.querySelector('figcaption')?.remove();
      const caption = el('figcaption', { textContent: `${pct(weights[e])} × E${e}'s output` });
      caption.append(el('span', { textContent: `outlines: E${e}'s output before scaling` }));
      cParts[k].root.prepend(caption);
      cParts[k].draw({
        values: p, ghost: outputs[k], lo, hi, ticks, tickLabel: String,
        describe: (i) => `feature ${i} · ${weights[e].toFixed(3)} × ${outputs[k][i].toFixed(2)} = ${p[i].toFixed(2)}`,
      });
    });
    cSum.draw({
      values: total, ghost: parts[0], lo, hi, ticks, tickLabel: String,
      describe: (i) => `feature ${i} · ${parts.map((p) => p[i].toFixed(2)).join(' + ')} = ${total[i].toFixed(2)}`,
    });
  }
  another.addEventListener('click', () => {
    seed++;
    update();
  });
  update();
  ({ router: sRouter, dispatch: sDispatch, combine: sCombine })[focus].scrollIntoView({ block: 'center' });
}

// ---------------------------------------------------------------------------------------------
// Embedding lookup: each token id selects one row of the table

const LOOKUP_WORDS = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'the', 'mat', 'was', 'warm'];
/** Features of the selected row shown as bars: the first ROW_HEAD and the last ROW_TAIL, with a gap between. */
const ROW_HEAD = 36;
const ROW_TAIL = 12;

/**
 * The prompt as (token, id) pairs. The beginning-of-text ids are the real ones (Llama 3: 128000, Mixtral: 1);
 * the others are illustrative, derived from the word so that repeated words share an id, as they do in a tokenizer.
 */
function lookupTokens(): { text: string; id: number }[] {
  const bos = model.vocabSize > 100000 ? { text: '<|begin_of_text|>', id: 128000 } : { text: '<s>', id: 1 };
  const hash = (w: string) => [...w].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7);
  return [bos, ...LOOKUP_WORDS.map((w) => ({ text: w, id: 1000 + (hash(w) % (Math.min(model.vocabSize, 100000) - 2000)) }))];
}

function lookupViz(body: HTMLElement): void {
  const toks = lookupTokens();
  const n = toks.length;
  const V = model.vocabSize;
  let selected = 2; // "cat"

  // 1. Token ids → rows → x.
  const sRows = stageSection(body, true, 'Token ids select rows of embed_tokens',
    `The table has one row of ${model.dModel.toLocaleString('en-US')} numbers per vocabulary entry (${V.toLocaleString('en-US')} rows). ` +
    "Each token id picks its row, which is copied into x: no multiplication, just indexing. Hover a token to follow it.");
  const W = 560;
  const rowH = 22;
  const top = 26;
  const H = top + n * rowH + 10;
  const table = { x: 250, w: 44, y: top, h: n * rowH - 4 };
  const diagram = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Token ids indexing rows of the embedding table' });
  const fig = el('figure', { className: 'op-chart' });
  fig.append(diagram);
  const fRows = el('div', { className: 'op-popup-facts' });
  sRows.append(fig, fRows);

  // 2. The selected row.
  const sRow = stageSection(body, false, 'One row: a token\'s embedding',
    `Illustrative values from both ends of the ${model.dModel.toLocaleString('en-US')} features. Learned in training; ` +
    'the same token always gets the same row, whatever its position (RoPE adds position later, inside attention).');
  const cRow = barChart('', '', 'weight');
  sRow.append(cRow.root);

  // The real cost in this pass.
  const fCost = el('div', { className: 'op-popup-facts' });
  body.append(fCost);
  const wb = DTYPES[pass.weightDtype].bytes;
  const rowBytes = model.dModel * wb;
  const tableBytes = V * rowBytes;
  const rows = pass.mode === 'decode' ? 1 : pass.T;
  fCost.innerHTML =
    `In this pass: <b>${rows.toLocaleString('en-US')}</b> ${rows === 1 ? 'row' : 'rows'} × ${formatBytes(rowBytes)} = ` +
    `<b>${formatBytes(rows * rowBytes)}</b> read from a <b>${formatBytes(tableBytes)}</b> table ` +
    `(${((100 * rows * rowBytes) / tableBytes).toPrecision(2)}%), and <b>0</b> FLOPs. ` +
    'The LM head at the top is the opposite: a full matrix multiply against a second table of the same shape ' +
    '(not tied to this one in Llama 3 or Mixtral), scoring every vocabulary entry.';

  const rowY = (id: number) => table.y + (id / V) * table.h;
  const chipY = (i: number) => top + i * rowH + rowH / 2;

  function draw() {
    diagram.replaceChildren();
    const heading = (x: number, text: string, anchor = 'start') => {
      const t = svg('text', { x, y: 12, class: 'tick', 'text-anchor': anchor });
      t.textContent = text;
      diagram.append(t);
    };
    heading(4, 'tokens and ids');
    heading(table.x + table.w / 2, `embed_tokens: ${V.toLocaleString('en-US')} rows`, 'middle');
    heading(W - 4, 'x: one row per token', 'end');
    diagram.append(svg('rect', { x: table.x, y: table.y, width: table.w, height: table.h, rx: 3, class: 'table' }));
    const zero = svg('text', { x: table.x + table.w + 4, y: table.y + 8, class: 'tick' });
    zero.textContent = '0';
    const last = svg('text', { x: table.x + table.w + 4, y: table.y + table.h, class: 'tick' });
    last.textContent = (V - 1).toLocaleString('en-US');
    diagram.append(zero, last);

    // Unselected first, so the selected path is drawn on top.
    const order = [...toks.keys()].sort((a, b) => Number(a === selected) - Number(b === selected));
    for (const i of order) {
      const { text, id } = toks[i];
      const on = toks[i].id === toks[selected].id;
      const cls = on ? 'link on' : 'link';
      const y = chipY(i);
      const ry = rowY(id);
      diagram.append(svg('path', { d: `M150,${y} C200,${y} 200,${ry} ${table.x},${ry}`, class: cls }));
      diagram.append(svg('line', { x1: table.x, x2: table.x + table.w, y1: ry, y2: ry, class: on ? 'row on' : 'row' }));
      diagram.append(svg('path', { d: `M${table.x + table.w},${ry} C${table.x + table.w + 50},${ry} 380,${y} 420,${y}`, class: cls }));
      const label = svg('text', { x: 4, y: y + 4, class: on ? 'tick selected' : 'tick' });
      label.textContent = text.length > 17 ? `${text.slice(0, 16)}…` : text;
      const idText = svg('text', { x: 146, y: y + 4, class: on ? 'tick selected' : 'tick', 'text-anchor': 'end' });
      idText.textContent = id.toLocaleString('en-US');
      diagram.append(label, idText);
      diagram.append(svg('rect', { x: 420, y: y - 6, width: W - 424, height: 12, rx: 2, class: on ? 'xrow on' : 'xrow' }));
      const hit = svg('rect', { x: 0, y: y - rowH / 2, width: W, height: rowH, class: 'hit' });
      hit.addEventListener('pointerenter', () => {
        if (selected === i) return;
        selected = i;
        update();
      });
      diagram.append(hit);
    }
  }

  function update() {
    draw();
    const { text, id } = toks[selected];
    const same = toks.filter((t) => t.id === id).length;
    const unique = new Set(toks.map((t) => t.id)).size;
    fRows.innerHTML =
      `<b>${text.replace(/</g, '&lt;')}</b> is id <b>${id.toLocaleString('en-US')}</b>` +
      (same > 1 ? `; it appears ${same === 2 ? 'twice' : `${same} times`}, and each copy reads the same row.` : '.') +
      ` These ${n} tokens read <b>${unique}</b> distinct rows. "The" and "the" are different tokens.` +
      ' Ids other than the first are illustrative.';
    const rand = rng(id);
    const normal = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
    // The whole row (embedding entries are small numbers, a few hundredths); the chart shows both ends of it.
    const row = Array.from({ length: model.dModel }, () => 0.02 * normal());
    const d = model.dModel;
    const hidden = d - ROW_HEAD - ROW_TAIL;
    // Columns: the first ROW_HEAD features, one gap column, the last ROW_TAIL.
    const feature = (i: number) => (i < ROW_HEAD ? i : d - (ROW_HEAD + 1 + ROW_TAIL - i));
    const values = Array.from({ length: ROW_HEAD + 1 + ROW_TAIL }, (_, i) => (i === ROW_HEAD ? 0 : row[feature(i)]));
    const span = Math.max(...values.map(Math.abs));
    cRow.root.querySelector('figcaption')?.remove();
    const caption = el('figcaption', { textContent: `row ${id.toLocaleString('en-US')} (${text})` });
    caption.append(el('span', { textContent: `all ${d.toLocaleString('en-US')} features: the first ${ROW_HEAD}, … ${hidden.toLocaleString('en-US')} more …, the last ${ROW_TAIL}` }));
    cRow.root.prepend(caption);
    cRow.draw({
      values, lo: -span, hi: span, gap: ROW_HEAD,
      ticks: [0, 16, 32, ROW_HEAD, ROW_HEAD + ROW_TAIL],
      tickLabel: (i) => (i === ROW_HEAD ? `+${hidden.toLocaleString('en-US')}` : feature(i).toLocaleString('en-US')),
      describe: (i) => (i === ROW_HEAD
        ? `features ${ROW_HEAD.toLocaleString('en-US')}–${(d - ROW_TAIL - 1).toLocaleString('en-US')}: ${hidden.toLocaleString('en-US')} not shown`
        : `feature ${feature(i).toLocaleString('en-US')} · ${values[i].toFixed(4)}`),
    });
  }
  update();
}

// ---------------------------------------------------------------------------------------------
// Attention: softmax(QKᵀ/√d + mask) · V for one head

const ATTN_TOKENS = ['<bos>', 'The', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired', '.'];
/** A token for innerHTML: <bos> would otherwise be parsed as a tag. */
const tokenHtml = (i: number) => ATTN_TOKENS[i].replace(/</g, '&lt;').replace(/>/g, '&gt;');

/**
 * Illustrative scaled scores of one head (queries × keys), with patterns real heads show: a sink on the first token,
 * a pull toward the previous token, and a few content links ("it" → "cat"). Computed for the full square, as eager
 * attention does; the causal mask removes the upper half.
 */
function illustrativeScores(): number[][] {
  const rand = rng(3);
  const normal = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const links: Record<string, number> = { '8,2': 4.5, '10,2': 3, '9,8': 2, '3,2': 2, '6,4': 1.5, '11,10': 1, '7,3': 1.2 };
  return ATTN_TOKENS.map((_, i) => ATTN_TOKENS.map((_, j) =>
    0.6 * normal() + (j === 0 ? 3 : 0) + (j === i ? 1 : 0) + (j === i - 1 ? 1.5 : 0) + (links[`${i},${j}`] ?? 0)));
}

/** Linear interpolation between two hex colors. */
function mix(a: string, b: string, t: number): string {
  const p = (h: string) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16));
  const [ca, cb] = [p(a), p(b)];
  return `rgb(${ca.map((v, k) => Math.round(v + (cb[k] - v) * Math.max(0, Math.min(1, t)))).join(',')})`;
}

/** Diverging for scores: blue below 0, neutral gray at 0, orange above. */
const scoreColor = (v: number, max: number) => (v < 0 ? mix('#334155', '#60a5fa', -v / max) : mix('#334155', '#f5a524', v / max));
/** Sequential for probabilities: dark at 0, bright at the maximum. */
const probColor = (p: number, max: number) => mix('#1e293b', '#fbbf24', Math.sqrt(p / max));

interface HeatmapDraw {
  value: (i: number, j: number) => number;
  color: (v: number) => string;
  masked: (i: number, j: number) => boolean;
  selected: number;
  describe: (i: number, j: number) => string;
  /** Legend: a gradient from lo to hi. */
  legend: { lo: number; hi: number; label: string; format: (v: number) => string };
}

/** A queries × keys heatmap with token labels, hatched masked cells, a selectable query row and hover tooltips. */
function heatmap(onSelect: (row: number) => void) {
  const n = ATTN_TOKENS.length;
  const cell = 30;
  const left = 70;
  const top = 58;
  const W = left + n * cell + 8;
  const H = top + n * cell + 34;
  const root = el('figure', { className: 'op-chart heatmap' });
  const plot = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Queries by keys' });
  const tip = el('div', { className: 'op-chart-tip' });
  tip.hidden = true;
  root.append(plot, tip);

  function draw(d: HeatmapDraw) {
    plot.replaceChildren();
    const defs = svg('defs');
    const pattern = svg('pattern', { id: 'hatch', width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' });
    pattern.append(svg('line', { x1: 0, y1: 0, x2: 0, y2: 6, class: 'hatch' }));
    defs.append(pattern);
    plot.append(defs);
    const keysLabel = svg('text', { x: left + (n * cell) / 2, y: 10, class: 'tick', 'text-anchor': 'middle' });
    keysLabel.textContent = 'keys →';
    const queriesLabel = svg('text', { x: 4, y: top - 6, class: 'tick' });
    queriesLabel.textContent = 'queries ↓';
    plot.append(keysLabel, queriesLabel);
    ATTN_TOKENS.forEach((tok, k) => {
      const col = svg('text', { x: left + k * cell + cell / 2, y: top - 6, class: 'tick', 'text-anchor': 'start', transform: `rotate(-45 ${left + k * cell + cell / 2} ${top - 6})` });
      col.textContent = tok;
      const row = svg('text', { x: left - 6, y: top + k * cell + cell / 2 + 3, class: k === d.selected ? 'tick selected' : 'tick', 'text-anchor': 'end' });
      row.textContent = tok;
      plot.append(col, row);
    });
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const x = left + j * cell;
        const y = top + i * cell;
        const masked = d.masked(i, j);
        plot.append(svg('rect', { x: x + 1, y: y + 1, width: cell - 2, height: cell - 2, rx: 3, fill: masked ? 'url(#hatch)' : d.color(d.value(i, j)), class: 'cell' }));
        const hit = svg('rect', { x, y, width: cell, height: cell, class: 'hit' });
        hit.addEventListener('pointerenter', () => {
          tip.textContent = d.describe(i, j);
          tip.hidden = false;
          placeTip(tip, (x + cell / 2) / W);
          tip.style.top = `${((y - 4) / H) * 100}%`;
        });
        hit.addEventListener('pointerleave', () => (tip.hidden = true));
        hit.addEventListener('click', () => onSelect(i));
        plot.append(hit);
      }
    }
    plot.append(svg('rect', { x: left, y: top + d.selected * cell, width: n * cell, height: cell, rx: 4, class: 'row-outline' }));

    // Legend: a gradient strip with its ends labeled.
    const ly = top + n * cell + 12;
    const lw = 160;
    const grad = svg('linearGradient', { id: `g-${d.legend.label.length}-${d.legend.hi}` });
    for (let s = 0; s <= 10; s++) {
      grad.append(svg('stop', { offset: `${s * 10}%`, 'stop-color': d.color(d.legend.lo + ((d.legend.hi - d.legend.lo) * s) / 10) }));
    }
    defs.append(grad);
    plot.append(svg('rect', { x: left, y: ly, width: lw, height: 8, rx: 2, fill: `url(#${grad.id})` }));
    const lo = svg('text', { x: left, y: ly + 20, class: 'tick' });
    lo.textContent = d.legend.format(d.legend.lo);
    const hi = svg('text', { x: left + lw, y: ly + 20, class: 'tick', 'text-anchor': 'end' });
    hi.textContent = d.legend.format(d.legend.hi);
    const name = svg('text', { x: left + lw + 10, y: ly + 8, class: 'tick' });
    name.textContent = d.legend.label;
    plot.append(svg('rect', { x: left + lw + 150, y: ly - 1, width: 10, height: 10, rx: 2, fill: 'url(#hatch)', class: 'cell' }));
    const maskName = svg('text', { x: left + lw + 164, y: ly + 8, class: 'tick' });
    maskName.textContent = 'masked';
    plot.append(lo, hi, name, maskName);
  }
  return { root, draw };
}

function attentionViz(body: HTMLElement, focus: AttnStage): void {
  const n = ATTN_TOKENS.length;
  const raw = illustrativeScores();
  const perGroup = model.nHeads / model.nKvHeads;
  let selected = 8; // "it"
  let causal = true;

  // 1. Scores.
  const sScores = stageSection(body, focus === 'scores', 'QKᵀ / √d: every query against every key',
    `One head of ${model.nHeads}: query i's score for key j is q_i · k_j / √${model.headDim}. Illustrative values for ` +
    `${n} tokens, with patterns real heads show: a sink on the first token, the previous token, and "it" → "cat".`);
  const maskRow = el('label', { className: 'op-popup-control' });
  const maskBox = el('input', { type: 'checkbox', checked: true });
  maskRow.append(maskBox, el('span', { textContent: 'Causal mask: a query may not see later keys (−∞ before softmax)' }));
  const hScores = heatmap((i) => select(i));
  const fScores = el('div', { className: 'op-popup-facts' });
  sScores.append(maskRow, hScores.root, fScores);

  // 2. Softmax.
  const sSoft = stageSection(body, focus === 'softmax', 'softmax over keys, row by row',
    'Each query row becomes probabilities that sum to 1; masked keys get exactly 0. Click a row to follow that query.');
  const hProbs = heatmap((i) => select(i));
  sSoft.append(hProbs.root);

  // 3. Output.
  const sOut = stageSection(body, focus === 'output', '· V: a weighted mix of value rows',
    'The output of a query is the probability-weighted sum of the V rows of the keys it attends to.');
  const cRow = barChart('', '');
  const fOut = el('div', { className: 'op-popup-facts' });
  sOut.append(cRow.root, fOut);

  // The real sizes of this pass.
  const fCost = el('div', { className: 'op-popup-facts' });
  body.append(fCost);
  const T = pass.T;
  const decode = pass.mode === 'decode';
  const scoreBytes = model.nHeads * (decode ? T + 1 : T * T) * 2;
  fCost.innerHTML = decode
    ? `In this decode step the grid is a single row: the new token (position ${T.toLocaleString('en-US')}) against all ` +
      `<b>${(T + 1).toLocaleString('en-US')}</b> cached keys, so there is nothing to mask. Each layer reads its whole K and V cache for it.`
    : `At T = <b>${T.toLocaleString('en-US')}</b> each head's grid is ${T.toLocaleString('en-US')} × ${T.toLocaleString('en-US')}: ` +
      `<b>${formatBytes(scoreBytes)}</b> per layer for ${model.nHeads} heads in bf16, and eager attention allocates it several times ` +
      '(scores, masked, fp32 softmax, cast). Fused kernels (FlashAttention / SDPA) never write it: they stream key blocks ' +
      `and keep a running softmax. Grouped-query attention: this head shares its K and V with ${perGroup - 1} other query heads.`;

  function probsOf(i: number): number[] {
    const row = raw[i].map((v, j) => (causal && j > i ? -Infinity : v));
    const m = Math.max(...row);
    const e = row.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  function select(i: number) {
    selected = i;
    update();
  }

  function update() {
    const probs = raw.map((_, i) => probsOf(i));
    const isMasked = (i: number, j: number) => causal && j > i;
    const maxAbs = Math.max(...raw.flat().map(Math.abs));
    hScores.draw({
      value: (i, j) => raw[i][j], color: (v) => scoreColor(v, maxAbs), masked: isMasked, selected,
      describe: (i, j) => `${ATTN_TOKENS[i]} → ${ATTN_TOKENS[j]} · score ${raw[i][j].toFixed(2)}${isMasked(i, j) ? ' · masked: −∞' : ''}`,
      legend: { lo: -maxAbs, hi: maxAbs, label: 'score', format: (v) => v.toFixed(1) },
    });
    fScores.innerHTML = causal
      ? `<b>${(n * (n - 1)) / 2}</b> of ${n * n} scores are masked. Eager attention still computes them, then adds −∞; ` +
        'fused kernels skip those blocks, about half the work.'
      : 'Without the mask, early tokens would read later ones: fine for an encoder, but a decoder generating left to ' +
        'right cannot see the future, so training would leak the answer.';
    const maxP = Math.max(...probs.flat());
    hProbs.draw({
      value: (i, j) => probs[i][j], color: (p) => probColor(p, maxP), masked: isMasked, selected,
      describe: (i, j) => `${ATTN_TOKENS[i]} → ${ATTN_TOKENS[j]} · p = ${(100 * probs[i][j]).toFixed(1)}%`,
      legend: { lo: 0, hi: maxP, label: 'probability', format: (p) => `${Math.round(100 * p)}%` },
    });
    const p = probs[selected];
    cRow.draw({
      values: p, lo: 0, hi: Math.max(...p), percent: true, ticks: ATTN_TOKENS.map((_, j) => j), tickLabel: (j) => ATTN_TOKENS[j],
      removed: p.map((_, j) => isMasked(selected, j)),
      describe: (j) => (isMasked(selected, j) ? `${ATTN_TOKENS[j]} · masked: 0` : `${ATTN_TOKENS[j]} · ${(100 * p[j]).toFixed(1)}% of its V row`),
    });
    const order = p.map((v, j) => [v, j]).sort((a, b) => b[0] - a[0]);
    fOut.innerHTML =
      `Query <b>${tokenHtml(selected)}</b> takes <b>${(100 * order[0][0]).toFixed(0)}%</b> of its output from ` +
      `<b>${tokenHtml(order[0][1])}</b> and <b>${(100 * order[1][0]).toFixed(0)}%</b> from <b>${tokenHtml(order[1][1])}</b>. ` +
      'The first token often soaks up attention a head does not need elsewhere: an "attention sink".';
  }
  maskBox.addEventListener('change', () => {
    causal = maskBox.checked;
    update();
  });
  update();
  ({ scores: sScores, softmax: sSoft, output: sOut })[focus].scrollIntoView({ block: 'center' });
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
  /** A column standing for values left out: drawn as an ellipsis, with no bar. */
  gap?: number;
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

  function draw({ values, lo, hi, describe, ghost, removed, picked, percent, refs = [], gap, yTick, tickLabel = (i) => `#${i + 1}`, ticks = [0, 9, 19, 29, 39] }: Draw) {
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
      if (i === gap) {
        // Columns left out: an ellipsis on the baseline between two dashed lines, instead of a bar.
        for (const x of [cx - band / 2, cx + band / 2]) {
          plot.append(svg('line', { x1: x, x2: x, y1: pad.t, y2: H - pad.b, class: 'gap-line' }));
        }
        const dots = svg('text', { x: cx, y: y(0) + 4, class: 'gap-mark', 'text-anchor': 'middle' });
        dots.textContent = '…';
        plot.append(dots);
      }
      const cls = removed?.[i] ? 'ghost' : i === picked ? 'bar picked' : tone === 'weight' ? 'bar weight' : 'bar';
      if (i === gap) {
        const hit = svg('rect', { x: cx - band / 2, y: pad.t, width: band, height: H - pad.t - pad.b, class: 'hit' });
        hit.addEventListener('pointerenter', () => {
          tip.textContent = describe(i);
          tip.hidden = false;
          placeTip(tip, cx / W);
        });
        hit.addEventListener('pointerleave', () => (tip.hidden = true));
        plot.append(hit);
        return;
      }
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
        placeTip(tip, cx / W);
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

/**
 * Puts a chart tooltip at `fraction` of the chart's width, sliding its anchor from its left edge (at 0) to its right
 * edge (at 1), so it never sticks out of the chart.
 */
function placeTip(tip: HTMLElement, fraction: number): void {
  const f = Math.max(0, Math.min(1, fraction));
  tip.style.left = `${f * 100}%`;
  tip.style.transform = `translateX(${-f * 100}%)`;
}
