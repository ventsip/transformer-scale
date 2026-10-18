import { formatBytes, formatFlops, formatInt, formatPercent } from '../format';
import { LLAMA_3_8B, type ModelConfig } from '../model/config';
import { DTYPES, type DType } from '../model/dtypes';
import { routing } from '../model/routing';
import { PALETTE, cssColor } from '../palette';
import { gaussian, rng } from '../random';
import { floorLayer, type Mode, type OpSpec, type OpViz, type VizStage } from '../model/tensors';

/** The current model: popups take its vocabulary, RoPE base, head and context sizes. */
let model: ModelConfig = LLAMA_3_8B;
/** The pass being drawn: popups quote its real sizes next to their illustrations. */
let pass: { mode: Mode; T: number; weightDtype: DType; actDtype: DType } = { mode: 'prefill', T: 2048, weightDtype: 'bf16', actDtype: 'bf16' };

export function setPopupContext(cfg: ModelConfig, current: { mode: Mode; T: number; weightDtype: DType; actDtype: DType }): void {
  model = cfg;
  pass = { mode: current.mode, T: current.T, weightDtype: current.weightDtype, actDtype: current.actDtype };
}

type Stage = VizStage<'sampling'>;
type GluStage = VizStage<'swiglu'>;
type MoeStage = VizStage<'moe'>;
type AttnStage = VizStage<'attention'>;
type RopeStage = VizStage<'rope'>;
type NormStage = VizStage<'rmsnorm'>;

/** Each popup panel: its title, and how to fill the popup body for an op tagged with it (see OpSpec.viz). */
const PANELS: { [P in OpViz['panel']]: { title: string; render: (body: HTMLElement, viz: Extract<OpViz, { panel: P }>, op: OpSpec) => void } } = {
  lookup: { title: 'Embedding lookup: token ids select rows', render: (body) => lookupViz(body) },
  sampling: { title: 'Sampling: from logits to the next token', render: (body, viz) => samplingViz(body, viz.stage) },
  moe: { title: 'Mixture of experts: route, dispatch, combine', render: (body, viz, op) => moeViz(body, viz.stage, floorLayer(op.floor)) },
  attention: { title: 'Attention: softmax(QKᵀ/√d) · V, one head', render: (body, viz) => attentionViz(body, viz.stage) },
  rope: { title: 'RoPE: rotary position embedding', render: (body, viz) => ropeViz(body, viz.stage) },
  rmsnorm: { title: 'RMSNorm: x / RMS(x) · weight', render: (body, viz) => rmsNormViz(body, viz.stage) },
  swiglu: { title: 'SwiGLU: a = SiLU(g) ⊙ u', render: (body, viz) => swigluViz(body, viz.stage) },
  gelu: { title: 'GELU: a = h · Φ(h), tanh approximation', render: (body) => geluViz(body) },
  layernorm: { title: 'LayerNorm: (x − mean) / std · weight + bias', render: (body) => layerNormViz(body) },
};

/** The interactive view of an op: its title and how to fill the popup body; null for ops without one. */
function vizFor(op: OpSpec): { title: string; render: (body: HTMLElement) => void } | null {
  const viz = op.viz;
  if (!viz) return null;
  const panel = PANELS[viz.panel] as { title: string; render: (body: HTMLElement, viz: OpViz, op: OpSpec) => void };
  return { title: panel.title, render: (body) => panel.render(body, viz, op) };
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


/**
 * Illustrative last-position logits: 60 candidates falling from about 18 to about 0, steeply at first and
 * flattening out, with some jitter; then the long tail of the remaining vocabulary well below, around −8.
 * Sorted descending, so index = rank − 1.
 */
function illustrativeLogits(): Float64Array {
  const rand = rng(7);
  const normal = gaussian(rand);
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
  const sLogits = section('logits', 'logits (fp32)', `The LM head scores every token in the vocabulary. Illustrative values; the top ${SHOWN} of ${formatInt(model.vocabSize)} shown, by rank.`);
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

  const pct = (x: number) => formatPercent(x, x < 0.1 ? 2 : 3);

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
    // After the top SHOWN ranks, a gap band stands for the rest of the vocabulary (no bars: their logits sit far below).
    const V = model.vocabSize;
    const rest = V - SHOWN;
    const withGap = <T>(xs: T[], fill: T) => [...xs, ...Array<T>(GAP_COLUMNS).fill(fill)];
    const ranks = `ranks ${formatInt(SHOWN + 1)}–${formatInt(V)}: ${formatInt(rest)} more tokens`;
    const axis = {
      gap: { at: SHOWN, columns: GAP_COLUMNS }, ticks: [0, 9, 19, 29, SHOWN + 1],
      tickLabel: (i: number) => (i >= SHOWN ? `+${formatInt(rest)}` : `#${i + 1}`),
    };
    const range = (xs: Float64Array) => `from ${xs[SHOWN].toFixed(1)} down to ${xs[V - 1].toFixed(1)}`;
    cLogits.draw({
      values: withGap(input, 0), lo: Math.min(0, ...input), hi: Math.max(...input), ...axis,
      describe: (i) => (i >= SHOWN ? `${ranks}, logits ${range(logits)}` : `#${i + 1} · logit ${input[i].toFixed(2)}`),
    });
    const sLo = Math.min(0, ...input, ...scaledShown);
    const sHi = Math.max(...input, ...scaledShown);
    cTemp.draw({
      values: withGap(scaledShown, 0), ghost: withGap(input, 0), lo: sLo, hi: sHi, ...axis,
      describe: (i) => (i >= SHOWN ? `${ranks}, scaled ${range(scaled)}` : `#${i + 1} · ${input[i].toFixed(2)} ÷ ${t.toFixed(2)} = ${scaledShown[i].toFixed(2)}`),
    });
    cFilter.draw({
      values: withGap(scaledShown, 0), removed: withGap(scaledShown.map((_, i) => i >= kept), true), lo: sLo, hi: sHi, ...axis,
      describe: (i) => (i >= SHOWN
        ? `${ranks}: ${kept > SHOWN ? `${formatInt(kept - SHOWN)} kept, the rest` : 'all'} removed → −∞`
        : i < kept ? `#${i + 1} · kept · ${scaledShown[i].toFixed(2)}` : `#${i + 1} · removed → −∞`),
    });
    const shownProbs = scaledShown.map((_, i) => (i < kept ? probs[i] : 0));
    const restMass = probs.slice(SHOWN).reduce((a, b) => a + b, 0);
    cSoft.draw({
      values: withGap(shownProbs, 0), lo: 0, hi: Math.max(...shownProbs), percent: true, picked, ...axis,
      describe: (i) => (i >= SHOWN ? `${ranks}: together p = ${pct(restMass)}` : `#${i + 1} · p = ${pct(shownProbs[i])}`),
    });

    filterFacts.innerHTML =
      `Top-k keeps <b>${formatInt(afterK)}</b> of ${formatInt(model.vocabSize)} tokens; top-p then keeps <b>${formatInt(kept)}</b>` +
      (kept > SHOWN ? ` (only the first ${SHOWN} are shown).` : '.');
    let cum = 0;
    let for90 = 0;
    while (for90 < kept && cum < 0.9) cum += probs[for90++];
    softFacts.innerHTML =
      `The top token gets <b>${pct(probs[0])}</b>; <b>${formatInt(for90)}</b> ${for90 === 1 ? 'token makes' : 'tokens make'} up 90% of the probability.`;
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
/** Illustrative pre-activations: the gate spread wider than u, so SiLU's bend shows. */
const GATE_SCALE = 1.6;
/** Features of d_ffn shown for one illustrative token: the first GLU_HEAD and the last GLU_TAIL. */
const GLU_HEAD = 24;
const GLU_TAIL = 8;

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
    `SiLU(g) decides, feature by feature, how much of u passes. Illustrative values for one token, from both ends of its ` +
    `${formatInt(model.dFfn)} d_ffn features, all on one scale.`);
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
    const normal = gaussian(rand);
    // Whole d_ffn vectors; the charts show both ends of them.
    const gFull = Array.from({ length: model.dFfn }, () => GATE_SCALE * normal());
    const uFull = Array.from({ length: model.dFfn }, () => normal());
    const ends = bothEnds(model.dFfn, GLU_HEAD, GLU_TAIL, [0, 8, 16]);
    const g = ends.values((k) => gFull[k]);
    const u = ends.values((k) => uFull[k]);
    const sg = g.map(silu);
    const a = sg.map((v, i) => v * u[i]);
    const all = [...g, ...u, ...a];
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    const f = (v: number) => v.toFixed(2);
    const axis = { lo, hi, gap: ends.gap, ticks: ends.ticks, tickLabel: ends.tickLabel };
    const feature = (i: number) => `feature ${formatInt(ends.index(i))}`;
    const describe = (text: (i: number) => string) => (i: number) => (ends.inGap(i) ? ends.gapText('features') : `${feature(i)} · ${text(i)}`);
    cGate.draw({ values: sg, ghost: g, ...axis, describe: describe((i) => `g = ${f(g[i])} → SiLU(g) = ${f(sg[i])}`) });
    cUp.draw({ values: u, ...axis, describe: describe((i) => `u = ${f(u[i])}`) });
    cOut.draw({ values: a, ghost: u, ...axis, describe: describe((i) => `${f(sg[i])} × ${f(u[i])} = ${f(a[i])}`) });
    const closed = gFull.filter((v) => v < 0).length;
    facts.innerHTML =
      `<b>${formatInt(closed)}</b> of ${formatInt(model.dFfn)} gates are negative: ` +
      `SiLU squeezes them into [${silu(SILU_MIN_X).toFixed(2)}, 0], ` +
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

// ---------------------------------------------------------------------------------------------
// GELU (GPT-2): a = GELU(h_fc), the tanh approximation ("gelu_new")

/** Error function, Abramowitz & Stegun 7.1.26 (error below 1.5e-7). */
function erf(x: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-x * x);
  return x < 0 ? -y : y;
}
/** Φ, the standard normal CDF. */
const phi = (x: number) => 0.5 * (1 + erf(x / Math.SQRT2));
/** Exact GELU: x · Φ(x). */
const geluExact = (x: number) => x * phi(x);
/** GPT-2's gelu_new, as HF NewGELUActivation computes it. */
const GELU_CUBIC = 0.044715;
const geluTanh = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + GELU_CUBIC * x ** 3)));

/** Where `f` is smallest on [lo, hi], for a function with one minimum there (ternary search). */
function argmin(f: (x: number) => number, lo: number, hi: number): number {
  for (let i = 0; i < 100; i++) {
    const a = lo + (hi - lo) / 3;
    const b = hi - (hi - lo) / 3;
    if (f(a) < f(b)) hi = b;
    else lo = a;
  }
  return (lo + hi) / 2;
}

/** The range the GELU curve is drawn over. */
const GELU_RANGE = 6;

/**
 * HF NewGELUActivation, 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³))), one PyTorch kernel per step: what each
 * reads and writes, in full T × d_ffn tensors (Python evaluates 0.5 · x first).
 */
const GELU_KERNELS: { what: string; reads: number }[] = [
  { what: '0.5 · x', reads: 1 },
  { what: 'x³', reads: 1 },
  { what: '0.044715 · x³', reads: 1 },
  { what: 'x + …', reads: 2 },
  { what: '√(2/π) · …', reads: 1 },
  { what: 'tanh(…)', reads: 1 },
  { what: '1 + …', reads: 1 },
  { what: '0.5x · (…)', reads: 2 },
];

function geluViz(body: HTMLElement): void {
  const minX = argmin(geluTanh, -3, 0);
  let maxDiff = 0;
  for (let x = -GELU_RANGE; x <= GELU_RANGE; x += 0.001) maxDiff = Math.max(maxDiff, Math.abs(geluTanh(x) - geluExact(x)));

  const sCurve = stageSection(body, true, 'GELU(x) = x · Φ(x)',
    'Φ(x) is the probability that a standard normal value is below x: GELU passes x weighted by how likely it is to be ' +
    'large. Like ReLU far from 0, smooth near it. GPT-2 uses the tanh approximation ("gelu_new"). No parameters.');
  sCurve.append(lineChart({
    caption: '<span class="op-legend"><i class="swatch-line"></i>GELU(x), tanh approximation</span>' +
      '<span class="op-legend"><i class="swatch-ref"></i>ReLU(x) = max(0, x), for reference</span>',
    label: `GELU and ReLU from x = −${GELU_RANGE} to ${GELU_RANGE}`,
    x: { lo: -GELU_RANGE, hi: GELU_RANGE, ticks: [-6, -4, -2, 0, 2, 4, 6], tickLabel: String },
    y: { lo: -1, hi: GELU_RANGE, ticks: [-1, 0, 1, 2, 3, 4, 5, 6] },
    f: geluTanh,
    ref: (x) => Math.max(0, x),
    marker: { x: minX, label: `min ${geluTanh(minX).toFixed(3)}` },
    describe: (x) => `x = ${x.toFixed(2)} · Φ(x) = ${phi(x).toFixed(3)} · GELU = ${geluTanh(x).toFixed(3)} · exact ${geluExact(x).toFixed(3)} · ReLU = ${Math.max(0, x).toFixed(2)}`,
  }));
  const fCurve = el('div', { className: 'op-popup-facts' });
  fCurve.innerHTML =
    `Its minimum is <b>${geluTanh(minX).toFixed(3)}</b> at x = <b>${minX.toFixed(2)}</b>. The tanh form differs from ` +
    `the exact x · Φ(x) by at most <b>${maxDiff.toExponential(1)}</b> on this range, below bf16's step of <b>2⁻⁷ ≈ 0.008</b> near 1: in bf16 the two forms barely differ. ` +
    `SwiGLU's SiLU, x · σ(x), is a close cousin: x · σ(1.702x) approximates GELU.`;
  sCurve.append(fCurve);

  const sApply = stageSection(body, false, 'a = GELU(h_fc): one token',
    `h_fc = h_norm · W_fc + b, ${formatInt(model.dFfn)} features wide (4 · d_model). Illustrative values from both ends ` +
    'of them; unlike SwiGLU there is no second projection: the activation alone decides what passes.');
  const resample = el('button', { textContent: 'Another token' });
  const row = el('div', { className: 'op-popup-control' });
  row.append(resample);
  const cIn = barChart('h_fc', '');
  const cOut = barChart('a = GELU(h_fc)', 'outlines: h_fc');
  const fApply = el('div', { className: 'op-popup-facts' });
  sApply.append(row, cIn.root, cOut.root, fApply);

  let seed = 13;
  function update() {
    const normal = gaussian(rng(seed));
    const hFull = Array.from({ length: model.dFfn }, () => GATE_SCALE * normal());
    const ends = bothEnds(model.dFfn, GLU_HEAD, GLU_TAIL, [0, 8, 16]);
    const h = ends.values((k) => hFull[k]);
    const a = h.map(geluTanh);
    const lo = Math.min(...h, ...a);
    const hi = Math.max(...h, ...a);
    const f = (v: number) => v.toFixed(2);
    const axis = { lo, hi, gap: ends.gap, ticks: ends.ticks, tickLabel: ends.tickLabel };
    const describe = (text: (i: number) => string) => (i: number) =>
      (ends.inGap(i) ? ends.gapText('features') : `feature ${formatInt(ends.index(i))} · ${text(i)}`);
    cIn.draw({ values: h, ...axis, describe: describe((i) => `h_fc = ${f(h[i])}`) });
    cOut.draw({ values: a, ghost: h, ...axis, describe: describe((i) => `GELU(${f(h[i])}) = ${f(a[i])}`) });
    const negative = hFull.filter((v) => v < 0).length;
    fApply.innerHTML =
      `<b>${formatInt(negative)}</b> of ${formatInt(model.dFfn)} features are negative: GELU squeezes them into ` +
      `[${geluTanh(minX).toFixed(2)}, 0]. Positive ones pass almost unchanged once above ~2.`;
  }
  resample.addEventListener('click', () => {
    seed++;
    update();
  });
  update();

  const sCost = stageSection(body, false, 'In memory: eight kernels',
    'HF writes gelu_new in plain PyTorch, so each step is its own kernel with its own full-size output. The scene draws ' +
    'it as one op; a fused kernel would read h_fc once and write a once.');
  const rows = pass.mode === 'decode' ? 1 : pass.T;
  const tensor = rows * model.dFfn * DTYPES[pass.actDtype].bytes;
  const reads = GELU_KERNELS.reduce((s, k) => s + k.reads, 0);
  const traffic = (reads + GELU_KERNELS.length) * tensor;
  const fCost = el('div', { className: 'op-popup-facts' });
  fCost.innerHTML =
    `${GELU_KERNELS.map((k) => k.what).join(' → ')}: <b>${GELU_KERNELS.length}</b> kernels, ${reads} tensor reads and ` +
    `${GELU_KERNELS.length} writes. In this pass each tensor is ${formatInt(rows)} × ${formatInt(model.dFfn)} ` +
    `${DTYPES[pass.actDtype].label} = <b>${formatBytes(tensor)}</b>, so GELU moves <b>${formatBytes(traffic)}</b> per layer, ` +
    `against <b>${formatBytes(2 * tensor)}</b> fused: ${((reads + GELU_KERNELS.length) / 2).toFixed(0)}× more, for about ` +
    `${formatFlops(rows * model.dFfn * GELU_KERNELS.length)} FLOPs of arithmetic (about one per element per kernel).`;
  sCost.append(fCost);
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
  const W = CHART_WIDTH;
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
  const caption = el('figcaption');
  caption.innerHTML = o.caption;
  const { root, plot, tip } = chartFrame({ height: H, label: o.label, caption });

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
    tip.show(o.describe(x), px(x));
  });
  hit.addEventListener('pointerleave', () => {
    for (const node of [cross, dot]) node.setAttribute('visibility', 'hidden');
    tip.hide();
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
  return n >= 100 || Number.isInteger(n) ? formatInt(Math.round(n)) : n.toPrecision(2);
}

function ropeViz(body: HTMLElement, focus: RopeStage): void {
  const pairs = model.headDim / 2;
  const base = model.ropeTheta!;
  const theta = Array.from({ length: pairs }, (_, i) => base ** ((-2 * i) / model.headDim));
  const wavelength = theta.map((t) => (2 * Math.PI) / t);
  const ctx = model.maxContext;

  // 1. Frequencies.
  const sFreq = stageSection(body, focus === 'freq', 'θ_i: one frequency per dimension pair',
    `Each head's ${model.headDim} dimensions form ${pairs} pairs; pair i rotates by θ_i = ` +
    `${formatInt(base)}^(−2i/${model.headDim}) radians per token. ` +
    `The cos and sin tables hold pos · θ_i for every position: the ${pairs} angles, written twice to fill head_dim.`);
  const cFreq = barChart('wavelength 2π/θ_i in tokens (log scale), by pair', 'line: context length');
  const fFreq = el('div', { className: 'op-popup-facts' });
  sFreq.append(cFreq.root, fFreq);
  const logLambda = wavelength.map(Math.log10);
  cFreq.draw({
    values: logLambda, lo: 0, hi: Math.max(...logLambda, Math.log10(ctx)) * 1.05,
    ticks: [0, 16, 32, 48, pairs - 1], tickLabel: String, yTick: (v) => tokens(10 ** v),
    refs: [{ value: Math.log10(ctx), label: `context ${formatInt(ctx)}`, start: true }],
    describe: (i) => `pair ${i} · θ = ${theta[i].toPrecision(3)} rad/token · one turn every ${tokens(wavelength[i])} tokens`,
  });
  const slow = wavelength.filter((l) => l > ctx).length;
  fFreq.innerHTML =
    `Wavelengths run from <b>${tokens(wavelength[0])}</b> to <b>${tokens(wavelength[pairs - 1])}</b> tokens. ` +
    `<b>${slow}</b> of ${pairs} pairs turn less than once within the ${formatInt(ctx)}-token context: ` +
    'fast pairs tell neighbors apart, slow pairs carry coarse, long-range position.';

  // 2. Rotation by position.
  const sRot = stageSection(body, focus === 'rotate', 'RoPE: rotate each pair by pos · θ_i',
    `q and k are split into pairs (HF rotate_half pairs dimension i with i + ${model.headDim / 2}, not neighbors), and each pair is ` +
    'rotated like a 2D vector. No parameters, nothing added: the vector keeps its length, only its angles change.');
  const maxLog = Math.round(Math.log2(ctx) * 100);
  const pos = el('input', { type: 'range', min: '0', max: String(maxLog), step: '1', value: '700' });
  const posShown = el('b');
  const posRow = el('label', { className: 'op-popup-control' });
  posRow.append(el('span', { textContent: 'Position' }), pos, posShown);
  const { root: dialFig, plot: dials } = chartFrame({ height: 250, label: 'Rotation of selected dimension pairs' });
  const fRot = el('div', { className: 'op-popup-facts' });
  sRot.append(posRow, dialFig, fRot);

  function drawDials() {
    const p = Math.round(2 ** (Number(pos.value) / 100)) - 1;
    posShown.textContent = formatInt(p);
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
      sub.textContent = turns >= 1 ? `${turns >= 100 ? formatInt(Math.round(turns)) : turns.toFixed(1)} turns` : `${((angle * 180) / Math.PI).toFixed(1)}°`;
      dials.append(label, sub);
    });
    fRot.innerHTML =
      `At position <b>${formatInt(p)}</b>, pair 0 has turned <b>${tokens(p * theta[0] / (2 * Math.PI))}</b> times ` +
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
    describe: (d) => `distance ${formatInt(d)} · score ${score(d).toFixed(3)}`,
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

/** Features of d_model shown for an expert output: the first MOE_HEAD and the last MOE_TAIL. */
const MOE_HEAD = 18;
const MOE_TAIL = 6;

function moeViz(body: HTMLElement, focus: MoeStage, layer: number): void {
  const { nExperts: E, topK: K } = model.moe!;
  const expertIds = Array.from({ length: E }, (_, e) => e);
  const expertTick = (e: number) => `E${e}`;
  const pct = (p: number) => `${(100 * p).toFixed(1)}%`;

  // 1. Router.
  const sRouter = stageSection(body, focus === 'router', `Router: softmax, top-${K}, renormalize`,
    `A ${formatInt(model.dModel)} × ${E} weight scores the ${E} experts for each token; softmax (in fp32) turns the ` +
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
    describe: (e) => (counts[e] ? `E${e} · ${formatInt(counts[e])} tokens` : `E${e} · idle: its weights are not read`),
  });
  const used = expertIds.filter((e) => counts[e] > 0);
  fDispatch.innerHTML = decode
    ? `The new token goes to experts <b>${used.join('</b> and <b>')}</b>; the other ${E - used.length} stay idle, so this ` +
      `layer reads only <b>${used.length}</b> of ${E} experts' weights. Generate to watch the pair change token by token.`
    : `${K} × ${formatInt(n)} = <b>${formatInt(K * n)}</b> token slots; the busiest expert gets ` +
      `<b>${formatInt(Math.max(...counts))}</b> (${(Math.max(...counts) / even).toFixed(2)}× its even share) and the quietest ` +
      `<b>${formatInt(Math.min(...counts))}</b>. HF's loop drops no tokens; training adds a load-balancing loss to keep this even.`;

  // 3. Combine.
  const sCombine = stageSection(body, focus === 'combine', 'Combine: m = Σ weight × expert output',
    `Each expert's output rows are scaled by their routing weights and added back at their tokens' positions (index_add_). ` +
    `Every row of m receives exactly ${K} contributions. Illustrative outputs for the token above, from both ends of their ${formatInt(model.dModel)} features.`);
  const cParts = Array.from({ length: K }, () => barChart('', ''));
  const cSum = barChart('m: the sum', 'outlines: the first expert alone');
  sCombine.append(...cParts.map((c) => c.root), cSum.root);

  let seed = 21;
  function update() {
    const rand = rng(seed);
    const normal = gaussian(rand);
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
      `The router is ${formatInt(model.dModel * E)} parameters per layer, against ` +
      `${(3 * model.dModel * model.dFfn / 1e6).toFixed(0)}M in each expert.`;

    // Illustrative expert outputs (different experts, different vectors), whole d_model rows shown by both ends.
    const ends = bothEnds(model.dModel, MOE_HEAD, MOE_TAIL, [0, 8]);
    const outputs = top.map(() => {
      const full = Array.from({ length: model.dModel }, () => normal());
      return ends.values((f) => full[f]);
    });
    const parts = outputs.map((y, k) => y.map((v) => v * weights[top[k]]));
    const total = parts[0].map((_, i) => parts.reduce((a, p) => a + p[i], 0));
    const all = [...outputs.flat(), ...parts.flat(), ...total];
    const axis = { lo: Math.min(...all), hi: Math.max(...all), gap: ends.gap, ticks: ends.ticks, tickLabel: ends.tickLabel };
    const describe = (text: (i: number) => string) => (i: number) =>
      (ends.inGap(i) ? ends.gapText('features') : `feature ${formatInt(ends.index(i))} · ${text(i)}`);
    parts.forEach((p, k) => {
      const e = top[k];
      cParts[k].root.querySelector('figcaption')?.remove();
      const caption = el('figcaption', { textContent: `${pct(weights[e])} × E${e}'s output` });
      caption.append(el('span', { textContent: `outlines: E${e}'s output before scaling` }));
      cParts[k].root.prepend(caption);
      cParts[k].draw({
        values: p, ghost: outputs[k], ...axis,
        describe: describe((i) => `${weights[e].toFixed(3)} × ${outputs[k][i].toFixed(2)} = ${p[i].toFixed(2)}`),
      });
    });
    cSum.draw({
      values: total, ghost: parts[0], ...axis,
      describe: describe((i) => `${parts.map((p) => p[i].toFixed(2)).join(' + ')} = ${total[i].toFixed(2)}`),
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
/** Illustrative token ids: from 1000 (clear of special tokens), below 100,000 or the vocabulary, minus a margin. */
const LOOKUP_IDS = { from: 1000, to: 100000, margin: 1000 };
/** Standard deviation of the illustrative embedding row: entries are a few hundredths. */
const EMBEDDING_STD = 0.02;
/** Features of the selected row shown as bars: the first ROW_HEAD and the last ROW_TAIL. */
const ROW_HEAD = 36;
const ROW_TAIL = 12;

/**
 * The prompt as (token, id) pairs. The beginning-of-text token is the model's real one (Llama 3: 128000, Mixtral: 1,
 * GPT-2: 50256); the others are illustrative, derived from the word so that repeated words share an id, as they do in
 * a tokenizer.
 */
function lookupTokens(): { text: string; id: number }[] {
  const bos = model.bos;
  const hash = (w: string) => [...w].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7);
  return [bos, ...LOOKUP_WORDS.map((w) => ({ text: w, id: LOOKUP_IDS.from + (hash(w) % (Math.min(model.vocabSize, LOOKUP_IDS.to) - LOOKUP_IDS.from - LOOKUP_IDS.margin)) }))];
}

function lookupViz(body: HTMLElement): void {
  const toks = lookupTokens();
  const n = toks.length;
  const V = model.vocabSize;
  let selected = 2; // "cat"

  // 1. Token ids → rows → x.
  const sRows = stageSection(body, true, 'Token ids select rows of embed_tokens',
    `The table has one row of ${formatInt(model.dModel)} numbers per vocabulary entry (${formatInt(V)} rows). ` +
    "Each token id picks its row, which is copied into x: no multiplication, just indexing. Hover a token to follow it.");
  const W = CHART_WIDTH;
  const rowH = 22;
  const top = 26;
  const H = top + n * rowH + 10;
  const table = { x: 250, w: 44, y: top, h: n * rowH - 4 };
  const { root: fig, plot: diagram } = chartFrame({ height: H, label: 'Token ids indexing rows of the embedding table' });
  const fRows = el('div', { className: 'op-popup-facts' });
  sRows.append(fig, fRows);

  // 2. The selected row.
  const sRow = stageSection(body, false, 'One row: a token\'s embedding',
    `Illustrative values from both ends of the ${formatInt(model.dModel)} features. Learned in training; ` +
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
    `In this pass: <b>${formatInt(rows)}</b> ${rows === 1 ? 'row' : 'rows'} × ${formatBytes(rowBytes)} = ` +
    `<b>${formatBytes(rows * rowBytes)}</b> read from a <b>${formatBytes(tableBytes)}</b> table ` +
    `(${((100 * rows * rowBytes) / tableBytes).toPrecision(2)}%), and <b>0</b> FLOPs. ` +
    (model.arch === 'gpt2'
      ? 'The LM head at the top is the opposite: a full matrix multiply against this very table (tied weights), ' +
        'scoring every vocabulary entry.'
      : 'The LM head at the top is the opposite: a full matrix multiply against a second table of the same shape ' +
        `(not tied to this one in ${model.name}), scoring every vocabulary entry.`);

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
    heading(table.x + table.w / 2, `embed_tokens: ${formatInt(V)} rows`, 'middle');
    heading(W - 4, 'x: one row per token', 'end');
    diagram.append(svg('rect', { x: table.x, y: table.y, width: table.w, height: table.h, rx: 3, class: 'table' }));
    const zero = svg('text', { x: table.x + table.w + 4, y: table.y + 8, class: 'tick' });
    zero.textContent = '0';
    const last = svg('text', { x: table.x + table.w + 4, y: table.y + table.h, class: 'tick' });
    last.textContent = formatInt(V - 1);
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
      idText.textContent = formatInt(id);
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
      `<b>${text.replace(/</g, '&lt;')}</b> is id <b>${formatInt(id)}</b>` +
      (same > 1 ? `; it appears ${same === 2 ? 'twice' : `${same} times`}, and each copy reads the same row.` : '.') +
      ` These ${n} tokens read <b>${unique}</b> distinct rows. "The" and "the" are different tokens.` +
      ' Ids other than the first are illustrative.';
    const rand = rng(id);
    const normal = gaussian(rand);
    // The whole row (embedding entries are small numbers, a few hundredths); the chart shows both ends of it.
    const row = Array.from({ length: model.dModel }, () => EMBEDDING_STD * normal());
    const ends = bothEnds(model.dModel, ROW_HEAD, ROW_TAIL, [0, 16, 32]);
    const values = ends.values((f) => row[f]);
    const span = Math.max(...values.map(Math.abs));
    cRow.root.querySelector('figcaption')?.remove();
    const caption = el('figcaption', { textContent: `row ${formatInt(id)} (${text})` });
    caption.append(el('span', {
      textContent: `all ${formatInt(model.dModel)} features: the first ${ROW_HEAD}, … ${formatInt(ends.hidden)} more …, the last ${ROW_TAIL}`,
    }));
    cRow.root.prepend(caption);
    cRow.draw({
      values, lo: -span, hi: span, gap: ends.gap, ticks: ends.ticks, tickLabel: ends.tickLabel,
      describe: (i) => (ends.inGap(i) ? ends.gapText('features') : `feature ${formatInt(ends.index(i))} · ${values[i].toFixed(4)}`),
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
  const normal = gaussian(rand);
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

/** Scores of 0 in the heatmap: the chart's neutral. */
const SCORE_ZERO = cssColor(PALETTE.neutral700);
/** Diverging for scores: blue below 0, neutral gray at 0, orange above. */
const scoreColor = (v: number, max: number) => (v < 0 ? mix(SCORE_ZERO, cssColor(PALETTE.blue400), -v / max) : mix(SCORE_ZERO, cssColor(PALETTE.activation), v / max));
/** Sequential for probabilities: dark at 0, bright at the maximum. */
const probColor = (p: number, max: number) => mix(cssColor(PALETTE.neutral800), cssColor(PALETTE.pulse), Math.sqrt(p / max));

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
  const { root, plot, tip } = chartFrame({ width: W, height: H, label: 'Queries by keys', className: 'op-chart heatmap' });

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
        tip.attach(hit, () => d.describe(i, j), x + cell / 2, y - 4);
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
  const scoreBytes = model.nHeads * (decode ? T + 1 : T * T) * DTYPES[pass.actDtype].bytes;
  fCost.innerHTML = decode
    ? `In this decode step the grid is a single row: the new token (position ${formatInt(T)}) against all ` +
      `<b>${formatInt(T + 1)}</b> cached keys, so there is nothing to mask. Each layer reads its whole K and V cache for it.`
    : `At T = <b>${formatInt(T)}</b> each head's grid is ${formatInt(T)} × ${formatInt(T)}: ` +
      `<b>${formatBytes(scoreBytes)}</b> per layer for ${model.nHeads} heads in bf16, and eager attention allocates it several times ` +
      '(scores, masked, fp32 softmax, cast). Fused kernels (FlashAttention / SDPA) never write it: they stream key blocks ' +
      'and keep a running softmax. ' +
      (perGroup > 1
        ? `Grouped-query attention: this head shares its K and V with ${perGroup - 1} other query heads.`
        : 'Multi-head attention: this head has K and V of its own, so the KV cache holds one pair per query head.');

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

/**
 * One illustrative token of the residual stream, with its outliers, and a norm's learned per-feature scales for it
 * (mostly 0.2–0.6, smaller on the outlier features). `normal` draws further values from the same generator.
 */
function illustrativeToken() {
  const normal = gaussian(rng(5));
  const base = Array.from({ length: NORM_FEATURES }, () => 0.8 * normal());
  for (const [i, v] of OUTLIERS) base[i] = v;
  const gamma = base.map(() => Math.abs(0.4 + 0.12 * normal()));
  for (const [i] of OUTLIERS) gamma[i] = 0.05;
  return { base, gamma, normal };
}

/** Column ticks and value format of the norm charts. */
const NORM_TICKS = [0, 16, 32, 48, NORM_FEATURES - 1];
const normValue = (v: number) =>
  Math.abs(v) >= 1000 ? formatInt(Math.round(v)) : Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(1) : v.toFixed(3);
/** The range of a chart of `vs`, including 0. */
const extentWithZero = (vs: number[]) => ({ lo: Math.min(0, ...vs), hi: Math.max(0, ...vs) });

function rmsNormViz(body: HTMLElement, focus: NormStage): void {
  const { base, gamma } = illustrativeToken();

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

  const ticks = NORM_TICKS;
  const tickLabel = String;
  const f = normValue;
  const extent = extentWithZero;

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

// ---------------------------------------------------------------------------------------------
// LayerNorm (GPT-2): (x − mean(x)) / √(var(x) + ε) · weight + bias

function layerNormViz(body: HTMLElement): void {
  const { base, gamma, normal } = illustrativeToken();
  // Learned biases: illustrative, small.
  const beta = base.map(() => 0.1 * normal());
  const eps = model.normEps;
  const mean = (vs: number[]) => vs.reduce((a, b) => a + b, 0) / vs.length;
  const f = normValue;
  const ticks = NORM_TICKS;
  const tickLabel = String;

  const sInput = stageSection(body, true, 'x: one token',
    `The residual stream of one token. Illustrative: ${NORM_FEATURES} features standing in for d_model (${formatInt(model.dModel)}), ` +
    'including two outliers. The sliders shift and scale x: LayerNorm undoes both.');
  const slider = (label: string, min: number, max: number, value: number) => {
    const input = el('input', { type: 'range', min: String(min), max: String(max), step: '0.01', value: String(value) });
    const shown = el('b');
    const row = el('label', { className: 'op-popup-control' });
    row.append(el('span', { textContent: label }), input, shown);
    return { input, shown, row };
  };
  const shift = slider('Shift x by', -5, 5, 0);
  const scale = slider('Scale x by', -2, 2, 0);
  const cX = barChart('x', 'line: mean(x)');
  const fX = el('div', { className: 'op-popup-facts' });
  sInput.append(shift.row, scale.row, cX.root, fX);

  const sCenter = stageSection(body, false, 'x − mean(x)',
    "Subtracts the token's mean, so the features average 0. This is the step RMSNorm leaves out.");
  const cCentered = barChart('x − mean(x)', 'outlines: x');
  const fCenter = el('div', { className: 'op-popup-facts' });
  sCenter.append(cCentered.root, fCenter);

  const sNorm = stageSection(body, false, 'Divide by √(var(x) + ε)',
    'The variance is the mean of the squared centered values; dividing by its square root gives the features a standard ' +
    'deviation of 1. The vector loses two degrees of freedom: its mean and its scale.');
  const cNorm = barChart('(x − mean) / std', 'lines: ±1');
  const fNorm = el('div', { className: 'op-popup-facts' });
  sNorm.append(cNorm.root, fNorm);

  const sAffine = stageSection(body, false, '× weight + bias',
    `A learned scale and a learned shift per feature: ${formatInt(2 * model.dModel)} parameters (2 · d_model; RMSNorm has ` +
    'only the scale). Illustrative values.');
  const cGamma = barChart('weight', '', 'weight');
  const cBeta = barChart('bias', '', 'weight');
  const cOut = barChart('(x − mean) / std · weight + bias', 'outlines: before weight and bias');
  sAffine.append(cGamma.root, cBeta.root, cOut.root);

  const sCost = stageSection(body, false, 'In memory: one kernel',
    'torch.nn.LayerNorm runs as one fused kernel: it reads x and the parameters once and writes the output, keeping the ' +
    'mean and variance in registers.');
  const rows = pass.mode === 'decode' ? 1 : pass.T;
  const tensor = rows * model.dModel * DTYPES[pass.actDtype].bytes;
  const fCost = el('div', { className: 'op-popup-facts' });
  fCost.innerHTML =
    `In this pass x is ${formatInt(rows)} × ${formatInt(model.dModel)} ${DTYPES[pass.actDtype].label} = ` +
    `<b>${formatBytes(tensor)}</b>: the kernel moves about <b>${formatBytes(2 * tensor)}</b> (x in, the output out), ` +
    `twice per layer (ln_1 and ln_2) and once more at the top (ln_f). Llama's RMSNorm, written in PyTorch, passes ` +
    'several fp32 temporaries through memory instead (see the expanded elementwise detail).';
  sCost.append(fCost);

  function update() {
    const b = Number(shift.input.value);
    const s = 10 ** Number(scale.input.value);
    shift.shown.textContent = `${b >= 0 ? '+' : '−'}${Math.abs(b).toFixed(2)}`;
    scale.shown.textContent = `×${s < 1 ? s.toPrecision(2) : s.toFixed(s < 10 ? 2 : 1)}`;
    const x = base.map((v) => v * s + b);
    const mu = mean(x);
    const centered = x.map((v) => v - mu);
    const variance = mean(centered.map((v) => v * v));
    const r = 1 / Math.sqrt(variance + eps);
    const n = centered.map((v) => v * r);
    const out = n.map((v, i) => v * gamma[i] + beta[i]);
    // RMSNorm of the same x, without centering: the shift leaks into its output.
    const rms = Math.sqrt(mean(x.map((v) => v * v)) + eps);
    const baseRms = Math.sqrt(mean(base.map((v) => v * v)));
    const rmsDrift = Math.sqrt(mean(x.map((v, i) => (v / rms - base[i] / baseRms) ** 2)));

    cX.draw({ values: x, ...extentWithZero(x), ticks, tickLabel, refs: [{ value: mu, label: 'mean' }], describe: (i) => `feature ${i} · x = ${f(x[i])}` });
    fX.innerHTML = `mean(x) = <b>${f(mu)}</b>; largest |x| = <b>${f(Math.max(...x.map(Math.abs)))}</b>.`;

    cCentered.draw({
      values: centered, ghost: x, ...extentWithZero([...centered, ...x]), ticks, tickLabel,
      describe: (i) => `feature ${i} · ${f(x[i])} − ${f(mu)} = ${f(centered[i])}`,
    });
    fCenter.innerHTML =
      `Shift x with the slider: the outlines move, the bars do not. RMSNorm divides x itself by its RMS, so a shift ` +
      `changes its output: by <b>${f(rmsDrift)}</b> (RMS of the change) at this setting.`;

    cNorm.draw({
      values: n, ...extentWithZero([...n, 1, -1]), ticks, tickLabel, refs: [{ value: 1, label: '+1' }, { value: -1, label: '−1' }],
      describe: (i) => `feature ${i} · ${f(centered[i])} × ${f(r)} = ${f(n[i])}`,
    });
    fNorm.innerHTML =
      `var(x) = <b>${f(variance)}</b>, ε = ${eps}, so 1/√(var + ε) = <b>${f(r)}</b>. After: mean <b>${Math.abs(mean(n)) < 1e-9 ? 0 : f(mean(n))}</b>, ` +
      `standard deviation <b>${Math.sqrt(mean(n.map((v) => v * v))).toFixed(3)}</b>, whatever the sliders say.`;

    cGamma.draw({ values: gamma, ...extentWithZero(gamma), ticks, tickLabel, describe: (i) => `feature ${i} · weight = ${f(gamma[i])}` });
    cBeta.draw({ values: beta, ...extentWithZero(beta), ticks, tickLabel, describe: (i) => `feature ${i} · bias = ${f(beta[i])}` });
    cOut.draw({
      values: out, ghost: n, ...extentWithZero([...out, ...n]), ticks, tickLabel,
      describe: (i) => `feature ${i} · ${f(n[i])} × ${f(gamma[i])} + ${f(beta[i])} = ${f(out[i])}`,
    });
  }
  shift.input.addEventListener('input', update);
  scale.input.addEventListener('input', update);
  update();
}

interface Draw {
  values: number[];
  /** Horizontal reference lines, labeled at the right (or the left, with `start`). */
  refs?: { value: number; label: string; start?: boolean }[];
  /**
   * Columns standing for values left out, from `at`: drawn as one band with an ellipsis and no bars. Hovering it
   * describes column `at`; a tick on its middle column labels it.
   */
  gap?: { at: number; columns: number };
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
  const W = CHART_WIDTH;
  const H = 150;
  const pad = { l: 40, r: 8, t: 10, b: 18 };
  let caption: HTMLElement | undefined;
  if (title) {
    caption = el('figcaption', { textContent: title });
    if (subtitle) caption.append(el('span', { textContent: subtitle }));
  }
  const { root, plot, tip } = chartFrame({ height: H, caption });

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
      if (gap && i >= gap.at && i < gap.at + gap.columns) {
        if (i > gap.at) return;
        // Values left out: one band over the gap's columns, with an ellipsis on the baseline between two dashed lines.
        const left = x0 + band * gap.at;
        const right = left + band * gap.columns;
        const mid = (left + right) / 2;
        for (const x of [left, right]) {
          plot.append(svg('line', { x1: x, x2: x, y1: pad.t, y2: H - pad.b, class: 'gap-line' }));
        }
        const dots = svg('text', { x: mid, y: y(0) + 4, class: 'gap-mark', 'text-anchor': 'middle' });
        dots.textContent = '…';
        plot.append(dots);
        const hit = svg('rect', { x: left, y: pad.t, width: right - left, height: H - pad.t - pad.b, class: 'hit' });
        tip.attach(hit, () => describe(i), mid);
        plot.append(hit);
        return;
      }
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
      tip.attach(hit, () => describe(i), cx);
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

/** Width of a gap band in a bar chart, in columns: wide enough to notice. */
const GAP_COLUMNS = 3;

/**
 * Columns for a vector too long to draw whole: its first `head` and last `tail` entries around a gap band that stands
 * for the rest. `ticks` labels the columns with real indices, and the gap with how many entries it hides.
 */
function bothEnds(length: number, head: number, tail: number, headTicks: number[]) {
  const cols = head + GAP_COLUMNS + tail;
  const inGap = (i: number) => i >= head && i < head + GAP_COLUMNS;
  /** The vector index a (non-gap) column shows. */
  const index = (i: number) => (i < head ? i : length - (cols - i));
  const hidden = length - head - tail;
  return {
    cols,
    inGap,
    index,
    hidden,
    gap: { at: head, columns: GAP_COLUMNS },
    ticks: [...headTicks, head + Math.floor(GAP_COLUMNS / 2), cols - 1],
    tickLabel: (i: number) => (inGap(i) ? `+${formatInt(hidden)}` : formatInt(index(i))),
    /** The columns' values from a function of the vector index; the gap gets 0 (it draws no bars). */
    values: (f: (index: number) => number) => Array.from({ length: cols }, (_, i) => (inGap(i) ? 0 : f(index(i)))),
    gapText: (what: string) => `${what} ${formatInt(head)}–${formatInt(length - tail - 1)}: ${formatInt(hidden)} not shown`,
  };
}

/** Width of the charts' coordinate system (they scale to the popup's width). */
const CHART_WIDTH = 560;

/**
 * A chart's frame: a figure with an optional caption, its SVG plot of `width` × `height` units, and a tooltip that
 * `tip.attach` shows while the pointer is over a part of the plot.
 */
function chartFrame(o: { height: number; width?: number; label?: string; caption?: HTMLElement; className?: string }) {
  const width = o.width ?? CHART_WIDTH;
  const root = el('figure', { className: o.className ?? 'op-chart' });
  const plot = svg('svg', { viewBox: `0 0 ${width} ${o.height}`, role: 'img', ...(o.label ? { 'aria-label': o.label } : {}) });
  const tipEl = el('div', { className: 'op-chart-tip' });
  tipEl.hidden = true;
  if (o.caption) root.append(o.caption);
  root.append(plot, tipEl);
  const tip = {
    /** Shows `text` at `x` (in plot units) and, if given, at `y` (the tooltip's top, in plot units). */
    show(text: string, x: number, y?: number) {
      tipEl.textContent = text;
      tipEl.hidden = false;
      placeTip(tipEl, x / width);
      if (y !== undefined) tipEl.style.top = `${(y / o.height) * 100}%`;
    },
    hide() {
      tipEl.hidden = true;
    },
    /** Shows `text()` while the pointer is over `target`. */
    attach(target: Element, text: () => string, x: number, y?: number) {
      target.addEventListener('pointerenter', () => tip.show(text(), x, y));
      target.addEventListener('pointerleave', tip.hide);
    },
  };
  return { root, plot, tip };
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
