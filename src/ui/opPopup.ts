import type { OpSpec } from '../model/tensors';

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

/** The interactive view of an op: its title and how to fill the popup body; null for ops without one. */
function vizFor(op: OpSpec): { title: string; render: (body: HTMLElement) => void } | null {
  const sampling = samplingStage(op);
  if (sampling) return { title: 'Sampling: from logits to the next token', render: (body) => samplingViz(body, sampling) };
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

/** Vocabulary size of the current model. */
let VOCAB = 128256;

export function setVocabSize(n: number): void {
  VOCAB = n;
}
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
  const logits = new Float64Array(VOCAB);
  for (let r = 0; r < VOCAB; r++) {
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
  const sLogits = section('logits', 'logits (fp32)', `The LM head scores every token in the vocabulary. Illustrative values; the top ${SHOWN} of ${VOCAB.toLocaleString('en-US')} shown, by rank.`);
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
      `Top-k keeps <b>${int(afterK)}</b> of ${int(VOCAB)} tokens; top-p then keeps <b>${int(kept)}</b>` +
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
  const W = 560;
  const H = 220;
  const pad = { l: 40, r: 12, t: 10, b: 22 };
  const [xLo, xHi, yLo, yHi] = [-6, 6, -1, 6];
  const px = (x: number) => pad.l + ((x - xLo) / (xHi - xLo)) * (W - pad.l - pad.r);
  const py = (y: number) => pad.t + ((yHi - y) / (yHi - yLo)) * (H - pad.t - pad.b);
  const root = el('figure', { className: 'op-chart' });
  const caption = el('figcaption');
  caption.innerHTML = '<span class="op-legend"><i class="swatch-line"></i>SiLU(x)</span>' +
    '<span class="op-legend"><i class="swatch-ref"></i>ReLU(x) = max(0, x), for reference</span>';
  const plot = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'SiLU and ReLU from x = −6 to 6' });
  const tip = el('div', { className: 'op-chart-tip' });
  tip.hidden = true;
  root.append(caption, plot, tip);

  for (let y = yLo; y <= yHi; y++) {
    plot.append(svg('line', { x1: pad.l, x2: W - pad.r, y1: py(y), y2: py(y), class: y === 0 ? 'axis' : 'grid' }));
    const label = svg('text', { x: pad.l - 6, y: py(y) + 3, class: 'tick', 'text-anchor': 'end' });
    label.textContent = String(y);
    plot.append(label);
  }
  for (let x = xLo; x <= xHi; x += 2) {
    if (x === 0) plot.append(svg('line', { x1: px(0), x2: px(0), y1: py(yLo), y2: py(yHi), class: 'axis' }));
    const label = svg('text', { x: px(x), y: H - 6, class: 'tick', 'text-anchor': 'middle' });
    label.textContent = String(x);
    plot.append(label);
  }
  const path = (f: (x: number) => number) => {
    let d = '';
    for (let i = 0; i <= 240; i++) {
      const x = xLo + ((xHi - xLo) * i) / 240;
      d += `${i ? 'L' : 'M'}${px(x).toFixed(1)},${py(f(x)).toFixed(1)}`;
    }
    return d;
  };
  plot.append(svg('path', { d: path((x) => Math.max(0, x)), class: 'ref' }));
  plot.append(svg('path', { d: path(silu), class: 'line' }));
  const minY = silu(SILU_MIN_X);
  plot.append(svg('circle', { cx: px(SILU_MIN_X), cy: py(minY), r: 4, class: 'marker' }));
  const minLabel = svg('text', { x: px(SILU_MIN_X), y: py(minY) + 16, class: 'tick', 'text-anchor': 'middle' });
  minLabel.textContent = `min ${minY.toFixed(3)}`;
  plot.append(minLabel);

  // Crosshair: follows the pointer along x.
  const cross = svg('line', { y1: py(yHi), y2: py(yLo), class: 'crosshair', visibility: 'hidden' });
  const dot = svg('circle', { r: 4, class: 'marker', visibility: 'hidden' });
  const hit = svg('rect', { x: pad.l, y: pad.t, width: W - pad.l - pad.r, height: H - pad.t - pad.b, class: 'hit-area' });
  plot.append(cross, dot, hit);
  hit.addEventListener('pointermove', (e) => {
    const box = plot.getBoundingClientRect();
    const sx = ((e.clientX - box.left) / box.width) * W;
    const x = Math.max(xLo, Math.min(xHi, xLo + ((sx - pad.l) / (W - pad.l - pad.r)) * (xHi - xLo)));
    for (const node of [cross, dot]) node.setAttribute('visibility', 'visible');
    cross.setAttribute('x1', String(px(x)));
    cross.setAttribute('x2', String(px(x)));
    dot.setAttribute('cx', String(px(x)));
    dot.setAttribute('cy', String(py(silu(x))));
    tip.textContent = `x = ${x.toFixed(2)} · σ(x) = ${sigmoid(x).toFixed(3)} · SiLU = ${silu(x).toFixed(3)} · ReLU = ${Math.max(0, x).toFixed(2)}`;
    tip.hidden = false;
    tip.style.left = `${(px(x) / W) * 100}%`;
  });
  hit.addEventListener('pointerleave', () => {
    for (const node of [cross, dot]) node.setAttribute('visibility', 'hidden');
    tip.hidden = true;
  });
  return root;
}

interface Draw {
  values: number[];
  /** Label of the column ticks (default: rank, #1). */
  tickLabel?: (i: number) => string;
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

/** A single-series column chart in SVG, with optional outlines, removed columns, a marked column and hover tooltips. */
function barChart(title?: string, subtitle = '') {
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

  function draw({ values, lo, hi, describe, ghost, removed, picked, percent, tickLabel = (i) => `#${i + 1}`, ticks = [0, 9, 19, 29, 39] }: Draw) {
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
      label.textContent = percent ? `${Math.round(v * 1000) / 10}%` : String(Math.round(v * 10) / 10);
      plot.append(label);
    }

    values.forEach((v, i) => {
      const cx = x0 + band * (i + 0.5);
      const cls = removed?.[i] ? 'ghost' : i === picked ? 'bar picked' : 'bar';
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
