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

export const hasViz = (op: OpSpec) => samplingStage(op) !== null;

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
  const stage = samplingStage(op);
  if (!stage) return;
  document.querySelector('.op-popup-backdrop')?.remove();
  const backdrop = el('div', { className: 'op-popup-backdrop' });
  const popup = el('div', { className: 'op-popup' });
  const head = el('div', { className: 'op-popup-head' });
  const close = el('button', { className: 'op-popup-close', textContent: '×', title: 'Close (Esc)' });
  head.append(el('h2', { textContent: 'Sampling: from logits to the next token' }), close);
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
  samplingViz(body, stage);
}

// ---------------------------------------------------------------------------------------------
// Sampling: logits → ÷ temperature → top-k / top-p → softmax → sample

const VOCAB = 128256;
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
    const s = el('section', { className: `op-stage${stage === focus ? ' focus' : ''}` });
    s.append(el('h3', { textContent: title }), el('p', { className: 'op-popup-note', textContent: text }));
    sections.set(stage, s);
    body.append(s);
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
  const sLogits = section('logits', 'logits (fp32)', 'The LM head scores every token in the vocabulary. Illustrative values; the top 40 of 128,256 shown, by rank.');
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

interface Draw {
  values: number[];
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
function barChart() {
  const W = 560;
  const H = 150;
  const pad = { l: 40, r: 8, t: 10, b: 18 };
  const root = el('figure', { className: 'op-chart' });
  const plot = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
  const tip = el('div', { className: 'op-chart-tip' });
  tip.hidden = true;
  root.append(plot, tip);

  function draw({ values, lo, hi, describe, ghost, removed, picked, percent }: Draw) {
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

    for (const i of [0, 9, 19, 29, 39].filter((i) => i < values.length)) {
      const label = svg('text', { x: x0 + band * (i + 0.5), y: H - 4, class: 'tick', 'text-anchor': 'middle' });
      label.textContent = `#${i + 1}`;
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
