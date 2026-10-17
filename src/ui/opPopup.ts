import type { OpSpec } from '../model/tensors';

/** An interactive explanation of one operation, rendered into the popup body. */
type OpViz = (body: HTMLElement, op: OpSpec) => void;

const VIZ: Record<string, OpViz> = {
  '÷ temperature': temperatureViz,
};

export const hasViz = (op: OpSpec) => op.label in VIZ;

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
  const viz = VIZ[op.label];
  if (!viz) return;
  document.querySelector('.op-popup-backdrop')?.remove();
  const backdrop = el('div', { className: 'op-popup-backdrop' });
  const popup = el('div', { className: 'op-popup' });
  const head = el('div', { className: 'op-popup-head' });
  const close = el('button', { className: 'op-popup-close', textContent: '×', title: 'Close (Esc)' });
  head.append(el('h2', { textContent: op.label }), close);
  const body = el('div', { className: 'op-popup-body' });
  popup.append(head, el('p', { className: 'op-popup-note', textContent: op.note }), body);
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
  viz(body, op);
}

// ---------------------------------------------------------------------------------------------
// ÷ temperature

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
 * Illustrative last-position logits: a few strong candidates falling off roughly with log(rank),
 * then a long tail of the remaining vocabulary around zero. Sorted descending.
 */
function illustrativeLogits(): Float64Array {
  const rand = rng(7);
  const normal = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const logits = new Float64Array(VOCAB);
  for (let r = 0; r < VOCAB; r++) {
    logits[r] = r < 400 ? 18 - 2 * Math.log(r + 1) + 0.4 * normal() : 1.5 * normal();
  }
  return logits.sort().reverse();
}

/** Probability facts about softmax(logits / T) over the whole vocabulary. */
function softmaxFacts(logits: Float64Array, t: number) {
  const max = logits[0] / t;
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] / t - max);
  const p = (i: number) => Math.exp(logits[i] / t - max) / sum;
  let shown = 0;
  for (let i = 0; i < SHOWN; i++) shown += p(i);
  let cum = 0;
  let for90 = 0;
  while (cum < 0.9 && for90 < logits.length) cum += p(for90++);
  return { top: p(0), shown, for90 };
}

function temperatureViz(body: HTMLElement): void {
  const logits = illustrativeLogits();
  const input = Array.from(logits.subarray(0, SHOWN));

  const slider = el('input', { type: 'range', min: '0.05', max: '2', step: '0.05', value: '0.7' });
  const tValue = el('b');
  const control = el('label', { className: 'op-popup-control' });
  control.append(el('span', { textContent: 'Temperature' }), slider, tValue);

  const top = barChart('Input: logits', 'Top 40 of 128,256 tokens, by rank (illustrative values)');
  const bottom = barChart('Output: logits ÷ T', 'Outlines: the input logits, on the same scale');
  const facts = el('div', { className: 'op-popup-facts' });
  body.append(top.root, control, bottom.root, facts);

  const update = () => {
    const t = Number(slider.value);
    tValue.textContent = `T = ${t.toFixed(2)}`;
    const output = input.map((v) => v / t);
    // The input keeps its own fixed scale; the output scale fits both the output and the input outlines.
    top.draw(input, null, Math.min(0, ...input), Math.max(...input), (i) => `#${i + 1} · logit ${input[i].toFixed(2)}`);
    bottom.draw(output, input, Math.min(0, ...input, ...output), Math.max(...input, ...output), (i) => `#${i + 1} · ${input[i].toFixed(2)} ÷ ${t.toFixed(2)} = ${output[i].toFixed(2)}`);
    const f = softmaxFacts(logits, t);
    const pct = (x: number) => `${(100 * x).toPrecision(x < 0.1 ? 2 : 3)}%`;
    facts.innerHTML =
      `After the softmax that follows: the top token gets <b>${pct(f.top)}</b>, the 40 shown hold <b>${pct(f.shown)}</b>, ` +
      `and <b>${f.for90.toLocaleString('en-US')}</b> ${f.for90 === 1 ? 'token makes' : 'tokens make'} up 90% of the probability. ` +
      (t < 1 ? 'T < 1 stretches the gaps between logits: sharper, more predictable.' : t > 1 ? 'T > 1 shrinks the gaps: flatter, more random.' : 'T = 1 leaves the logits unchanged.');
  };
  slider.addEventListener('input', update);
  update();
}

/** A single-series column chart drawn in SVG, with optional outline "ghost" columns and hover tooltips. */
function barChart(title: string, subtitle: string) {
  const W = 560;
  const H = 170;
  const pad = { l: 40, r: 8, t: 8, b: 18 };
  const root = el('figure', { className: 'op-chart' });
  const caption = el('figcaption');
  caption.append(el('b', { textContent: title }), el('span', { textContent: subtitle }));
  const plot = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': title });
  const tip = el('div', { className: 'op-chart-tip' });
  tip.hidden = true;
  root.append(caption, plot, tip);

  function draw(values: number[], ghost: number[] | null, lo: number, hi: number, describe: (i: number) => string) {
    plot.replaceChildren();
    const x0 = pad.l;
    const band = (W - pad.l - pad.r) / values.length;
    const barW = Math.min(24, band - 2);
    const y = (v: number) => pad.t + ((hi - v) / (hi - lo)) * (H - pad.t - pad.b);

    // Recessive grid with clean ticks.
    const step = niceStep((hi - lo) / 4);
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
      plot.append(svg('line', { x1: x0, x2: W - pad.r, y1: y(v), y2: y(v), class: v === 0 ? 'axis' : 'grid' }));
      const label = svg('text', { x: x0 - 6, y: y(v) + 3, class: 'tick', 'text-anchor': 'end' });
      label.textContent = String(Math.round(v * 10) / 10);
      plot.append(label);
    }

    values.forEach((v, i) => {
      const cx = x0 + band * (i + 0.5);
      plot.append(column(cx, barW, y(0), y(v), 'bar'));
      // Outline on top, so it shows whether the output bar is taller or shorter.
      if (ghost) plot.append(column(cx, barW, y(0), y(ghost[i]), 'ghost'));
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
