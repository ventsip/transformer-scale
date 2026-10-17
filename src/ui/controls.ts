import type { ThicknessMode } from '../layout';
import type { DType } from '../model/dtypes';
import { formatInt } from '../format';

export interface Settings {
  T: number;
  focusFloor: number;
  activations: 'focused' | 'all';
  showScores: boolean;
  showFlow: boolean;
  thickness: ThicknessMode;
  weightDtype: DType;
  actDtype: DType;
  kvDtype: DType;
  floorSpacing: number;
}

export interface ControlsOptions {
  maxT: number;
  floorCount: number;
  floorName: (floor: number) => string;
  onChange: (key: keyof Settings) => void;
  onOverview: () => void;
  onFrameFocus: () => void;
}

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, props: object = {}): HTMLElementTagNameMap[K] =>
  Object.assign(document.createElement(tag), props);

/** Builds the settings panel; mutates `s` in place and reports which key changed. */
export function createControls(root: HTMLElement, s: Settings, o: ControlsOptions) {
  const refreshers: (() => void)[] = [];

  const row = (label: string, input: HTMLElement, value?: () => string) => {
    const wrap = el('label', { className: 'row' });
    const head = el('span', { className: 'row-head' });
    head.append(el('span', { textContent: label }));
    if (value) {
      const v = el('span', { className: 'row-value' });
      head.append(v);
      refreshers.push(() => (v.textContent = value()));
    }
    wrap.append(head, input);
    root.append(wrap);
  };

  const range = (min: number, max: number, step: number, get: () => number, set: (v: number) => void, key: keyof Settings) => {
    const input = el('input', { type: 'range', min: String(min), max: String(max), step: String(step) });
    refreshers.push(() => (input.value = String(get())));
    input.addEventListener('input', () => {
      set(Number(input.value));
      refresh();
      o.onChange(key);
    });
    return input;
  };

  const select = <T extends string>(key: keyof Settings, options: [T, string][]) => {
    const input = el('select');
    for (const [value, text] of options) input.append(el('option', { value, textContent: text }));
    refreshers.push(() => (input.value = String(s[key])));
    input.addEventListener('change', () => {
      (s as unknown as Record<string, unknown>)[key] = input.value;
      o.onChange(key);
    });
    return input;
  };

  // Context length on a log scale: slider value is 100 × log2(T).
  const maxLog = Math.round(Math.log2(o.maxT) * 100);
  row('Prompt length (prefill)', range(0, maxLog, 1, () => Math.round(Math.log2(s.T) * 100), (v) => {
    s.T = Math.min(o.maxT, Math.max(1, Math.round(2 ** (v / 100))));
  }, 'T'), () => `${formatInt(s.T)} tokens`);

  row('Focused floor', range(0, o.floorCount - 1, 1, () => s.focusFloor, (v) => (s.focusFloor = v), 'focusFloor'),
    () => o.floorName(s.focusFloor));

  row('Transient activations', select('activations', [['focused', 'Focused floor only (inference)'], ['all', 'Every floor']]));

  const checkbox = (key: 'showScores' | 'showFlow', text: string) => {
    const input = el('input', { type: 'checkbox' });
    refreshers.push(() => (input.checked = s[key]));
    input.addEventListener('change', () => {
      s[key] = input.checked;
      o.onChange(key);
    });
    const wrap = el('label', { className: 'row check' });
    wrap.append(input, el('span', { textContent: text }));
    root.append(wrap);
  };
  checkbox('showFlow', 'Show data flow on the focused floor');
  checkbox('showScores', 'Show materialized attention scores');

  row('Thickness', select('thickness', [['bytes', 'Bytes per element'], ['elements', 'Uniform (elements only)']]));
  row('Weights dtype', select('weightDtype', [['bf16', 'bf16'], ['fp8', 'fp8'], ['int4', 'int4'], ['fp32', 'fp32']]));
  row('Activations dtype', select('actDtype', [['bf16', 'bf16'], ['fp32', 'fp32']]));
  row('KV cache dtype', select('kvDtype', [['bf16', 'bf16'], ['fp8', 'fp8']]));

  row('Floor spacing', range(1000, 40000, 500, () => s.floorSpacing, (v) => (s.floorSpacing = v), 'floorSpacing'),
    () => formatInt(s.floorSpacing));

  const buttons = el('div', { className: 'buttons' });
  const overview = el('button', { textContent: 'Overview' });
  overview.addEventListener('click', o.onOverview);
  const frame = el('button', { textContent: 'Frame focused floor (F)' });
  frame.addEventListener('click', o.onFrameFocus);
  buttons.append(overview, frame);
  root.append(buttons);

  function refresh() {
    for (const r of refreshers) r();
  }
  refresh();
  return { refresh };
}
