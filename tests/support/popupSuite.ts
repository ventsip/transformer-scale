// Every popup variant the app can open for a model, with the settings that change its text: rendered, then exercised
// through each of its controls and hovered over. Run per model from tests/ui/popups/, so models run in parallel.
import { afterEach, describe, expect, it } from 'vitest';
import { MODELS, type ModelId } from '../../src/model/config';
import { buildModel, type BuildOptions, type OpSpec } from '../../src/model/tensors';
import { hasViz, openOpPopup, setPopupContext } from '../../src/ui/opPopup';
import { baseOptions } from './matrix';

export interface PopupCase {
  name: string;
  op: OpSpec;
  options: BuildOptions;
}

/** One op per distinct popup tag and pass. */
export function popupCases(id: ModelId): PopupCase[] {
  const cfg = MODELS[id];
  const cases: PopupCase[] = [];
  const seen = new Set<string>();
  for (const mode of ['prefill', 'decode'] as const)
    for (const attention of ['fused', 'eager'] as const)
      for (const kvCache of ['dynamic', 'static'] as const)
        for (const elementwise of ['compact', 'expanded'] as const)
          for (const mlaCache of cfg.mla ? (['hf', 'latent'] as const) : (['hf'] as const)) {
            const options = baseOptions(cfg, { mode, attention, kvCache, elementwise, mlaCache, T: 700, cacheCapacity: 4096 });
            for (const op of buildModel(cfg, options).ops) {
              if (!op.viz) continue;
              const key = `${JSON.stringify(op.viz)} ${mode} ${attention} ${kvCache} ${mlaCache}`;
              if (seen.has(key)) continue;
              seen.add(key);
              cases.push({ name: `${op.label} ${JSON.stringify(op.viz)} ${mode} ${attention} ${kvCache}${cfg.mla ? ` ${mlaCache}` : ''}`, op, options });
            }
          }
  return cases;
}

/** Text that only shows up when a computation went wrong. */
const BROKEN = /NaN|undefined|\[object|Infinity(?! tokens)|null/;
const GEOMETRY = ['x', 'y', 'width', 'height', 'cx', 'cy', 'r', 'x1', 'y1', 'x2', 'y2', 'd', 'points'];

/** No broken text in `root`, and no broken numbers in its SVG geometry. */
export function expectSound(root: Element, where: string) {
  const text = root.textContent ?? '';
  const at = text.search(BROKEN);
  expect(at < 0 ? undefined : text.slice(Math.max(0, at - 60), at + 20), where).toBeUndefined();
  for (const node of root.querySelectorAll(GEOMETRY.map((a) => `[${a}]`).join(','))) {
    for (const attr of GEOMETRY) {
      const v = node.getAttribute(attr);
      if (v !== null) expect(v, `${where}: <${node.tagName} ${attr}="${v}">`).not.toMatch(/NaN|Infinity|undefined/);
    }
  }
}

/** Opens the popup of a case, with the pass context the app would set. */
export function open({ op, options }: PopupCase, id: ModelId): Element {
  setPopupContext(MODELS[id], { ...options, mlaCache: options.mlaCache ?? 'hf' });
  expect(hasViz(op)).toBe(true);
  openOpPopup(op);
  const popup = document.querySelector('.op-popup');
  expect(popup).not.toBeNull();
  return popup!;
}

/** Points at everything that reacts to the pointer: chart crosshairs, bars and cells, and clicks heatmap rows. */
function hoverAll(popup: Element) {
  for (const hit of popup.querySelectorAll('.hit-area')) {
    for (const x of [5, 280, 555]) {
      hit.dispatchEvent(new MouseEvent('pointermove', { clientX: x, clientY: 50 }));
      expectSound(popup, `crosshair at ${x}`);
    }
    hit.dispatchEvent(new MouseEvent('pointerleave'));
  }
  // Bars, columns and heatmap cells show their values on pointerenter; heatmap cells also select a row on click.
  const targets = [...popup.querySelectorAll('svg *')];
  for (const t of targets) t.dispatchEvent(new MouseEvent('pointerenter'));
  for (const t of targets) t.dispatchEvent(new MouseEvent('pointerleave'));
  expectSound(popup, 'hovered');
  for (const t of targets.filter((_, i) => i % 7 === 0)) t.dispatchEvent(new MouseEvent('click', { bubbles: true }));
  expectSound(popup, 'clicked');
}

export function popupSuite(id: ModelId) {
  afterEach(() => document.body.replaceChildren());
  const cases = popupCases(id);

  describe(`${id} popups`, () => {
    it.each(cases)('$name', (c) => {
      const popup = open(c, id);
      expect(popup.querySelector('h2')!.textContent!.length).toBeGreaterThan(5);
      expect(popup.querySelectorAll('section.op-stage').length).toBeGreaterThan(0);
      // The clicked op's stage is highlighted.
      expect(popup.querySelectorAll('section.op-stage.focus').length).toBeGreaterThan(0);
      expectSound(popup, 'opened');

      // Every control: buttons clicked, sliders to both ends, checkboxes toggled both ways.
      for (const button of popup.querySelectorAll<HTMLButtonElement>('.op-popup-body button')) {
        button.click();
        expectSound(popup, `after "${button.textContent}"`);
      }
      for (const range of popup.querySelectorAll<HTMLInputElement>('input[type=range]')) {
        for (const v of [range.min, range.max, range.defaultValue]) {
          range.value = v;
          range.dispatchEvent(new Event('input'));
          expectSound(popup, `slider at ${v}`);
        }
      }
      for (const box of popup.querySelectorAll<HTMLInputElement>('input[type=checkbox]')) {
        for (let i = 0; i < 2; i++) {
          box.checked = !box.checked;
          box.dispatchEvent(new Event('change'));
          expectSound(popup, `checkbox ${box.checked}`);
        }
      }

      // Esc closes it.
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
      expect(document.querySelector('.op-popup')).toBeNull();
    });

    // Hovering costs more: once per panel and stage.
    const byStage = new Map<string, PopupCase>();
    for (const c of cases) {
      const { panel, stage } = c.op.viz as { panel: string; stage?: string };
      byStage.set(`${panel} ${stage ?? ''}`, byStage.get(`${panel} ${stage ?? ''}`) ?? c);
    }
    it.each([...byStage.values()])('hover: $name', (c) => hoverAll(open(c, id)));
  });
}
