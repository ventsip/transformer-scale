// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { MODELS } from '../../src/model/config';
import { buildModel, type OpSpec } from '../../src/model/tensors';
import { hasViz, openOpPopup, setPopupContext } from '../../src/ui/opPopup';
import { MODEL_IDS, baseOptions } from '../support/matrix';
import { popupCases } from '../support/popupSuite';

afterEach(() => document.body.replaceChildren());

describe('popup coverage', () => {
  it('every panel is opened by some model', () => {
    const panels = new Set(MODEL_IDS.flatMap((id) => popupCases(id).map((c) => c.op.viz!.panel)));
    expect([...panels].sort()).toEqual(['attention', 'gelu', 'kvcache', 'layernorm', 'lookup', 'mla', 'moe', 'rmsnorm', 'rope', 'sampling', 'swiglu']);
  });
});

describe('popup behavior', () => {
  const cfg = MODELS['gemma-3-1b'];
  const options = baseOptions(cfg, { attention: 'eager', T: 700 });
  const ops = buildModel(cfg, options).ops;
  const first = (pred: (o: OpSpec) => boolean) => ops.find(pred)!;
  const openFirst = (pred: (o: OpSpec) => boolean) => {
    setPopupContext(cfg, { ...options, mlaCache: 'hf' });
    openOpPopup(first(pred));
  };

  it('ignores ops without a popup', () => {
    const add = first((o) => o.label === '+' && !o.viz);
    expect(hasViz(add)).toBe(false);
    openOpPopup(add);
    expect(document.querySelector('.op-popup')).toBeNull();
  });

  it('replaces an open popup, and closes from its button or the backdrop', () => {
    openFirst((o) => o.viz?.panel === 'lookup');
    openFirst((o) => o.viz?.panel === 'sampling');
    expect(document.querySelectorAll('.op-popup')).toHaveLength(1);
    document.querySelector<HTMLButtonElement>('.op-popup-close')!.click();
    expect(document.querySelector('.op-popup')).toBeNull();
    openFirst((o) => o.viz?.panel === 'lookup');
    // A click inside the popup keeps it; one on the backdrop closes it.
    document.querySelector('.op-popup')!.dispatchEvent(new Event('pointerdown', { bubbles: true }));
    expect(document.querySelector('.op-popup')).not.toBeNull();
    document.querySelector('.op-popup-backdrop')!.dispatchEvent(new Event('pointerdown', { bubbles: true }));
    expect(document.querySelector('.op-popup')).toBeNull();
  });

  it('titles Gemma variants', () => {
    const title = (pred: (o: OpSpec) => boolean) => {
      openFirst(pred);
      return document.querySelector('.op-popup h2')!.textContent;
    };
    expect(title((o) => o.viz?.panel === 'swiglu')).toBe('GeGLU: a = GELU(g) ⊙ u');
    expect(title((o) => o.viz?.panel === 'rmsnorm' && !o.viz.width)).toBe('RMSNorm: x / RMS(x) · (1 + weight)');
    expect(title((o) => o.viz?.panel === 'rmsnorm' && !!o.viz.width)).toBe('RMSNorm per head: x / RMS(x) · (1 + weight)');
    expect(title((o) => o.viz?.panel === 'rope' && o.viz.window !== undefined)).toContain('sliding-window layers');
    expect(title((o) => o.viz?.panel === 'attention' && o.viz.window !== undefined)).toContain('sliding window');
  });

  it('opens a sliding layer\'s attention with the window mask on, a global layer\'s with it off', () => {
    const windowBox = () => document.querySelectorAll<HTMLInputElement>('.op-popup input[type=checkbox]')[1];
    openFirst((o) => o.viz?.panel === 'attention' && o.viz.window !== undefined);
    expect(windowBox().checked).toBe(true);
    openFirst((o) => o.viz?.panel === 'attention' && o.floor === 6);
    expect(windowBox().checked).toBe(false);
  });

  it('quotes the real sizes of the pass', () => {
    // Cache ops exist in decode (torch.cat) or with a static cache.
    const decode = baseOptions(cfg, { mode: 'decode', T: 700 });
    setPopupContext(cfg, { ...decode, mlaCache: 'hf' });
    openOpPopup(buildModel(cfg, decode).ops.find((o) => o.viz?.panel === 'kvcache')!);
    const text = document.querySelector('.op-popup')!.textContent!;
    expect(text).toContain('22 of 26 layers attend within a 512-token sliding window');
    expect(text).toContain('26.0 KiB across 26 layers');
  });
});
