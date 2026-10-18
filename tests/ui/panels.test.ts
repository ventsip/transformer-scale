// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { formatCount } from '../../src/format';
import { layoutTensors } from '../../src/layout';
import { MODELS, type ModelId } from '../../src/model/config';
import { floorParts } from '../../src/model/sections';
import { summarize } from '../../src/model/stats';
import { buildModel, floorCount, floorName, keptEntries, layerFloor } from '../../src/model/tensors';
import type { Pick } from '../../src/scene/tensorScene';
import { renderLegend, renderStats } from '../../src/statsPanel';
import type { Edge } from '../../src/scene/flowScene';
import { applyPalette } from '../../src/palette';
import { Tooltips } from '../../src/tooltips';
import { createControls, maxTokens, type Settings } from '../../src/ui/controls';
import { MODEL_IDS, baseOptions } from '../support/matrix';

afterEach(() => document.body.replaceChildren());

const settingsFor = (id: ModelId, overrides: Partial<Settings> = {}): Settings => {
  const cfg = MODELS[id];
  return {
    ...baseOptions(cfg), model: id, focusFloor: 1, activations: 'focused', mlaCache: 'hf', showFlow: true, thickness: 'bytes',
    floorSpacing: 8000, playSpeed: '4', ...overrides,
  } as Settings;
};

const rows = (el: HTMLElement) => Object.fromEntries([...el.querySelectorAll('.stat')].map((r) => [r.querySelector('span')!.textContent, r.querySelector('b')!.textContent]));

describe('stats panel', () => {
  it.each(MODEL_IDS)('%s', (id) => {
    const cfg = MODELS[id];
    for (const s of [settingsFor(id), settingsFor(id, { mode: 'decode', kvCache: 'static', attention: 'eager' })]) {
      const model = buildModel(cfg, { ...s, cacheCapacity: cfg.maxContext });
      const el = document.createElement('div');
      renderStats(el, { cfg, settings: s, stats: summarize(model, cfg), expertTokens: model.expertTokens, selection: null, expertSelection: null });
      const r = rows(el);
      expect(r.Parameters).toMatch(/^[\d.]+[MB] · [\d.,]+ [GM]iB$/);
      expect(el.textContent).not.toMatch(/NaN|undefined|Infinity/);
      if (cfg.moe) expect(Object.keys(r).some((k) => k.startsWith('Active per token'))).toBe(true);
      if (s.attention === 'eager') expect(r['Eager score tensors, one layer']).toBeDefined();
      if (s.kvCache === 'static') expect(r['KV cache filled']).toBe(`301 of ${cfg.maxContext.toLocaleString('en-US')} slots`);
    }
  });

  it('describes a selected head and expert', () => {
    const cfg = MODELS['mixtral-8x7b'];
    const s = settingsFor('mixtral-8x7b');
    const model = buildModel(cfg, { ...s });
    const el = document.createElement('div');
    renderStats(el, { cfg, settings: s, stats: summarize(model, cfg), expertTokens: model.expertTokens, selection: { group: 2, head: 9 }, expertSelection: 5 });
    expect(rows(el)['Selected head (Esc clears)']).toBe('Q head 9 · KV group 2');
    expect(rows(el)['Selected expert (Esc clears)']).toBe('E5');
  });

  it('summarizes many experts instead of listing them (DeepSeek)', () => {
    const cfg = MODELS['deepseek-v2-lite'];
    const s = settingsFor('deepseek-v2-lite', { focusFloor: layerFloor(3) });
    const model = buildModel(cfg, { ...s });
    const el = document.createElement('div');
    renderStats(el, { cfg, settings: s, stats: summarize(model, cfg), expertTokens: model.expertTokens, selection: null, expertSelection: null });
    expect(rows(el)['Tokens per expert, Layer 3']).toMatch(/^\d+–\d+, mean \d+/);
  });

  it('renders the legend', () => {
    const el = document.createElement('div');
    renderLegend(el);
    expect(el.querySelectorAll('i').length).toBeGreaterThanOrEqual(10);
    expect(el.textContent).toContain('1 unit = 1 element');
  });
});

describe('controls', () => {
  const make = (id: ModelId, overrides: Partial<Settings> = {}) => {
    const root = document.createElement('div');
    const s = settingsFor(id, overrides);
    const cfg = MODELS[id];
    const onChange = vi.fn();
    const onPlay = vi.fn();
    createControls(root, s, {
      maxT: cfg.maxContext, mla: !!cfg.mla, floorCount: floorCount(cfg), floorName: (f) => floorName(cfg, f),
      onChange, onOverview: vi.fn(), onFrameFocus: vi.fn(), onPlay, isPlaying: () => false,
    });
    const select = (label: string) => [...root.querySelectorAll('label.row')]
      .find((r) => r.querySelector('.row-head span')?.textContent === label)?.querySelector('select, input') as HTMLSelectElement | HTMLInputElement | undefined;
    return { root, s, onChange, onPlay, select };
  };

  it('lists every model and shows the MLA setting only for MLA models', () => {
    const { select } = make('llama-3-8b');
    expect([...(select('Model') as HTMLSelectElement).options].map((o) => o.value)).toEqual(MODEL_IDS);
    expect(select('MLA cache')).toBeUndefined();
    expect(make('deepseek-v2-lite').select('MLA cache')).toBeDefined();
  });

  it('writes a changed select into the settings and reports the key', () => {
    const { s, onChange, select } = make('llama-3-8b');
    const attention = select('Attention') as HTMLSelectElement;
    attention.value = 'eager';
    attention.dispatchEvent(new Event('change'));
    expect(s.attention).toBe('eager');
    expect(onChange).toHaveBeenCalledWith('attention');
  });

  it('maps the token slider logarithmically and keeps room for the new token in decode', () => {
    const { s, select, root } = make('gpt2', { mode: 'decode' });
    const tokens = select('Tokens') as HTMLInputElement;
    tokens.value = tokens.max;
    tokens.dispatchEvent(new Event('input'));
    expect(s.T).toBe(1023);
    expect(maxTokens(s, 1024)).toBe(1023);
    expect(root.textContent).toContain('1,023 in cache, +1 new');
    tokens.value = '0';
    tokens.dispatchEvent(new Event('input'));
    expect(s.T).toBe(1);
  });

  it('plays', () => {
    const { root, onPlay } = make('gpt2');
    root.querySelector<HTMLButtonElement>('button.play')!.click();
    expect(onPlay).toHaveBeenCalledOnce();
  });
});

describe('sections', () => {
  it('names Gemma\'s sliding and global attention', () => {
    const cfg = MODELS['gemma-3-1b'];
    const model = buildModel(cfg, baseOptions(cfg));
    const [sliding] = floorParts(cfg, model, layerFloor(0));
    const [global] = floorParts(cfg, model, layerFloor(5));
    expect(sliding.name).toBe('Sliding-window attention');
    expect(sliding.note).toContain('512');
    expect(global.name).toBe('Global attention');
  });

  it('splits a mixture-of-experts layer into router, shared experts, experts and each expert', () => {
    const cfg = MODELS['deepseek-v2-lite'];
    const model = buildModel(cfg, baseOptions(cfg));
    expect(floorParts(cfg, model, layerFloor(0)).map((p) => p.name)).toEqual(['Attention', 'Dense MLP']);
    const parts = floorParts(cfg, model, layerFloor(1));
    expect(parts.slice(0, 4).map((p) => p.kind)).toEqual(['attention', 'router', 'shared', 'experts']);
    expect(parts.filter((p) => p.kind === 'expert')).toHaveLength(64);
    // The router's tray encloses the shared experts; its totals leave them out.
    const router = parts.find((p) => p.kind === 'router')!;
    const shared = parts.find((p) => p.kind === 'shared')!;
    expect(router.extentIds!.length).toBe(router.tensorIds.length + shared.tensorIds.length);
  });

  it('adds up to the layer: the parts\' parameters are the floor\'s', () => {
    for (const id of MODEL_IDS) {
      const cfg = MODELS[id];
      const model = buildModel(cfg, baseOptions(cfg));
      const floor = layerFloor(cfg.nLayers - 1);
      const parts = floorParts(cfg, model, floor).filter((p) => p.kind !== 'expert');
      const onFloor = model.tensors.filter((s) => s.floor === floor && s.kind === 'weight').reduce((a, s) => a + s.x * s.z * s.tiles, 0);
      expect(parts.reduce((a, p) => a + p.params, 0), id).toBe(onFloor);
    }
  });

  it('has no parts off the layer floors', () => {
    const cfg = MODELS['gpt2'];
    expect(floorParts(cfg, buildModel(cfg, baseOptions(cfg)), 0)).toEqual([]);
  });
});

describe('tooltips', () => {
  const cfg = MODELS['gemma-3-1b'];
  const T = 2048;
  const model = buildModel(cfg, baseOptions(cfg, { T, attention: 'eager' }));
  const stats = summarize(model, cfg);
  const placed = layoutTensors(model.tensors, { floorSpacing: 8000, thickness: 'bytes', focusFloor: 1 }).placed;
  const el = document.createElement('div');
  const tips = new Tooltips(el, () => ({ cfg, mode: 'prefill', T, focusFloor: 1, stats, parts: floorParts(cfg, model, 1), expertTokens: undefined }));
  const event = new MouseEvent('pointermove', { clientX: 10, clientY: 10 }) as PointerEvent;
  const pickOf = (id: string, part: Pick['part']): Pick => {
    const item = placed.find((p) => p.spec.id === id)!;
    return { item, box: item.boxes[0], head: item.heads[0], rows: item.rows[0], role: item.roles[0], expert: item.experts[0], part };
  };

  it('counts the band a sliding window keeps, and the entries outside it', () => {
    tips.tensor(pickOf('1:scores_masked', 'kept'), event);
    expect(el.hidden).toBe(false);
    expect(el.textContent).toContain(`${formatCount(keptEntries(T, 512))} entries`);
    tips.tensor(pickOf('1:scores_masked', 'outside'), event);
    expect(el.textContent).toContain('Outside the window');
    tips.tensor(pickOf('6:scores_masked', 'masked'), event);
    expect(el.textContent).toContain('Causally masked');
  });

  it('describes a tensor and a section, a pipe, and hides', () => {
    tips.tensor(pickOf('1:k', 'box'), event);
    expect(el.textContent).toContain('KV head 0 of 1, shared by Q heads 0–3');
    tips.part(floorParts(cfg, model, 1)[0], event);
    expect(el.textContent).toContain('Sliding-window attention · Layer 0');
    tips.edge({ text: 'k → k_cache', bytes: 2 ** 20, diameter: 123.4 } as unknown as Edge, event);
    expect(el.textContent).toContain('carries 1.00 MiB');
    expect(el.textContent).toContain('123 units across');
    tips.hide();
    expect(el.hidden).toBe(true);
  });
});

describe('tooltips across models', () => {
  /** Tooltip text for the first box of every tensor on `floor`, in each part a causal tile has. */
  function everyTooltip(id: ModelId, overrides: Partial<Settings>, floor: number): string[] {
    const cfg = MODELS[id];
    const model = buildModel(cfg, { ...baseOptions(cfg), ...overrides });
    const stats = summarize(model, cfg);
    const parts = floorParts(cfg, model, floor);
    const el = document.createElement('div');
    const tips = new Tooltips(el, () => ({ cfg, mode: overrides.mode ?? 'prefill', T: overrides.T ?? 300, focusFloor: floor, stats, parts, expertTokens: model.expertTokens }));
    const event = new MouseEvent('pointermove') as PointerEvent;
    const texts: string[] = [];
    for (const item of layoutTensors(model.tensors, { floorSpacing: 8000, thickness: 'bytes', focusFloor: floor }).placed.filter((p) => p.spec.floor <= floor)) {
      item.boxes.forEach((box, i) => {
        if (i > 3 && i % 50) return;
        for (const part of item.spec.causal ? (['kept', 'masked', 'outside'] as const) : (['box'] as const)) {
          tips.tensor({ item, box, head: item.heads[i], rows: item.rows[i], role: item.roles[i], expert: item.experts[i], part }, event);
          texts.push(el.textContent!);
        }
      });
    }
    for (const p of parts) {
      tips.part(p, event);
      texts.push(el.textContent!);
    }
    return texts;
  }

  it.each(MODEL_IDS)('%s: every tensor and section describes itself soundly', (id) => {
    const texts = [
      ...everyTooltip(id, { attention: 'eager', T: 700 }, layerFloor(1)),
      ...everyTooltip(id, { mode: 'decode', kvCache: 'static', T: 700, cacheCapacity: 4096 }, layerFloor(1)),
    ];
    expect(texts.length).toBeGreaterThan(20);
    for (const t of texts) expect(t).not.toMatch(/NaN|undefined|Infinity|\[object/);
    const all = texts.join('\n');
    expect(all).toMatch(/of all parameters/);
    expect(all).toMatch(/of the KV cache/);
    expect(all).toMatch(/Written by this step: 1 row|Reserved, not yet written/);
  });

  it('names heads, fused Q/K/V thirds and router columns', () => {
    const gpt2 = everyTooltip('gpt2', {}, layerFloor(0)).join('\n');
    expect(gpt2).toMatch(/[QKV] · Head \d+ of 12 \(multi-head attention: its own K and V\)/);
    const llama = everyTooltip('llama-3-8b', {}, layerFloor(0)).join('\n');
    expect(llama).toMatch(/Q head \d+ of 32 · KV group \d/);
    const mixtral = everyTooltip('mixtral-8x7b', {}, layerFloor(0)).join('\n');
    expect(mixtral).toMatch(/Expert \d's column/);
    expect(mixtral).toMatch(/E\d · Layer 0/);
  });
});

describe('palette', () => {
  it('publishes CSS variables', () => {
    applyPalette();
    expect(document.documentElement.style.getPropertyValue('--c-weight')).toMatch(/^#[0-9a-f]{6}$/);
  });
});

describe('controls: the remaining inputs', () => {
  it('toggles the data flow and moves the focused floor', () => {
    const root = document.createElement('div');
    const s = settingsFor('gpt2');
    const onChange = vi.fn();
    const cfg = MODELS.gpt2;
    createControls(root, s, {
      maxT: cfg.maxContext, mla: false, floorCount: floorCount(cfg), floorName: (f) => floorName(cfg, f),
      onChange, onOverview: vi.fn(), onFrameFocus: vi.fn(), onPlay: vi.fn(), isPlaying: () => true,
    });
    const flow = root.querySelector<HTMLInputElement>('input[type=checkbox]')!;
    flow.checked = false;
    flow.dispatchEvent(new Event('change'));
    expect(s.showFlow).toBe(false);
    const floor = root.querySelectorAll<HTMLInputElement>('input[type=range]')[1];
    floor.value = '13';
    floor.dispatchEvent(new Event('input'));
    expect(s.focusFloor).toBe(13);
    expect(root.textContent).toContain('LM head');
    expect(root.textContent).toContain('❚❚ Pause');
    expect(onChange.mock.calls.map((c) => c[0])).toEqual(['showFlow', 'focusFloor']);
  });
});

describe('stats panel: decode routing', () => {
  it('lists the experts a decode step uses (Mixtral)', () => {
    const cfg = MODELS['mixtral-8x7b'];
    const s = settingsFor('mixtral-8x7b', { mode: 'decode' });
    const model = buildModel(cfg, { ...s });
    const el = document.createElement('div');
    renderStats(el, { cfg, settings: s, stats: summarize(model, cfg), expertTokens: model.expertTokens, selection: { group: 3, head: null }, expertSelection: null });
    expect(rows(el)['Experts used, Layer 0']).toMatch(/^E\d · E\d$/);
    expect(rows(el)['Selected head (Esc clears)']).toBe('KV group 3 · Q heads 12–15');
    const prefill = settingsFor('mixtral-8x7b');
    renderStats(el, { cfg, settings: prefill, stats: summarize(buildModel(cfg, { ...prefill }), cfg), expertTokens: buildModel(cfg, { ...prefill }).expertTokens, selection: { group: 3, head: null }, expertSelection: null });
    expect(rows(el)['Tokens per expert, Layer 0']).toMatch(/^(\d+ · ){7}\d+$/);
  });

  it('describes a selected head of a multi-head model', () => {
    const cfg = MODELS.gpt2;
    const s = settingsFor('gpt2');
    const el = document.createElement('div');
    renderStats(el, { cfg, settings: s, stats: summarize(buildModel(cfg, { ...s }), cfg), selection: { group: 4, head: null }, expertSelection: null });
    expect(rows(el)['Selected head (Esc clears)']).toBe('Head 4, with its own K and V');
  });
});
