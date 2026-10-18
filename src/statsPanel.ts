import { formatBytes, formatCount, formatFlops, formatInt } from './format';
import { THICKNESS_PER_BYTE } from './layout';
import type { ModelConfig } from './model/config';
import type { Stats } from './model/stats';
import { floorLayer, floorName, type HeadSelection } from './model/tensors';
import { cssColor } from './palette';
import { LOOP_COLOR, RESIDUAL_COLOR, pipeDiameter } from './scene/flowScene';
import { KIND_COLORS, MASKED_COLORS, ROLE_COLORS } from './scene/tensorScene';
import type { Settings } from './ui/controls';

/** What the stats panel shows: the model, this pass's totals, and the current selections. */
export interface StatsContext {
  cfg: ModelConfig;
  settings: Settings;
  stats: Stats;
  /** Mixture of experts: tokens routed to each expert, per decoder layer. */
  expertTokens?: number[][];
  selection: HeadSelection | null;
  expertSelection: number | null;
}

/** Values longer than this take a row of their own. */
const WIDE_VALUE = 28;

export function renderStats(el: HTMLElement, c: StatsContext): void {
  const { cfg, settings, stats } = c;
  const isStatic = settings.kvCache === 'static';
  const decode = settings.mode === 'decode';
  // Tokens in the cache once this pass is done.
  const cached = settings.T + (decode ? 1 : 0);
  const kvPerToken = stats.kvBytes / (isStatic ? settings.cacheCapacity : cached);
  const pass = decode ? 'per generated token' : 'for the prompt';
  const rows: [string, string][] = [
    ['Parameters', `${formatCount(stats.params)} · ${formatBytes(stats.paramBytes)}`],
    ...(cfg.moe ? expertRows(c) : []),
    [isStatic ? `KV cache, static: ${formatInt(settings.cacheCapacity)} slots` : `KV cache @ ${formatInt(cached)} tokens`,
      `${formatCount(stats.kvElements)} · ${formatBytes(stats.kvBytes)}`],
    ['KV cache per token', formatBytes(kvPerToken)],
    ...(isStatic ? [['KV cache filled', `${formatInt(cached)} of ${formatInt(settings.cacheCapacity)} slots`] as [string, string]] : []),
    ['Transient activations, one layer', formatBytes(stats.layerActBytes)],
    [`FLOPs ${pass}`, formatFlops(stats.flops)],
    [`Weights + KV read ${pass}`, formatBytes(stats.bytesRead)],
    ['FLOPs per byte read', (stats.flops / stats.bytesRead).toPrecision(3)],
  ];
  if (settings.attention === 'eager') rows.push(['Eager score tensors, one layer', formatBytes(stats.scoresBytes)]);
  if (c.selection) rows.push(['Selected head (Esc clears)', describeSelection(cfg, c.selection)]);
  if (c.expertSelection !== null) rows.push(['Selected expert (Esc clears)', `E${c.expertSelection}`]);
  el.replaceChildren(
    ...rows.map(([k, v]) => {
      const div = document.createElement('div');
      div.className = v.length > WIDE_VALUE ? 'stat wide' : 'stat';
      div.innerHTML = `<span>${k}</span><b>${v}</b>`;
      return div;
    }),
  );
}

function describeSelection(cfg: ModelConfig, sel: HeadSelection): string {
  const perGroup = cfg.nHeads / cfg.nKvHeads;
  if (perGroup === 1) return `Head ${sel.group}, with its own K and V`;
  const first = sel.group * perGroup;
  const group = `KV group ${sel.group}`;
  return sel.head === null ? `${group} · Q heads ${first}–${first + perGroup - 1}` : `Q head ${sel.head} · ${group}`;
}

/** Mixture of experts: active parameters, and how the focused layer routes this pass's tokens. */
function expertRows({ cfg, settings, stats, expertTokens }: StatsContext): [string, string][] {
  const { nExperts, topK } = cfg.moe!;
  const rows: [string, string][] = [[`Active per token (${topK} of ${nExperts} experts)`, formatCount(stats.activeParams)]];
  const counts = expertTokens?.[floorLayer(settings.focusFloor)];
  if (!counts) return rows;
  const layer = floorName(cfg, settings.focusFloor);
  if (settings.mode === 'decode') {
    const used = counts.flatMap((n, e) => (n > 0 ? [e] : []));
    rows.push([`Experts used, ${layer}`, used.map((e) => `E${e}`).join(' · ')]);
  } else if (counts.length <= MAX_LISTED_EXPERTS) {
    rows.push([`Tokens per expert, ${layer}`, counts.map(formatInt).join(' · ')]);
  } else {
    const idle = counts.filter((n) => n === 0).length;
    rows.push([`Tokens per expert, ${layer}`,
      `${formatInt(Math.min(...counts))}–${formatInt(Math.max(...counts))}, mean ${formatInt(Math.round(counts.reduce((a, b) => a + b, 0) / counts.length))}` +
      (idle ? ` · ${idle} idle` : '')]);
  }
  return rows;
}

/** Above this many experts, the stats summarize the tokens per expert instead of listing them. */
const MAX_LISTED_EXPERTS = 16;

/** Example payloads for the legend's pipe widths. */
const WIDTH_EXAMPLES = [['8 KiB', 2 ** 13], ['16 MiB', 2 ** 24], ['1 GiB', 2 ** 30]] as const;

/** The color legend and the scale notes. */
export function renderLegend(el: HTMLElement): void {
  const widths = WIDTH_EXAMPLES.map(([name, bytes]) => `${name} ≈ ${formatInt(Math.round(pipeDiameter(bytes)))}`).join(', ');
  const swatch = (hex: number, label: string) => `<span><i style="background:${cssColor(hex)}"></i>${label}</span>`;
  el.innerHTML = [
    swatch(KIND_COLORS.weight.fill, 'Weights'),
    swatch(KIND_COLORS.activation.fill, 'Transient activations'),
    swatch(KIND_COLORS.kv.fill, 'K, V: KV cache'),
    swatch(ROLE_COLORS.q, 'W_q: query projection'),
    swatch(ROLE_COLORS.k, 'W_k: key projection'),
    swatch(ROLE_COLORS.v, 'W_v: value projection'),
    swatch(RESIDUAL_COLOR, 'Residual stream'),
    swatch(LOOP_COLOR, 'Generation loop'),
    swatch(KIND_COLORS.buffer.fill, 'Buffers (fixed, not learned)'),
    swatch(KIND_COLORS.scores.fill, 'Attention scores (eager only)'),
    swatch(MASKED_COLORS.fill, 'Causally masked scores'),
  ].join('') +
    `<p>1 unit = 1 element along both floor axes. Thickness: 1 byte = ${THICKNESS_PER_BYTE} units (exaggerated).
  Weights: input dim → X, output dim → Z. Activations: features → X, tokens → Z.
  Flow pipe diameter grows with the log of the bytes carried (in world units, so with perspective): ${widths}.</p>`;
}
