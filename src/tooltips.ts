import { formatBytes, formatCount, formatFlops, formatInt, formatPercent } from './format';
import type { ModelConfig } from './model/config';
import { DTYPES } from './model/dtypes';
import type { Part } from './model/sections';
import type { Stats } from './model/stats';
import { byteCount, elementCount, floorLayer, floorName, isLayerFloor, kvGroup, type Mode, type TensorSpec } from './model/tensors';
import { cssColor } from './palette';
import type { Edge } from './scene/flowScene';
import { KIND_COLORS, type Pick } from './scene/tensorScene';

/** What the tooltips describe: the current model, pass and focused floor. */
export interface TooltipContext {
  cfg: ModelConfig;
  mode: Mode;
  T: number;
  focusFloor: number;
  stats: Stats;
  /** Sections of the focused floor. */
  parts: Part[];
  /** Mixture of experts: tokens routed to each expert, per decoder layer. */
  expertTokens?: number[][];
}

/** Gap between the pointer and the tooltip, and between the tooltip and the window's edges. */
const OFFSET = 16;
const EDGE_MARGIN = 8;

/** The hover tooltip: for a block (a tensor, or a part of one), a section tray, or a flow pipe. */
export class Tooltips {
  constructor(private readonly el: HTMLElement, private readonly context: () => TooltipContext) {}

  hide(): void {
    this.el.hidden = true;
  }

  tensor(p: Pick, e: PointerEvent): void {
    const { cfg } = this.context();
    const s = p.item.spec;
    const dims = s.dims.map((d) => `${d.name} ${formatInt(d.size)}`).join(' × ');
    this.show(`
    <div class="tt-title">${s.label}</div>
    <div class="tt-kind" style="color:${cssColor(KIND_COLORS[s.kind].edge)}">${s.kind} · ${DTYPES[s.dtype].label} · ${floorName(cfg, s.floor)}</div>
    <div>${dims}</div>
    <div><b>${formatCount(elementCount(s))}</b> elements · <b>${formatBytes(byteCount(s))}</b></div>
    ${this.headLine(p)}
    ${expertLine(p)}
    ${this.maskLine(p)}
    ${rowLine(p)}
    <div class="tt-share">${this.share(s)}</div>
    ${s.note ? `<div class="tt-note">${s.note}</div>` : ''}`, e);
  }

  /** Totals of a section of the focused floor: parameters and their share, memory, and this pass's work. */
  part(part: Part, e: PointerEvent): void {
    const { cfg, mode, focusFloor, stats, parts, expertTokens } = this.context();
    // Shares are of the whole layer: its top-level parts (each expert is also inside "Experts").
    const layer = parts.filter((p) => p.kind !== 'expert');
    const sum = (f: (p: Part) => number) => layer.reduce((a, p) => a + f(p), 0);
    const pct = (x: number, of: number) => (of > 0 ? formatPercent(x / of) : '–');
    const pass = mode === 'decode' ? 'this decode step' : 'the prompt';
    const routed = part.expert !== undefined ? expertTokens?.[floorLayer(focusFloor)]?.[part.expert] : undefined;
    const rows: string[] = [
      `<div><b>${formatCount(part.params)}</b> parameters · <b>${formatBytes(part.weightBytes)}</b></div>`,
      `<div class="tt-head">${pct(part.params, sum((p) => p.params))} of this layer's parameters · ${pct(part.params, stats.params)} of the model's</div>`,
      `<div>Transient activations: <b>${formatBytes(part.activationBytes)}</b></div>`,
    ];
    if (part.kvBytes > 0) rows.push(`<div>KV cache: <b>${formatBytes(part.kvBytes)}</b></div>`);
    rows.push(
      `<div>FLOPs for ${pass}: <b>${formatFlops(part.flops)}</b> (${pct(part.flops, sum((p) => p.flops))} of the layer)</div>`,
      `<div>Weights + KV read: <b>${formatBytes(part.bytesRead)}</b> (${pct(part.bytesRead, sum((p) => p.bytesRead))} of the layer)</div>`,
    );
    if (routed !== undefined) {
      rows.push(`<div class="tt-note">${part.idle ? 'No token is routed to this expert in this pass: its weights are not read.'
        : `${formatInt(routed)} ${routed === 1 ? 'token is' : 'tokens are'} routed to this expert in this pass.`}</div>`);
    }
    this.show(`<div class="tt-title">${part.name} · ${floorName(cfg, focusFloor)}</div>
    <div class="tt-kind">section</div>${rows.join('')}`, e);
  }

  edge(edge: Edge, e: PointerEvent): void {
    this.show(`
    <div class="tt-title">${edge.text}</div>
    <div>carries <b>${formatBytes(edge.bytes)}</b></div>
    <div class="tt-note">Pipe width grows with the log of the bytes: ${formatInt(Math.round(edge.diameter))} units across.</div>`, e);
  }

  private show(html: string, e: PointerEvent): void {
    const el = this.el;
    el.innerHTML = html;
    el.hidden = false;
    const x = Math.min(e.clientX + OFFSET, window.innerWidth - el.offsetWidth - EDGE_MARGIN);
    const y = Math.min(e.clientY + OFFSET, window.innerHeight - el.offsetHeight - EDGE_MARGIN);
    el.style.transform = `translate(${x}px, ${y}px)`;
  }

  private share(s: TensorSpec): string {
    const { cfg, stats } = this.context();
    const b = byteCount(s);
    const pct = (x: number, of: number) => formatPercent(x / of);
    if (s.kind === 'weight') return `${pct(elementCount(s), stats.params)} of all parameters`;
    if (s.kind === 'kv') return `${pct(b, stats.kvBytes)} of the KV cache`;
    if (s.kind === 'activation' && isLayerFloor(cfg, s.floor)) return `${pct(b, stats.layerActBytes)} of this layer's transient activations`;
    return '';
  }

  private maskLine({ part }: Pick): string {
    const { T } = this.context();
    if (part === 'kept') return `<div class="tt-head">Kept by the causal mask (key ≤ query): <b>${formatCount((T * (T + 1)) / 2)}</b> entries per T×T tile</div>`;
    if (part === 'masked') return `<div class="tt-head">Causally masked (key > query): <b>${formatCount((T * (T - 1)) / 2)}</b> entries per T×T tile: −∞ before softmax, 0 after</div>`;
    return '';
  }

  private headLine({ item, box, head, role }: Pick): string {
    const { cfg } = this.context();
    const split = item.spec.heads;
    if (!split || head === null) return '';
    const band = box.sx * box.sz;
    const what = cfg.nHeads === cfg.nKvHeads
      ? `Head ${head} of ${split.count} (multi-head attention: its own K and V)`
      : split.perGroup === 1
      ? `KV head ${head} of ${split.count}, shared by Q heads ${head * (cfg.nHeads / cfg.nKvHeads)}–${(head + 1) * (cfg.nHeads / cfg.nKvHeads) - 1}`
      : `Q head ${head} of ${split.count} · KV group ${kvGroup(split, head)}`;
    // In a fused Q/K/V tensor, also which third the band is in.
    const part = item.spec.fusedRoles && role ? `${role.toUpperCase()} · ` : '';
    return `<div class="tt-head">${part}${what}: <b>${formatCount(band)}</b> elements · <b>${formatBytes(band * DTYPES[item.spec.dtype].bytes)}</b></div>`;
  }
}

function rowLine({ item, rows }: Pick): string {
  if (item.spec.filledZ === undefined) return '';
  const fresh = item.spec.freshZ ?? 0;
  const line = (text: string) => `<div class="tt-head">${text}</div>`;
  if (rows === 'reserved') return line(`Reserved, not yet written: <b>${formatInt(item.spec.z - item.spec.filledZ)}</b> of ${formatInt(item.spec.z)} slots`);
  if (rows === 'fresh') return line(`Written by this step: <b>${formatInt(fresh)}</b> row (the new token)`);
  return line(`Written by earlier passes: <b>${formatInt(item.spec.filledZ - fresh)}</b> rows`);
}

/** For a router tensor's column: which expert it scores. */
function expertLine({ item, box, expert }: Pick): string {
  if (!item.spec.expertAxis || expert === null) return '';
  const band = box.sx * box.sz;
  return `<div class="tt-head">Expert ${expert}'s column: <b>${formatCount(band)}</b> elements · ` +
    `<b>${formatBytes(band * DTYPES[item.spec.dtype].bytes)}</b> (click to select the expert)</div>`;
}
