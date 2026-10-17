# Design principles

## Encoding

- **1 unit = 1 element** along both floor axes. A tensor's footprint is its true shape, so area = element count.
- **Thickness = bytes per element**, on a separate exaggerated scale (1 byte = 64 units). Footprint × thickness ∝ memory.
  A "uniform" mode drops precision and compares element counts only.
- **Orientation:** weights lie with their *input* dim along X and *output* dim along Z.
  Activations lie with their *feature* dim along X and *tokens* along Z, so the token axis grows toward the viewer.

## Layout

- One **floor** per stage: embedding, one per decoder layer, LM head. Floors stack along Y.
- A layer floor has two **sections**: attention in front, MLP behind it. This keeps floors near 2.5:1
  instead of a 5.6:1 strip.
- In each section, **stations** run left to right in data-flow order around a **bus** line. Weights sit
  behind the bus (−Z), the activations they produce in front of it (+Z).
- Each floor shows the input it receives. A layer's output (h + m) is the next floor's `x`.
- Floors above the focused one are lifted out of the way and other floors are dimmed.

## Data flow

- **Operations** (`@`, RMSNorm, `+`, attention, SiLU ⊙, lookup) have no parameters, so they get no world size:
  they are screen-space labels on the bus, right behind the tensor they produce. Hover for shapes and FLOPs.
- **Edges** arc from each input to the operation and from the operation to its output, colored by the tensor
  kind they carry; moving dots show direction.
- Flow is drawn for the focused floor only. Data leaving or entering the floor ends in a short labelled stub
  ("→ Layer 1", "from Embedding") rather than an arc across floors.

## Attention heads

- Head-structured tensors (`W_q`, `W_k`, `W_v`, Q, K, V, attn, `W_o`, scores) are drawn as one band per head with
  no gaps, so the footprint stays exact. Bands run along whichever axis carries the head dimension.
- Bands are tinted by **KV group**: a K/V head and the query heads that share it get the same hue, which makes
  grouped-query attention visible (8 groups of 4 strips in Q next to 8 single strips in K).
- **Clicking a band selects that head** (a K/V band selects its whole group): bands off its path dim, and flow
  arcs and ops move to the selected bands. The selection holds across layers; Esc or empty space clears it.

## Honesty

- **Lifetime matters.** In inference, transient activations exist for one layer at a time, so by default they are
  drawn only on the focused floor. K and V are written to the KV cache and are drawn on every floor.
- **Not-stored tensors are ghosts.** Attention scores (heads × T × T) are never materialized by fused attention
  kernels; they are optional, translucent, and labelled as such.
- **Never exaggerate silently.** Sub-pixel tensors (norm weights, logits) stay at true size; 1px edges keep them visible.
  The thickness exaggeration is stated in the legend.
- **Every object is inspectable:** shape, element count, bytes, and share of its category on hover.

## Scope

- Model: Llama-3-8B (`src/model/config.ts`); everything is derived from the config.
- Mode: static prefill snapshot for a prompt of T tokens. Logits are computed for the last position only.
- Simplifications: all weights share one dtype (no per-tensor quantization scales); RoPE, softmax and SiLU have no
  parameters and are not drawn as objects.
