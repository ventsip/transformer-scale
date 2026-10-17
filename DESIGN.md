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
- Floors above the focused one are lifted and floors below it lowered, opening space around it; other floors are dimmed.

## Data flow

- **Operations** (`@`, RMSNorm, `+`, attention, SiLU ⊙, lookup) have no parameters, so they get no world size:
  they are screen-space labels hovering above the center of the tensor they produce. Hover for shapes and FLOPs.
- **Edges** arc from each input up into the operation and drop straight into its output, colored by the tensor
  kind they carry, as pipes whose diameter grows with the log of the bytes they carry. Diameters are in world
  units (8 KiB = 65, 16 MiB ≈ 320), so pipes have perspective like the blocks; unlike the blocks they are not to scale;
  the embedding lookup counts only the rows it reads, and a selected head only its share. Moving dots show direction.
- **The residual stream** (x, h and the adds between them) is drawn in cyan; reads that branch off it
  and branch outputs joining it keep their kind's color.
- Flow is drawn for the focused floor, plus the op on the floor below that produces its input. An op whose output is
  on a higher floor sits just below that output; its inputs rise to it. Edges between
  floors are risers: they leave heading up and enter the upper block from below.

## Operation popups

- Clicking an op label that has an interactive view (accent border, pointer cursor) opens a popup explaining it.
- **Sampling** (opened from any sampling op: to fp32, ÷ temperature, top-k / top-p, softmax, sample; the clicked step
  is highlighted): illustrative logits for the top 40 of 128,256 tokens, then each step with its controls and chart -
  temperature (outlines show the input on the same scale), top-k and top-p (removed tokens as outlines), softmax, and a
  Sample button that marks the drawn token. Defaults follow Llama 3 (T = 0.6, top-p = 0.9) and HF (top-k = 50); all
  numbers are computed over the whole vocabulary.

## Attention heads

- Head-structured tensors (`W_q`, `W_k`, `W_v`, Q, K, V, attn, `W_o`, scores) are drawn as one band per head with
  no gaps, so the footprint stays exact. Bands run along whichever axis carries the head dimension.
- Bands are tinted by **KV group**: a K/V head and the query heads that share it get the same hue, which makes
  grouped-query attention visible (8 groups of 4 strips in Q next to 8 single strips in K).
- **Clicking a band selects that head** (a K/V band selects its whole group): bands off its path dim, and flow
  arcs and ops move to the selected bands. The selection holds across layers; Esc or empty space clears it.

## Position encoding (RoPE)

- Llama uses rotary position embedding: no parameters. The embedding floor holds what HF allocates for it once per
  forward pass: `position_ids`, the `inv_freq` buffer (64 frequencies; a fixed buffer, drawn in its own color and not
  counted as parameters) and the `cos`/`sin` tables (T × head_dim), shared by every layer and head.
- In each layer, `@` produces Q and K before rotation; a `RoPE` op rotates them, fed by risers from the tables below.
  K is cached after rotation.

## Elementwise detail

- *Compact*: RMSNorm and SwiGLU are single ops.
- *Expanded*: every intermediate HF allocates. RMSNorm: fp32 upcast, x², mean (T × 1), rsqrt (T × 1), scale, cast
  back, × weight. SwiGLU: SiLU(g) as its own T × d_ffn tensor, then ⊙ u. Intermediates stack along Z in their station,
  so floors get deeper rather than wider.

## KV cache

- *Dynamic* (HF default): in prefill the cache keeps K and V as computed, so it is exactly T slots per layer.
- *Static*: K/V buffers are preallocated per layer for the full capacity (the model's max context). K and V become
  transient, a `cache write` copies them into the first T slots, and attention reads from the buffers. Reserved slots
  are drawn dimmed. (HF attends over all slots with a mask in this mode; attention shapes are drawn for T only.)

## Decode

- *Mode* switches between **prefill** (a T-token prompt) and **decode** (one generation step: 1 new token, L already
  cached). In decode every activation is a single row while every weight is still read in full, so the panel shows
  ~1 FLOP per byte read: decode is memory-bound. Prefill reuses each weight across T tokens (~T FLOPs per byte).
- Attention reads the whole cache: K/V pipes carry L+1 rows; eager scores are one row per head.
- Cache writes differ: a static cache writes one row into its buffer; a dynamic cache (HF `DynamicCache`) uses
  torch.cat, copying the whole previous cache (drawn faded) into a new tensor one row longer. Rows written by this
  step are highlighted.
- **The generation loop**: a white pipe from the sampled token, down the outside of the building, into input_ids.
  It carries 8 bytes, so it is the thinnest pipe in the scene: the whole output of a pass that reads every weight.
- **Generate** steps through decode one token at a time (1 to 64 tokens/s): each step adds a row to every cache and
  the loop flashes.

## Honesty

- **Lifetime matters.** In inference, transient activations exist for one layer at a time, so by default they are
  drawn only on the focused floor. K and V are written to the KV cache and are drawn on every floor.
- **Only draw what is allocated.** The attention setting picks the implementation. *Fused* (FlashAttention / SDPA)
  has no score tensors at all. *Eager* (HF `eager_attention_forward`) draws every tensor it allocates: `repeat_kv`
  copies of K and V, raw scores, masked scores, fp32 softmax, its cast back, plus one T×T causal mask on the
  embedding floor shared by all layers.
- **The causal mask** splits tiles where it applies along the diagonal (keys along X, queries along Z): the half with
  key ≤ query is kept, the other half is drawn as masked (−∞, or 0 after softmax). Raw scores are drawn whole,
  because eager attention computes them in full. Fused kernels skip the masked half, which is why the fused
  attention op reports about half the FLOPs.
- **Never exaggerate silently.** Sub-pixel tensors (norm weights, logits) stay at true size; 1px edges keep them visible.
  The thickness exaggeration is stated in the legend.
- **Every object is inspectable:** shape, element count, bytes, and share of its category on hover.

## Scope

- Model: Llama-3-8B (`src/model/config.ts`); everything is derived from the config.
- Mode: static prefill snapshot for a prompt of T tokens. Logits are computed for the last position only, then
  sampled as in HF `generate`: fp32 copy, temperature, top-k / top-p, softmax, and one int64 token id.
- Simplifications: all weights share one dtype (no per-tensor quantization scales); RoPE, softmax and SiLU have no
  parameters and are not drawn as objects.
