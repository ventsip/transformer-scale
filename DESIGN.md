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
  The gap grows with the deepest floor (1.5 × its depth, at least 40,000), so deep floors stay visible from the focus view.

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
- **SwiGLU** (opened from `SiLU ⊙`, or in expanded mode `SiLU` and `⊙`; in Llama and in every Mixtral expert): SiLU(x) =
  x·σ(x) as a line against a dashed ReLU reference, with a hover crosshair and its minimum (−0.278 at x ≈ −1.28)
  marked; then illustrative g, u and a = SiLU(g) ⊙ u for 32 features of one token on one shared scale (outlines show
  each input), with a button to draw another token.
- **RMSNorm** (opened from `RMSNorm`, or any step of the expanded chain; identified by output id, since labels like
  `× weight` repeat elsewhere): one illustrative 64-feature token with two outlier features and a scale slider; x² with
  its mean, and 1/√(mean + ε); x / RMS(x) against ±1 lines with x as outlines, which stays put as the slider scales x
  (until mean(x²) nears ε); then the learned weight and the output.
- **RoPE** (opened from `cos(pos·θ)` / `sin(pos·θ)` on the embedding floor, or `RoPE` in any layer; computed from the
  model's RoPE base, head_dim and context): the wavelength of each of the 64 pairs on a log axis against the context
  length; dials for 8 pairs rotating with a position slider; and the score of identical q and k against their distance
  m − n (log axis), which shows that RoPE makes scores depend on distance only, decaying with ripples.
- **Attention** (opened from fused `attention`, or any eager step: repeat_kv, the causal mask, QKᵀ/√d, + mask, softmax,
  its cast, · V): one illustrative head over 12 tokens, with patterns real heads show (a sink on the first token, the
  previous token, "it" → "cat"). Scores as a diverging heatmap with a causal-mask toggle (masked cells hatched), softmax
  probabilities as a sequential heatmap, and the selected query's weights over the keys (click a row). A note gives the
  real cost of the current pass: the T × T grids eager attention allocates, which fused kernels never write, or in decode
  the single row against the whole cache.
- **Mixture of experts** (opened from the router ops, any `gather (E…)`, or `index_add` / the expanded `× weight`):
  one illustrative token's router logits, softmax with the top 2 kept and their renormalized weights; the tokens each
  expert receives in this layer and pass, from the same seeded routing the scene draws (or the 2 experts the new token
  uses, in decode); and the combine, two weighted expert outputs summing to m.

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

## Mixture of experts (Mixtral-8x7B)

- Mixtral shares Llama-3-8B's attention, norms and dimensions; only the MLP differs, so switching models isolates MoE.
  Each layer has 8 SwiGLU experts (HF `MixtralSparseMoeBlock`), and each token runs through 2 of them.
- The MLP section holds the router (`gate.weight`, 4096 × 8; router logits, fp32 softmax, top-2 expert ids and
  renormalized routing weights) and `m`, where `index_add` sums the weighted expert outputs back into token order.
  Behind it the 8 experts form a **4 × 2 grid**, one section per row. Each expert is a near-square block of two
  stations: w1 and w3 back to back (both read the same rows) with x, g and u in front, then w2 with a and y.
- Per expert, `gather` copies the h_norm rows of its tokens; its activations have one row per routed token, so in
  prefill their depths show the load balance. An expert no token picks keeps its weights, dimmed, and gets no
  activations, ops or reads. Every expert reserves the width and depth of the busiest one, so the layout does not
  depend on routing and experts stay put while generating.
- In decode only 2 of 8 experts per layer light up, and the pair changes from token to token: parameters 46.7B,
  active per token 12.9B, and decode reads ~26 GB per token instead of all ~93 GB.
- Routing is **illustrative and seeded** (`src/model/routing.ts`): mild per-layer imbalance, and consecutive tokens
  repeat their first expert more often than chance, as the Mixtral paper observes. Real routing depends on the text.

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

- Models: Llama-3-8B and Mixtral-8x7B (`src/model/config.ts`); everything is derived from the config.
- Mode: static prefill snapshot for a prompt of T tokens. Logits are computed for the last position only, then
  sampled as in HF `generate`: fp32 copy, temperature, top-k / top-p, softmax, and one int64 token id.
- Simplifications: all weights share one dtype (no per-tensor quantization scales); RoPE, softmax and SiLU have no
  parameters and are not drawn as objects.
