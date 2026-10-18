# transformer-scale

An interactive 3D visualization of a decoder-only transformer, drawn **to scale**, to build intuition for the relative sizes of parameters, activations and the KV cache.

**Live:** https://ventsip.github.io/transformer-scale/ (WebGPU in recent Chrome or Edge; other browsers fall back to WebGL2. Add `?webgl` to the URL to force WebGL2, `?model=gpt2`, `gpt2-xl`, `llama-3-70b`, `mixtral-8x7b`, `deepseek-v2-lite` or `gemma-3-1b` to open another model.)

## Stack

- [Three.js](https://threejs.org/) with the WebGPU renderer (automatic WebGL2 fallback)
- TypeScript + [Vite](https://vite.dev/)

## Getting started

```sh
npm install
npm run dev
```

Then open the URL Vite prints (usually http://localhost:5173).

## Testing

```sh
npm run typecheck      # app and test code
npm test               # unit, model and DOM tests (Vitest)
npm run test:coverage  # the same, with coverage thresholds enforced
npm run test:e2e       # the production build in Chromium (Playwright; first run: npx playwright install chromium)
npm run check          # all of the above
```

- **Unit** (`tests/unit/`): formatting, the seeded random numbers, RoPE and YaRN frequencies, expert routing, the
  model configs and the builder's helpers.
- **Model** (`tests/model/`): every model under every combination of settings (mode, attention, KV cache, elementwise
  detail, MLA cache, three lengths). Graph invariants (every op linked, every activation produced once, shapes
  matching dims); parameter counts, KV-cache sizes and FLOPs against independent formulas and published totals; the
  layout (finite boxes, footprints to scale, no overlaps); and golden snapshots of the totals and of a few floors.
- **DOM** (`tests/ui/`, jsdom): every popup variant of every model, opened, driven through all its controls and
  hovered, checked for broken numbers; the stats panel, controls, section trays and tooltips.
- **End-to-end** (`e2e/`): each model loads and renders with WebGPU and with WebGL2 without errors, and the controls,
  popups, generation loop and tooltips work in the browser.

An intended change to the model shows up as a snapshot diff: review it, then accept it with `npm run test:update`.
CI (`.github/workflows/ci.yml`) runs everything on pull requests and branches; the deploy runs it first, so a failing
test blocks the release.
