# transformer-scale

An interactive 3D visualization of a decoder-only transformer, drawn **to scale**, to build intuition for the relative sizes of parameters, activations and the KV cache.

**Live:** https://ventsip.github.io/transformer-scale/ (WebGPU in recent Chrome or Edge; other browsers fall back to WebGL2. Add `?webgl` to the URL to force WebGL2, `?model=gpt2`, `gpt2-xl`, `llama-3-70b` or `mixtral-8x7b` to open another model.)

## Stack

- [Three.js](https://threejs.org/) with the WebGPU renderer (automatic WebGL2 fallback)
- TypeScript + [Vite](https://vite.dev/)

## Getting started

```sh
npm install
npm run dev
```

Then open the URL Vite prints (usually http://localhost:5173).
