# transformer-scale

An interactive 3D visualization of a decoder-only transformer, drawn **to scale**, to build intuition for the relative sizes of parameters, activations and the KV cache.

**Live:** https://ventsip.github.io/transformer-scale/ (needs a WebGPU browser, e.g. recent Chrome or Edge; falls back to WebGL2)

## Stack

- [Three.js](https://threejs.org/) with the WebGPU renderer (automatic WebGL2 fallback)
- TypeScript + [Vite](https://vite.dev/)

## Getting started

```sh
npm install
npm run dev
```

Then open the URL Vite prints (usually http://localhost:5173).
