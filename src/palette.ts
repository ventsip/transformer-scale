/**
 * The colors shared by the 3D scene, the stylesheet and the popup charts, as 0xRRGGBB. The scene reads them directly;
 * CSS reads them as variables named --c-<key in kebab case> (e.g. --c-weight-edge), which applyPalette sets.
 */
export const PALETTE = {
  background: 0x0b0d12,
  // Tensor kinds: fill and edge.
  weight: 0x4f8ff7,
  weightEdge: 0x9cc2ff,
  activation: 0xf5a524,
  activationEdge: 0xffd27a,
  kv: 0x22c55e,
  kvEdge: 0x86efac,
  scores: 0xef4444,
  scoresEdge: 0xf87171,
  buffer: 0xa78bfa,
  bufferEdge: 0xc4b5fd,
  // Attention projections: queries, keys, values.
  q: 0xe879f9,
  k: 0xfb923c,
  v: 0x2dd4bf,
  // Flow: the residual stream, the generation loop, and the pulse along pipes when a token is generated.
  residual: 0x67e8f9,
  loop: 0xf8fafc,
  pulse: 0xfbbf24,
  // Hover highlights.
  highlight: 0xffffff,
  // Section trays: attention and the router lean blue and teal, the MLP and experts warm.
  trayAttention: 0x5b7bb5,
  trayMlp: 0xa0785a,
  trayRouter: 0x4f9a94,
  trayExpert: 0xc9a27a,
  // Lights: the hemisphere light's ground color.
  ground: 0x223344,
  // Neutrals (lightest to darkest) and accents, for masked blocks, chart lines and backgrounds.
  neutral100: 0xf1f5f9,
  neutral200: 0xe2e8f0,
  neutral400: 0x94a3b8,
  neutral500: 0x64748b,
  neutral600: 0x475569,
  neutral700: 0x334155,
  neutral800: 0x1e293b,
  amber200: 0xfde68a,
  blue400: 0x60a5fa,
} as const;

/** A palette color as a CSS hex color. */
export const cssColor = (c: number) => `#${c.toString(16).padStart(6, '0')}`;

/** Publishes the palette as CSS variables on the document root. */
export function applyPalette(root = document.documentElement): void {
  for (const [key, c] of Object.entries(PALETTE)) {
    root.style.setProperty(`--c-${key.replace(/([a-z])([A-Z0-9])/g, '$1-$2').toLowerCase()}`, cssColor(c));
  }
}
