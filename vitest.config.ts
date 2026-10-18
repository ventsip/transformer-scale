import { defineConfig, mergeConfig } from 'vitest/config';
import viteConfig from './vite.config';

export default mergeConfig(viteConfig, defineConfig({
  test: {
    include: ['tests/**/*.test.ts'],
    // Node by default; DOM tests opt in with a `@vitest-environment jsdom` comment.
    environment: 'node',
    setupFiles: ['tests/support/setup.ts'],
    // The model sweeps build hundreds of models; give slow CI machines room.
    testTimeout: 60_000,
    coverage: {
      provider: 'v8',
      include: ['src/**/*.ts'],
      // WebGPU / WebGL rendering and app wiring: covered by the Playwright tests in e2e/, not by unit tests.
      exclude: ['src/main.ts', 'src/camera.ts', 'src/scene/**'],
      reporter: ['text-summary', 'html', 'lcov'],
      thresholds: { lines: 99, functions: 98, branches: 95, statements: 99 },
    },
  },
}));
