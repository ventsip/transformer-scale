import { defineConfig, devices } from '@playwright/test';

// End-to-end tests against the production build (vite preview), in Chromium.
const PORT = 4173;

export default defineConfig({
  testDir: 'e2e',
  fullyParallel: true,
  // Each page renders a large scene, in software on machines without a GPU: a few at a time.
  workers: process.env.CI ? 1 : 3,
  timeout: 90_000,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['github'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: `http://localhost:${PORT}/`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1400, height: 900 } } }],
  webServer: {
    command: `npm run build && npx vite preview --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}/`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
});
