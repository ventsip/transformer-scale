import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset paths, so the build works under any sub-path (GitHub Pages serves it at /transformer-scale/).
  base: './',
  server: {
    // Native file events on Windows sometimes miss rapid successive writes, leaving stale modules.
    watch: { usePolling: true, interval: 200 },
  },
});
