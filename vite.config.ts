import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    // Native file events on Windows sometimes miss rapid successive writes, leaving stale modules.
    watch: { usePolling: true, interval: 200 },
  },
});
