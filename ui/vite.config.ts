/// <reference types="vitest/config" />
import { svelte } from '@sveltejs/vite-plugin-svelte'
import { defineConfig } from 'vite'

// Dev server proxies API and outputs to the dw server (python -m dw.serve)
export default defineConfig({
  plugins: [svelte()],
  // Vite's 500 kB notice fires on Monaco's own chunk (~3.9 MB, loaded only
  // when the JSON editor opens). What a page loads is budgeted, and failed,
  // by e2e/performance.spec.ts instead - raised only with Don
  build: { chunkSizeWarningLimit: 4500 },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8765',
      '/outputs': 'http://127.0.0.1:8765',
    },
  },
  // Svelte's server-side build (the default package resolution) has no
  // mount()/component lifecycle - @testing-library/svelte needs the
  // client build, which the 'browser' condition selects.
  resolve: {
    conditions: process.env.VITEST ? ['browser'] : undefined,
  },
  test: {
    exclude: ['e2e/**', 'node_modules/**'],
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
  },
})
