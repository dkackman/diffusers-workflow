/// <reference types="vitest/config" />
import { svelte } from '@sveltejs/vite-plugin-svelte'
import { defineConfig } from 'vite'

// Dev server proxies API and outputs to the dw server (python -m dw.serve)
export default defineConfig({
  plugins: [svelte()],
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
