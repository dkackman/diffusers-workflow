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
  test: {
    exclude: ['e2e/**', 'node_modules/**'],
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
  },
  resolve: {
    // vitest otherwise resolves svelte's server-side (SSR) build, which
    // has no mount() - component tests need the browser build.
    conditions: process.env.VITEST ? ['browser'] : undefined,
  },
})
