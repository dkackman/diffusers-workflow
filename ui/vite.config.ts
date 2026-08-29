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
})
