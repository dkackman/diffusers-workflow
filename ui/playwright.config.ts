import { defineConfig } from '@playwright/test'

/** E2E smoke: real server, real browser, no GPU jobs. The server import
 * chain includes torch, so startup gets a generous timeout. */
export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  retries: 0,
  use: {
    baseURL: 'http://127.0.0.1:8971',
  },
  webServer: {
    command: './venv/bin/python -m dw.serve --port 8971',
    cwd: '..',
    url: 'http://127.0.0.1:8971/api/health',
    reuseExistingServer: false,
    timeout: 90_000,
  },
})
