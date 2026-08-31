import { defineConfig } from '@playwright/test'

/** E2E smoke: real server, real browser, no GPU jobs. The server runs
 * against a scratch copy of the example workflows and prompts (see
 * e2e/serve_fixture.py) so the save/delete specs never touch the repo. The
 * server import chain includes torch, so startup gets a generous timeout. */
export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  retries: 0,
  // Serialized on purpose: the fixture server's first-time import of a
  // diffusers pipeline module is not concurrency-safe (CPython import-lock
  // deadlock) - two specs hitting introspection/validation/save at once
  // (e.g. an editor page load racing another editor page's save) can wedge
  // the server. A real fix belongs in the engine, not the test config; this
  // is a recorded follow-up. Until then the suite runs on one worker so the
  // gate is deterministic.
  workers: 1,
  use: {
    baseURL: 'http://127.0.0.1:8971',
  },
  webServer: {
    command: './venv/bin/python ui/e2e/serve_fixture.py --port 8971',
    cwd: '..',
    url: 'http://127.0.0.1:8971/api/health',
    reuseExistingServer: false,
    timeout: 90_000,
  },
})
