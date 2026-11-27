import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { defineConfig } from '@playwright/test'
import { SERVERS, serverPort, serverScratch } from './e2e/serverPool'

/** The interpreter the fixture server runs on: DW_E2E_PYTHON when set
 * (scripts/preflight.sh passes the one its pytest step used), else the
 * repo's own venv under either name, else python3 on PATH - so a worktree
 * or a `.venv` checkout runs e2e too, not only a checkout with `./venv`. */
function e2ePython(): string {
  if (process.env.DW_E2E_PYTHON) return process.env.DW_E2E_PYTHON
  for (const candidate of ['venv/bin/python', '.venv/bin/python']) {
    if (existsSync(fileURLToPath(new URL(`../${candidate}`, import.meta.url))))
      return `./${candidate}`
  }
  return 'python3'
}

/** E2E smoke: real server, real browser, no GPU jobs. The server runs
 * against a scratch copy of the example workflows and prompts (see
 * e2e/serve_fixture.py) so the save/delete specs never touch the repo. The
 * server import chain includes torch, so startup gets a generous timeout. */
export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  retries: 0,
  // One fixture server per worker (e2e/serverPool.ts): the specs write to
  // its scratch workspace, so two workers never share one. A file runs whole
  // on one worker, in order (no fullyParallel). DW_E2E_WORKERS sets how many
  workers: SERVERS,
  use: {
    baseURL: `http://127.0.0.1:${serverPort(0)}`,
  },
  webServer: Array.from({ length: SERVERS }, (_, index) => ({
    command: `${e2ePython()} ui/e2e/serve_fixture.py --port ${serverPort(index)}`,
    cwd: '..',
    url: `http://127.0.0.1:${serverPort(index)}/api/health`,
    reuseExistingServer: false,
    timeout: 90_000,
    // The UI's response contract runs strict here too: a key a route emits
    // that its response model has not declared is a 500 (dw/server/api_models.py)
    env: {
      ...process.env,
      DW_STRICT_RESPONSES: '1',
      DW_E2E_SCRATCH: serverScratch(index),
    },
  })),
})
