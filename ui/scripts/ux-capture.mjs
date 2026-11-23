// Screenshots every route of the UI, in both themes at two widths, against the
// e2e fixture server, so a reviewer sees the whole UI in one folder.
//
//   npm run ux:capture [OUT]      (run `npm run build` first; ui/dist is served)
//
// Files are <name>-<theme>-<width>.png. Port 8972 (e2e uses 8971).
import { spawn } from 'node:child_process'
import { existsSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from '@playwright/test'
import {
  SERVER_SECTIONS,
  SHARED_SECTIONS,
  WS_SECTIONS,
} from '../src/lib/routes.ts'

const PORT = 8972
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..')
const THEMES = ['light', 'dark']
const WIDTHS = [
  { width: 1440, height: 900 },
  { width: 390, height: 844 },
]

/** Same interpreter choice as playwright.config.ts's e2ePython. */
function python() {
  if (process.env.DW_E2E_PYTHON) return process.env.DW_E2E_PYTHON
  for (const candidate of ['venv/bin/python', '.venv/bin/python']) {
    if (existsSync(join(ROOT, candidate))) return join(ROOT, candidate)
  }
  return 'python3'
}

export function captureRoutes() {
  const route = (name, hash) => ({ name, hash })
  return [
    ...WS_SECTIONS.map((s) => route(`ws-${s}`, `ws/default/${s}`)),
    route('ws-workflows-e2e-card', 'ws/default/workflows/e2e-card'),
    route('ws-edit-e2e-card', 'ws/default/edit/e2e-card'),
    route('ws-requests-e2e-launch', 'ws/default/requests/e2e-launch'),
    route('ws-jobs-e2erestart01', 'ws/default/jobs/e2erestart01'),
    ...SHARED_SECTIONS.map((s) => route(`shared-${s}`, `shared/${s}`)),
    ...SERVER_SECTIONS.map((s) => route(`server-${s}`, `server/${s}`)),
  ]
}

async function waitForHealth(url, child) {
  const deadline = Date.now() + 90_000
  while (Date.now() < deadline) {
    if (child.exitCode !== null) throw new Error('fixture server exited early')
    try {
      if ((await fetch(url)).ok) return
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 500))
  }
  throw new Error('fixture server did not become healthy in 90 s')
}

async function main(argv) {
  const out = argv[0] ?? mkdtempSync(join(tmpdir(), 'dw-ux-capture-'))
  const base = `http://127.0.0.1:${PORT}`
  const server = spawn(
    python(),
    ['ui/e2e/serve_fixture.py', '--port', String(PORT)],
    {
      cwd: ROOT,
      env: { ...process.env, DW_STRICT_RESPONSES: '1' },
      stdio: 'ignore',
    },
  )
  let browser
  let count = 0
  try {
    await waitForHealth(`${base}/api/health`, server)
    browser = await chromium.launch()
    for (const theme of THEMES) {
      for (const size of WIDTHS) {
        const context = await browser.newContext({
          viewport: size,
          colorScheme: theme,
        })
        await context.addInitScript((t) => {
          try {
            localStorage.setItem('dw-theme', t)
          } catch {
            /* fine */
          }
          document.documentElement.dataset.theme = t
        }, theme)
        const page = await context.newPage()
        for (const { name, hash } of captureRoutes()) {
          await page.goto(`${base}/#/${hash}`)
          await page.waitForLoadState('networkidle')
          await page.waitForTimeout(400)
          await page.screenshot({
            path: join(out, `${name}-${theme}-${size.width}.png`),
          })
          count++
        }
        await context.close()
      }
    }
  } finally {
    if (browser) await browser.close()
    server.kill()
  }
  console.log(out)
  console.error(`${count} screenshots`)
  return 0
}

if (process.argv[1] === fileURLToPath(import.meta.url))
  process.exit(await main(process.argv.slice(2)))
