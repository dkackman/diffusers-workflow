// Screenshots every route of the UI, in both themes at two widths, against the
// e2e fixture server, so a reviewer sees the whole UI in one folder.
//
//   npm run ux:capture [OUT]      (run `npm run build` first; ui/dist is served)
//
// The server runs in its own process group, stopped on exit, SIGINT or SIGTERM.
// Files are <name>-<theme>-<width>.png. Port 8972 (e2e uses 8971).
import { spawn } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync } from 'node:fs'
import { connect } from 'node:net'
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

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

/** True when anything accepts a TCP connection on the port. */
function listening(port) {
  return new Promise((resolve) => {
    const socket = connect({ port, host: '127.0.0.1' })
    socket.once('connect', () => (socket.destroy(), resolve(true)))
    socket.once('error', () => resolve(false))
  })
}

async function healthy(url) {
  try {
    return (await fetch(url)).ok
  } catch {
    return false
  }
}

function groupAlive(pid) {
  try {
    process.kill(-pid, 0)
    return true
  } catch {
    return false
  }
}

/** SIGTERM the server's process group, wait up to 5 s, then SIGKILL it. The
 *  group holds dw.serve's worker children, which outlive a lone kill. */
async function stopGroup(child) {
  const { pid } = child
  if (pid === undefined) return
  const signal = (sig) => {
    try {
      process.kill(-pid, sig)
    } catch {
      /* already gone */
    }
  }
  signal('SIGTERM')
  const deadline = Date.now() + 5000
  while (groupAlive(pid) && Date.now() < deadline) await sleep(100)
  if (groupAlive(pid)) signal('SIGKILL')
  while (groupAlive(pid)) await sleep(50)
}

/** Spawn the fixture server in its own process group, keeping the last lines
 *  of its output for an error message. */
function startServer() {
  const tail = []
  const keep = (chunk) => {
    tail.push(...String(chunk).split('\n').filter(Boolean))
    if (tail.length > 40) tail.splice(0, tail.length - 40)
  }
  const child = spawn(
    python(),
    ['ui/e2e/serve_fixture.py', '--port', String(PORT)],
    {
      cwd: ROOT,
      env: { ...process.env, DW_STRICT_RESPONSES: '1' },
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: true,
    },
  )
  child.stdout.on('data', keep)
  child.stderr.on('data', keep)
  const failed = new Promise((_, reject) => {
    child.on('error', (e) =>
      reject(new Error(`could not start ${python()}: ${e.message}`)),
    )
  })
  failed.catch(() => {})
  return { child, tail, failed }
}

async function waitForHealth(url, { child, tail, failed }) {
  const deadline = Date.now() + 90_000
  while (Date.now() < deadline) {
    if (child.exitCode !== null || child.signalCode !== null)
      throw new Error(`fixture server exited early:\n${tail.join('\n')}`)
    if (await Promise.race([healthy(url), failed])) return
    await sleep(500)
  }
  throw new Error(
    `fixture server did not become healthy in 90 s:\n${tail.join('\n')}`,
  )
}

/** Load the route, then wait until nothing is loading: no aria-busy region and
 *  no loading skeleton. A page that never settles is shot as it is. */
async function settle(page, base, hash) {
  await page.goto(`${base}/#/${hash}`, { waitUntil: 'load' })
  await page
    .waitForFunction(
      () =>
        !document.querySelector(
          '[aria-busy="true"], [role="status"][aria-label="loading"]',
        ),
      undefined,
      { timeout: 10_000 },
    )
    .catch(() => {})
  await page.waitForLoadState('networkidle').catch(() => {})
}

async function shoot(browser, base, out) {
  let count = 0
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
        await settle(page, base, hash)
        await page.screenshot({
          path: join(out, `${name}-${theme}-${size.width}.png`),
        })
        count++
      }
      await context.close()
    }
  }
  return count
}

async function main(argv) {
  const base = `http://127.0.0.1:${PORT}`
  if (await listening(PORT)) {
    console.error(
      `Something already listens on port ${PORT}. Stop it first, or a stale server would be captured.`,
    )
    return 1
  }
  const out = argv[0] ?? mkdtempSync(join(tmpdir(), 'dw-ux-capture-'))
  mkdirSync(out, { recursive: true })
  const server = startServer()
  let browser
  let stopping
  const cleanup = () => {
    stopping ??= (async () => {
      if (browser) await browser.close().catch(() => {})
      await stopGroup(server.child)
    })()
    return stopping
  }
  for (const sig of ['SIGINT', 'SIGTERM'])
    process.once(sig, () => cleanup().finally(() => process.exit(130)))
  process.once('SIGHUP', () => cleanup().finally(() => process.exit(129)))
  let count
  try {
    await waitForHealth(`${base}/api/health`, server)
    // Playwright's own signal handlers would exit before the server is stopped.
    browser = await chromium.launch({
      handleSIGINT: false,
      handleSIGTERM: false,
      handleSIGHUP: false,
    })
    count = await shoot(browser, base, out)
  } finally {
    await cleanup()
  }
  console.log(out)
  console.error(`${count} screenshots`)
  return 0
}

if (process.argv[1] === fileURLToPath(import.meta.url))
  process.exit(await main(process.argv.slice(2)))
