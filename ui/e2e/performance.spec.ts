import { gzipSync } from 'node:zlib'
import { expect, test, type Page, type Response } from '@playwright/test'

// The UI's weight on the wire, measured in a real browser against the built
// UI: the bytes the server sent for a response body (as compressed; a KB is 1,000 bytes), summed
// over the responses of one load in a fresh context. Sizes, never timings, so
// a budget cannot flake. A budget is raised only with Don.
/** The entry bundle (index-*.js + index-*.css). Today 282 KB. */
const ENTRY_BUDGET_KB = 300
/** The Overview's first load, every response. Today 356 KB. */
const FIRST_LOAD_BUDGET_KB = 420
/** The JSON view's first open, every response. Today 1,351 KB. Don set 1,400
 * (2026-11-24) in place of the plan's 1,250: every Monaco feature stays. */
const JSON_VIEW_BUDGET_KB = 1400

const OVERVIEW = '/#/ws/default/overview'
const ENTRY = /\/assets\/index-[^/]+\.(js|css)$/
const ENTRY_JS = '**/assets/index-*.js'

/** Record each response a page receives, from now on */
function recordResponses(page: Page): Response[] {
  const seen: Response[] = []
  page.on('response', (response) => seen.push(response))
  return seen
}

/** Encoded body bytes of `responses`, once the browser has all of them. A
 * worker's script is fetched from the worker, where the browser reports no
 * size: its decoded body is gzipped at the server's level (6) instead */
async function wireBytes(responses: Response[]): Promise<number> {
  let total = 0
  for (const response of responses) {
    const sizes = await response.request().sizes()
    total +=
      sizes.responseBodySize ||
      (response.status() === 200
        ? gzipSync(await response.body(), { level: 6 }).length
        : 0)
  }
  return total
}

const kb = (bytes: number) => bytes / 1000

test("the UI's files are compressed and cached", async ({ page }) => {
  const seen = recordResponses(page)
  const root = await page.goto('/')
  expect(root?.headers()['cache-control']).toContain('no-cache')
  await page.waitForLoadState('networkidle')
  const js = seen.find((r) =>
    /\/assets\/index-[^/]+\.js$/.test(new URL(r.url()).pathname),
  )
  expect(js, 'the entry JS was loaded').toBeDefined()
  expect(js?.headers()['content-encoding']).toBe('gzip')
  expect(js?.headers()['cache-control']).toContain('immutable')
})

test('the entry bundle stays within budget', async ({ page }) => {
  const seen = recordResponses(page)
  await page.goto(OVERVIEW)
  await page.waitForLoadState('networkidle')
  const entry = seen.filter((r) => ENTRY.test(new URL(r.url()).pathname))
  expect(entry.map((r) => r.url().split('.').pop()).sort()).toEqual([
    'css',
    'js',
  ])
  const total = kb(await wireBytes(entry))
  console.log(`entry bundle: ${total.toFixed(1)} KB`)
  expect(total).toBeLessThanOrEqual(ENTRY_BUDGET_KB)
})

test('a first load of the Overview stays within budget', async ({ page }) => {
  const seen = recordResponses(page)
  await page.goto(OVERVIEW)
  await page.waitForLoadState('networkidle')
  const total = kb(await wireBytes(seen))
  console.log(`overview first load: ${total.toFixed(1)} KB`)
  expect(total).toBeLessThanOrEqual(FIRST_LOAD_BUDGET_KB)
})

test('the JSON view stays within budget and loads only the JSON language', async ({
  page,
}) => {
  await page.goto('/#/ws/default/edit/models/z-image')
  const json = page.getByRole('button', { name: 'JSON', exact: true })
  await expect(json).toBeVisible()
  await page.waitForLoadState('networkidle')
  const seen = recordResponses(page)
  const schema = page.waitForResponse((r) => r.url().endsWith('/api/schema'))
  await json.click()
  await expect(page.locator('.monaco-editor').first()).toBeVisible({
    timeout: 20_000,
  })
  await schema
  // the schema marker: a key the schema does not know is underlined, which
  // takes the JSON language's worker; the icon font loads with the first icon
  await page.locator('.view-lines').first().click()
  await page.keyboard.press('ControlOrMeta+a')
  await page.keyboard.insertText('{"id":"X","stepz":[]}')
  await expect(page.locator('.squiggly-warning').first()).toBeVisible({
    timeout: 20_000,
  })
  await page.evaluate(() => document.fonts.load('16px codicon'))
  await page.waitForLoadState('networkidle')
  const names = seen.map((r) => new URL(r.url()).pathname)
  for (const other of [
    'ts.worker',
    'css.worker',
    'html.worker',
    'typescript',
    'cssMode',
    'htmlMode',
  ]) {
    expect(
      names.filter((n) => n.includes(other)),
      other,
    ).toEqual([])
  }
  const total = kb(await wireBytes(seen))
  console.log(`json view first open: ${total.toFixed(1)} KB`)
  expect(total).toBeLessThanOrEqual(JSON_VIEW_BUDGET_KB)
})

test('the loading shell paints before the bundle', async ({ page }) => {
  await page.route(ENTRY_JS, async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 2000))
    await route.continue()
  })
  await page.goto(OVERVIEW, { waitUntil: 'commit' })
  const boot = page.locator('#boot')
  await expect(boot).toContainText('Loading dw…')
  const paints = await boot.evaluate((el) => {
    const probe = document.createElement('div')
    probe.style.background = 'var(--bg)'
    document.body.append(probe)
    const bg = getComputedStyle(probe).backgroundColor
    probe.remove()
    return { shell: getComputedStyle(el).backgroundColor, bg }
  })
  expect(paints.shell).toBe(paints.bg)
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
  await expect(page.locator('#boot')).toHaveCount(0)
})

test('the shell says the app did not load when the bundle never arrives', async ({
  page,
}) => {
  test.setTimeout(45_000)
  await page.route(ENTRY_JS, () => new Promise<void>(() => {})) // never answered
  await page.goto(OVERVIEW, { waitUntil: 'commit' })
  const failed = page.locator('#boot .failed')
  await expect(failed).toBeVisible({ timeout: 20_000 })
  await expect(failed).toContainText("dw didn't load.")
  await expect(page.locator('#boot .loading')).toBeHidden()
})

test('a normal load never shows the did-not-load line', async ({ page }) => {
  await page.goto(OVERVIEW)
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
  await expect(page.locator('#boot')).toHaveCount(0)
  await expect(page.getByText("dw didn't load.")).toHaveCount(0)
})
