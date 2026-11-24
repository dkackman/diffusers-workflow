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

/** Encoded body bytes of `responses`, once the browser has all of them.
 * Where the browser reports a size it is used as is. A worker's script is
 * fetched from the worker, where it reports none: that body is then gzipped
 * at the server's level (6) - but only after the response is shown to have
 * been served gzip (a body of 1 KB or more, as the server's threshold), so a
 * server that stopped compressing fails here instead of being priced as if
 * it had not. A body under 1 KB is never compressed and counts as is. */
async function wireBytes(responses: Response[]): Promise<number> {
  let total = 0
  for (const response of responses) {
    const sizes = await response.request().sizes()
    if (sizes.responseBodySize) {
      total += sizes.responseBodySize
      continue
    }
    if (response.status() !== 200) continue
    const body = await response.body()
    if (body.length < 1000) {
      total += body.length
      continue
    }
    expect(
      response.headers()['content-encoding'],
      `${response.url()} (no transfer size reported) was served compressed`,
    ).toBe('gzip')
    total += gzipSync(body, { level: 6 }).length
  }
  return total
}

/** The path of every response, for the floors: a budget that counted nothing
 * (or missed the files it is about) must fail, not pass */
const paths = (responses: Response[]) =>
  responses.map((r) => new URL(r.url()).pathname)

/** The JSON editor's own Monaco-family chunks, by unhashed name */
const JSON_EDITOR_CHUNKS = [
  'monaco',
  'jsonMode',
  'editor.worker',
  'json.worker',
]

/** A hashed asset's path is `/assets/<chunk>-<hash>.<ext>`. Vite's hash may
 * hold `-` and `_`, so a file is matched by its known chunk name as a prefix
 * and never by stripping a hash off its tail */
const isChunk = (path: string, chunk: string, ext: string) =>
  path.startsWith(`/assets/${chunk}-`) &&
  new RegExp(`^[\\w-]+\\.${ext}$`).test(path.slice(`/assets/${chunk}-`.length))

/** `names` has a file of this chunk's name (the hash varies per build) */
const hasChunk = (names: string[], chunk: string, ext: string) =>
  names.some((n) => isChunk(n, chunk, ext))

test('chunk names are matched whatever characters the hash holds', () => {
  for (const hash of ['Ab12Cd34', 'Ab-12_Cd', '-a-b-c-', '_x_']) {
    expect(isChunk(`/assets/monaco-${hash}.js`, 'monaco', 'js')).toBe(true)
    expect(
      isChunk(`/assets/editor.worker-${hash}.js`, 'editor.worker', 'js'),
    ).toBe(true)
  }
  expect(isChunk('/assets/monaco-x.css', 'monaco', 'js')).toBe(false)
  expect(isChunk('/assets/tsMode-x.js', 'monaco', 'js')).toBe(false)
  expect(isChunk('/assets/monaco-x/y.js', 'monaco', 'js')).toBe(false)
})

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
  expect(total, 'the entry bundle was counted').toBeGreaterThan(20)
  expect(total).toBeLessThanOrEqual(ENTRY_BUDGET_KB)
})

test('a first load of the Overview stays within budget', async ({ page }) => {
  const seen = recordResponses(page)
  await page.goto(OVERVIEW)
  await page.waitForLoadState('networkidle')
  const names = paths(seen)
  expect(hasChunk(names, 'index', 'js'), 'the entry JS was counted').toBe(true)
  expect(hasChunk(names, 'index', 'css'), 'the entry CSS was counted').toBe(
    true,
  )
  expect(
    names.some((n) => /^\/api\/workflows(\/?|\?.*)$/.test(n)),
    `the Overview's workflows read was counted (${names.join(', ')})`,
  ).toBe(true)
  const total = kb(await wireBytes(seen))
  console.log(`overview first load: ${total.toFixed(1)} KB`)
  // The API reads ride on top of the entry bundle, whose size follows the
  // fixture's contents (earlier specs may save into it), so the 15% margin
  // under the budget covers that and no more. The floor only proves the
  // measurement saw response bodies (the by-name checks above prove it saw the
  // right files); it stays far below today's size so a smaller bundle passes
  expect(total, 'the first load was counted').toBeGreaterThan(50)
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
  const names = paths(seen)
  // what this open must have loaded, by chunk name
  for (const [chunk, ext] of [
    ['monaco', 'js'],
    ['monaco', 'css'],
    ['jsonMode', 'js'],
    ['editor.worker', 'js'],
    ['json.worker', 'js'],
    ['codicon', 'ttf'],
  ]) {
    expect(hasChunk(names, chunk, ext), `${chunk}.${ext} was counted`).toBe(
      true,
    )
  }
  expect(names, 'the schema was counted').toContain('/api/schema')
  // a chunk's hash changes per build, so a file is judged by its name: every
  // Monaco-family file (a language mode, a worker, the typescript service)
  // the page fetched must be one of the JSON editor's own. Another language's
  // mode or worker, whatever its hash, is a regression
  const family = names.filter(
    (n) =>
      n.startsWith('/assets/') &&
      /mode|worker|typescript|tsserver|lib\./i.test(n.slice(8)),
  )
  expect(
    family.filter(
      (n) =>
        !JSON_EDITOR_CHUNKS.some(
          (c) => isChunk(n, c, 'js') || isChunk(n, c, 'css'),
        ),
    ),
  ).toEqual([])
  const total = kb(await wireBytes(seen))
  console.log(`json view first open: ${total.toFixed(1)} KB`)
  // a sanity floor, far below today's size: the by-name checks above prove
  // the right files were seen, so a leaner editor still passes
  expect(total, 'the JSON view was counted').toBeGreaterThan(300)
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
