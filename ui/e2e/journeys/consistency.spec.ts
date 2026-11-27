import { expect, test, type Locator, type Page } from '../test'

// The journeys of the consistency phase, end to end against the real server:
// every page names itself the way the sidebar does, an unknown route is a
// page of its own, a list's filters and place survive opening something and
// going Back (even while the list reloads late), and what can be put back is
// undone while what destroys asks. Stars are set and cleared through the
// HTTP API, so the seeded gallery is left as the fixture made it.
const GALLERY = '/#/ws/default/gallery'
const BASE_TITLE = 'diffusers-workflow'
// Two of the fixture's run files: the gallery shows them under Final
const STILLS = [
  'e2e-run-group/20260906-101612-ffffffff/still.png',
  'e2e-run-group/20260906-101500-a1b2c3d4/still.png',
]
// The height a narrowed, starred gallery scrolls in: its ten run files (the
// fixture's six and the four of e2e-find, all of which `e2e` finds)
// stand taller than this viewport
const SHORT = { width: 1000, height: 500 }

type Request = Page['request']

async function galleryNames(request: Request): Promise<string[]> {
  const read = await request.get('/api/gallery')
  expect(read.status()).toBe(200)
  return (await read.json()).files.map((f: { name: string }) => f.name)
}

async function star(request: Request, name: string, starred: boolean) {
  const set = await request.put('/api/stars', { data: { name, starred } })
  expect(set.ok()).toBe(true)
}

async function isStarred(request: Request, name: string): Promise<boolean> {
  const read = await request.get('/api/gallery')
  const files: { name: string; starred?: boolean }[] = (await read.json()).files
  return !!files.find((f) => f.name === name)?.starred
}

const scrollY = (page: Page) => page.evaluate(() => window.scrollY)

/** The gallery narrowed to `e2e`, starred only, and scrolled down by a
 * wheel the way a person scrolls; returns where it came to rest */
async function narrowAndScroll(page: Page, by = 400): Promise<number> {
  await page.setViewportSize(SHORT)
  await page.goto(GALLERY)
  await page.getByPlaceholder('search names, prompts and notes…').fill('e2e')
  await page.getByRole('button', { name: 'Starred', exact: true }).click()
  await expect(page).toHaveURL(/q=e2e/)
  await expect(page).toHaveURL(/starred=1/)
  // at least the fixture's ten: other specs' runs add to what `e2e` finds
  await expect
    .poll(() => page.locator('button.filetile').count())
    .toBeGreaterThanOrEqual(10)
  await page.mouse.move(600, 300)
  await page.mouse.wheel(0, by)
  await expect.poll(() => scrollY(page)).toBeGreaterThan(by - 50)
  return scrollY(page)
}

/** Open the two stills in Compare from the bulk bar. A dispatched click,
 * because Playwright's own scrolls the sticky bar into view first and would
 * move the page the test means to leave where it is */
async function compareStills(page: Page) {
  // a dispatched click skips the actionability a real one waits for
  const press = async (target: Locator) => {
    await expect(target).toBeVisible()
    await expect(target).toBeEnabled()
    await target.dispatchEvent('click')
  }
  for (const name of STILLS)
    await press(page.getByRole('checkbox', { name: `select ${name}` }))
  const before = await scrollY(page)
  await press(
    page.getByRole('button', {
      name: 'compare the selected files side by side',
    }),
  )
  await expect(page).toHaveURL(/\/compare\//)
  return before
}

/** Leave the tab and come back: what watches for changes asks at once then,
 * so a write made over the API is seen without waiting out its timer */
async function leaveAndReturn(page: Page) {
  await page.evaluate(() => {
    for (const state of ['hidden', 'visible']) {
      Object.defineProperty(document, 'visibilityState', {
        configurable: true,
        get: () => state,
      })
      document.dispatchEvent(new Event('visibilitychange'))
    }
  })
}

/** Every listing read from now on answers `ms` late, as a slow server does */
async function slowListing(page: Page, ms: number) {
  await page.route(
    (url) => url.pathname === '/api/gallery',
    async (route) => {
      await new Promise((resolve) => setTimeout(resolve, ms))
      await route.continue()
    },
  )
}

/** Watch the page's scroll from here: the lowest it went once it first left
 * the top, so a reload that throws it back to the top is seen even if the
 * page goes down again before the test looks */
async function watchScroll(page: Page) {
  await page.evaluate(() => {
    const w = window as unknown as { lowest: number | null }
    w.lowest = null
    window.addEventListener('scroll', () => {
      if (window.scrollY > 0 || w.lowest !== null)
        w.lowest = Math.min(w.lowest ?? window.scrollY, window.scrollY)
    })
  })
  return () =>
    page.evaluate(() => (window as unknown as { lowest: number | null }).lowest)
}

// Only a file in a run directory can carry a star
test.beforeEach(async ({ request }) => {
  for (const name of await galleryNames(request))
    if (name.includes('/')) await star(request, name, true)
})

test.afterEach(async ({ request }) => {
  for (const name of await galleryNames(request))
    if (name.includes('/')) await star(request, name, false)
})

test("each sidebar section's tab title and h1 are its sidebar label", async ({
  page,
}) => {
  // A page that names its subject rather than its section says so here:
  // the overview is the workspace's, the editor's is its document's, and
  // the shared Assets page qualifies the label
  const subject: Record<string, string> = {
    '#/ws/default/overview': 'default',
    '#/ws/default/edit': 'New workflow',
    '#/shared/assets': 'Shared assets',
  }
  await page.goto('/#/ws/default/jobs')
  const items = await page
    .locator('aside a[title]')
    .evaluateAll((links) =>
      links.map((a) => [a.getAttribute('href')!, a.getAttribute('title')!]),
    )
  expect(items.length).toBe(14)
  for (const [href, label] of items) {
    await page.locator(`aside a[title][href="${href}"]`).click()
    await expect(page).toHaveURL(new RegExp(`${href.replace(/\//g, '\\/')}$`))
    const h1 = subject[href] ?? label
    if (subject[href] && h1 !== 'default' && h1 !== 'New workflow')
      expect(h1.toLowerCase()).toContain(label.toLowerCase())
    await expect(page.locator('h1')).toHaveText(h1)
    await expect(page).toHaveTitle(`${h1} · ${BASE_TITLE}`)
  }
})

test('an unknown route is a page of its own, with a way to the overview', async ({
  page,
}) => {
  await page.goto('/#/ws/default/nope')
  await expect(
    page.getByRole('heading', { level: 1, name: 'Page not found' }),
  ).toBeVisible()
  await expect(
    page.getByText('There is no page at /ws/default/nope.'),
  ).toBeVisible()
  await expect(page).toHaveTitle(`Page not found · ${BASE_TITLE}`)
  await page.getByRole('link', { name: 'Go to the overview' }).click()
  await expect(page).toHaveURL(/#\/ws\/default\/overview$/)
  await expect(page.locator('h1')).toHaveText('default')
})

test('Back to a narrowed, scrolled gallery that loads late keeps its filters and place, through a quiet reload', async ({
  page,
  request,
}) => {
  await narrowAndScroll(page)
  const left = await compareStills(page)
  expect(left).toBeGreaterThan(300)
  // the list answers late
  await slowListing(page, 2000)
  await page.goBack()
  const lowest = await watchScroll(page)
  await expect(page).toHaveURL(/q=e2e/)
  await expect(page).toHaveURL(/starred=1/)
  await expect(
    page.getByPlaceholder('search names, prompts and notes…'),
  ).toHaveValue('e2e')
  await expect(
    page.getByRole('button', { name: 'Starred', exact: true }),
  ).toHaveAttribute('aria-pressed', 'true')
  await expect.poll(() => scrollY(page), { timeout: 8_000 }).toBe(left)
  // a write on the server makes the page read the list again, quietly, and
  // it stays where it was put
  // the note's mark shows once the list read after the write is on screen
  const mark = 'e2e quiet reload'
  const note = (text: string | null) =>
    request.put('/api/notes', { data: { target: STILLS[0], note: text } })
  try {
    expect((await note(mark)).status()).toBe(200)
    await leaveAndReturn(page)
    await expect(page.getByTitle(mark, { exact: true })).toBeVisible()
    expect(await scrollY(page)).toBe(left)
    // and it never went back to the top on the way
    expect(await lowest()).toBeGreaterThan(0)
  } finally {
    await note(null)
  }
})

test('content shown above a list while it loads late does not end its restore', async ({
  page,
}) => {
  await narrowAndScroll(page)
  const left = await compareStills(page)
  // the list is held until the banner is up
  let release = () => {}
  const held = new Promise<void>((resolve) => (release = resolve))
  await page.route(
    (url) => url.pathname === '/api/gallery',
    async (route) => {
      await held
      await route.continue()
    },
  )
  await page.goBack()
  // the restore has gone as far as the loading page reaches
  await expect.poll(() => scrollY(page)).toBeGreaterThan(0)
  expect(await scrollY(page)).toBeLessThan(left)
  // the server goes quiet and the shell's banner grows above the list
  await page.route('**/api/**', (route) => route.abort())
  await expect(
    page.getByText("Can't reach the server. Retrying every few seconds."),
  ).toBeVisible({ timeout: 7_000 })
  // the banner neither ended the restore nor found scroll anchoring on (the
  // browser does not always anchor against this banner, so that is read off)
  expect(await scrollY(page)).toBeLessThan(left)
  expect(
    await page.evaluate(() => document.documentElement.style.overflowAnchor),
  ).toBe('none')
  // back again: the held list answers, and the restore, still going, ends
  await page.unroute('**/api/**')
  release()
  await expect.poll(() => scrollY(page), { timeout: 8_000 }).toBe(left)
  await expect
    .poll(() =>
      page.evaluate(() => document.documentElement.style.overflowAnchor),
    )
    .toBe('')
})

test("Back and Forward between two queries on the same path restore each entry's place", async ({
  page,
}) => {
  const first = await narrowAndScroll(page, 300)
  // a pushed query on the same path: a new entry, and the page stays put
  await page.evaluate(() => {
    location.hash = '#/ws/default/gallery?starred=1'
  })
  await expect(
    page.getByPlaceholder('search names, prompts and notes…'),
  ).toHaveValue('')
  expect(await scrollY(page)).toBe(first)
  await page.mouse.wheel(0, -200)
  await expect.poll(() => scrollY(page)).toBeLessThan(first - 150)
  const second = await scrollY(page)

  await page.goBack()
  await expect(
    page.getByPlaceholder('search names, prompts and notes…'),
  ).toHaveValue('e2e')
  await expect.poll(() => scrollY(page)).toBe(first)
  await page.goForward()
  await expect(
    page.getByPlaceholder('search names, prompts and notes…'),
  ).toHaveValue('')
  await expect.poll(() => scrollY(page)).toBe(second)
})

test('typing in the gallery filter leaves focus in the field', async ({
  page,
}) => {
  await page.goto(GALLERY)
  const filter = page.getByPlaceholder('search names, prompts and notes…')
  await filter.click()
  await page.keyboard.type('still', { delay: 50 })
  await expect(page).toHaveURL(/q=still/)
  await expect(filter).toBeFocused()
  await expect(filter).toHaveValue('still')
  await expect(page.locator('button.filetile')).toHaveCount(2)
  await expect(filter).toBeFocused()
})

test('a star is undone, not asked about', async ({ page, request }) => {
  const [name] = STILLS
  await star(request, name, false)
  await page.goto(GALLERY)
  await page.getByRole('button', { name: new RegExp(`^${name}`) }).click()
  const viewer = page.getByRole('dialog')
  await viewer
    .getByRole('button', { name: 'Star', exact: true, pressed: false })
    .click()
  await expect.poll(() => isStarred(request, name)).toBe(true)
  const toast = page
    .locator('[data-sonner-toast]')
    .filter({ hasText: 'Starred still.png' })
  await toast.getByRole('button', { name: 'Undo' }).click()
  await expect.poll(() => isStarred(request, name)).toBe(false)
  await expect(
    viewer.getByRole('button', { name: 'Star', exact: true, pressed: false }),
  ).toBeVisible()
})

test('deleting a starred file says it is starred and asks in danger', async ({
  page,
  request,
}) => {
  const [name] = STILLS
  await page.goto(GALLERY)
  await page.getByRole('button', { name: new RegExp(`^${name}`) }).click()
  await page
    .getByRole('dialog')
    .getByRole('button', { name: 'Delete', exact: true })
    .click()
  const dialog = page.getByRole('alertdialog')
  await expect(dialog).toContainText(`Delete ${name}?`)
  await expect(dialog).toContainText('This removes it on disk.')
  await expect(dialog).toContainText('It is starred.')
  await expect(
    dialog.getByRole('button', { name: 'Delete', exact: true }),
  ).toHaveClass(/danger/)
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  await expect(dialog).toBeHidden()
  expect(await galleryNames(request)).toContain(name)
})
