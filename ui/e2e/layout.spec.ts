import { expect, test, type Locator, type Page } from '@playwright/test'

// Layout claims, which jsdom cannot check: it lays nothing out and applies no
// component styles. The component tests hold which element gets which class;
// what the class then does is measured here, in a real browser.

/** A 200 by 2000 picture: taller than any window the specs use */
const TALL =
  '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="2000">' +
  '<rect width="200" height="2000" fill="#888"/></svg>'

/** Put markup carrying the global tile classes (patterns.css) on the page */
async function mount(page: Page, html: string): Promise<Locator> {
  await page.evaluate((markup) => {
    const probe = document.createElement('div')
    probe.id = 'probe'
    probe.style.cssText =
      'position:fixed;left:0;top:0;width:400px;z-index:9999;background:var(--bg)'
    probe.innerHTML = markup
    document.body.append(probe)
  }, html)
  return page.locator('#probe')
}

const box = async (target: Locator) => {
  const found = await target.boundingBox()
  expect(found).not.toBeNull()
  return found!
}

test('tiles: a row is 32 px, a mini 48, an audio cover 72', async ({
  page,
}) => {
  await page.goto('/')
  await expect(page.getByRole('main')).toBeVisible()
  const probe = await mount(
    page,
    '<div class="filetile tile-row"></div>' +
      '<div class="filetile tile-mini"></div>' +
      '<div class="filetile tile-mini kind-audio"></div>',
  )
  const tiles = probe.locator('.filetile')
  expect((await box(tiles.nth(0))).width).toBe(32)
  expect((await box(tiles.nth(1))).width).toBe(48)
  expect((await box(tiles.nth(2))).width).toBe(72)
})

test('tiles: a picture is letterboxed, never cropped', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('main')).toBeVisible()
  const probe = await mount(page, '<img class="tile-pic" alt="" />')
  await expect(probe.locator('img')).toHaveCSS('object-fit', 'contain')
})

test('tiles: active, hover and picked look different', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('main')).toBeVisible()
  const probe = await mount(
    page,
    ['', 'picked', 'active', '']
      .map(
        (state) =>
          `<button class="filetile ${state}" style="width:60px;height:60px"></button>`,
      )
      .join(''),
  )
  const looks = (n: number) =>
    probe
      .locator('button')
      .nth(n)
      .evaluate((el) => {
        const s = getComputedStyle(el)
        return {
          shadow: s.boxShadow,
          outline: `${s.outlineStyle} ${s.outlineWidth}`,
          background: s.backgroundColor,
        }
      })
  const plain = await looks(0)
  const picked = await looks(1)
  const active = await looks(2)
  // picked draws an edge, active an outline clear of it
  expect(picked.shadow).not.toBe(plain.shadow)
  expect(picked.outline).toBe(plain.outline)
  expect(active.outline).toBe('solid 2px')
  // hover tints the panel and draws no edge or outline
  await probe.locator('button').nth(3).hover()
  await expect
    .poll(async () => (await looks(3)).background)
    .not.toBe(plain.background)
  const hover = await looks(3)
  expect(hover.shadow).toBe(plain.shadow)
  expect(hover.outline).toBe(plain.outline)
})

test('viewer: a tall picture fits the window with its controls on it', async ({
  page,
}) => {
  await page.route('**/outputs/**e2e-one.png*', (route) =>
    route.fulfill({ contentType: 'image/svg+xml', body: TALL }),
  )
  await page.goto('/#/ws/default/gallery?open=e2e-one.png')
  const sheet = page.getByRole('dialog', { name: 'e2e-one.png' })
  const img = sheet.locator('img.media-fit')
  await expect(img).toBeVisible()
  await expect
    .poll(() => img.evaluate((el: HTMLImageElement) => el.naturalHeight))
    .toBe(2000)
  const window = page.viewportSize()!
  const picture = await box(img)
  expect(picture.y).toBeGreaterThanOrEqual(0)
  expect(picture.y + picture.height).toBeLessThanOrEqual(window.height)
  const close = await box(sheet.getByRole('button', { name: 'Close' }))
  expect(close.y).toBeGreaterThanOrEqual(0)
  expect(close.y + close.height).toBeLessThanOrEqual(picture.y + 1)
})

/** The Hero poster's row, shut: the page may open it by itself */
async function shutRow(page: Page) {
  await page.goto('/#/ws/default/requests/e2e-launch')
  const title = page.locator('.titleline button.title', {
    hasText: 'Hero poster',
  })
  await expect(title).toBeVisible()
  if ((await title.getAttribute('aria-expanded')) === 'true')
    await title.click()
  await expect(title).toHaveAttribute('aria-expanded', 'false')
  const row = page.locator('.row', { has: title })
  return { title, row }
}

test("requests: an open row's proof cell takes no room", async ({ page }) => {
  const { title, row } = await shutRow(page)
  const proof = row.locator('.proofcell')
  await expect(proof).toBeVisible()
  const shut = await box(title)
  await title.click()
  await expect(title).toHaveAttribute('aria-expanded', 'true')
  // the takes below show the picture: the title moves into the cell's track
  await expect(proof).toBeHidden()
  expect((await box(title)).x).toBeLessThan(shut.x)
})

test('requests: in a narrow panel the chip goes under the title', async ({
  page,
}) => {
  const { row } = await shutRow(page)
  const chip = row.locator('.chipcell')
  await expect(chip).toBeVisible()
  // under the title column: below it, and starting left of where it ends
  const below = async () => {
    const t = await box(row.locator('.main'))
    const c = await box(chip)
    return c.y >= t.y + t.height - 1 && c.x < t.x + t.width
  }
  // the rule asks about the panel, not the window: set the panel's width
  const panel = (width: number) =>
    page.locator('.panel.list').evaluate((el: HTMLElement, w) => {
      el.style.width = el.style.maxWidth = `${w}px`
    }, width)
  await panel(1000)
  await expect.poll(below).toBe(false)
  await panel(800)
  await expect.poll(below).toBe(true)
})

test('overview: a long job id is clipped inside its row', async ({ page }) => {
  await page.goto('/#/ws/default')
  const jid = page.locator('.jobs .when .jid').first()
  await expect(jid).toBeVisible()
  await jid.evaluate((el) => (el.textContent = 'e2e' + 'x'.repeat(300)))
  await expect(jid).toHaveCSS('text-overflow', 'ellipsis')
  expect(await jid.evaluate((el) => el.scrollWidth > el.clientWidth)).toBe(true)
  const row = await box(
    page.locator('.jobs li', { has: page.locator('.jid') }).first(),
  )
  const clipped = await box(jid)
  expect(clipped.x + clipped.width).toBeLessThanOrEqual(row.x + row.width + 1)
})

test("models: the phone rule hides only the table's own wide columns", async ({
  page,
}) => {
  await page.setViewportSize({ width: 400, height: 800 })
  await page.goto('/#/models')
  const column = page.locator('.datatable th.wideonly').first()
  await expect(column).toBeAttached()
  await expect(column).toBeHidden()
  // the page's CSS stays loaded for the rest of the app: it must not reach
  // a .wideonly of anyone else's
  const probe = await mount(page, '<div class="wideonly">elsewhere</div>')
  await expect(probe.locator('.wideonly')).toBeVisible()
})
