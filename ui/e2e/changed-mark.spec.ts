import { expect, test, type Locator, type Page } from '@playwright/test'

// A quiet reload marks what it changed (`.row-changed`). jsdom does not
// paint, so the check that the mark is visible runs here, on the real
// stylesheets. Tiles and cards put a picture over their own inset shadow, so
// the mark is an overlay bar (frames.css); this reads that bar's computed
// style. The class is set by hand: what sets it is unit-tested, and a real
// quiet reload would have to be caught inside its 4 s fade.

async function bar(el: Locator) {
  await el.evaluate((node) => node.classList.add('row-changed'))
  return el.evaluate((node) => {
    const s = getComputedStyle(node, '::after')
    return {
      content: s.content,
      position: s.position,
      width: s.width,
      height: s.height,
      background: s.backgroundColor,
      opacity: Number(s.opacity),
      host: node.getBoundingClientRect().height,
    }
  })
}

function expectVisible(b: Awaited<ReturnType<typeof bar>>) {
  expect(b.content).not.toBe('none')
  expect(b.position).toBe('absolute')
  expect(b.width).toBe('3px')
  // the host's border sits outside the bar's box
  expect(parseFloat(b.height)).toBeGreaterThan(b.host - 4)
  expect(b.background).not.toBe('rgba(0, 0, 0, 0)')
  expect(b.opacity).toBeGreaterThan(0.5)
}

test('a changed gallery tile shows the bar over its picture, picked or not', async ({
  page,
}) => {
  await page.goto('/#/gallery')
  const tile = page.locator('.cellwrap').first()
  await expect(tile).toBeVisible()
  expectVisible(await bar(tile))
  await tile.locator('input.pick').check({ force: true })
  await expect(tile).toHaveClass(/picked/)
  expectVisible(await bar(tile))
  await page.screenshot({
    path: 'test-results/changed-gallery-tile.png',
    clip: (await tile.boundingBox())!,
  })
})

test('a changed workflow card shows the bar over its picture', async ({
  page,
}) => {
  await page.goto('/#/ws/default/workflows')
  const card = page.locator('a.card').first()
  await expect(card).toBeVisible()
  expectVisible(await bar(card))
  await page.screenshot({
    path: 'test-results/changed-workflow-card.png',
    clip: (await card.boundingBox())!,
  })
})

test("an asset tile's change mark shows", async ({ page }) => {
  await page.goto('/#/ws/default/assets')
  const tile = page.locator('.cellwrap', { hasText: 'e2e-proof-asset' })
  await expect(tile).toBeVisible()
  await expect(tile.locator('img')).toBeVisible()
  expectVisible(await bar(tile))
})

test("a workflow card's change mark shows", async ({ page }) => {
  await page.goto('/#/ws/default/workflows')
  const card = page.locator('a.card', { hasText: 'card-proof' })
  await expect(card).toBeVisible()
  await expect(card.locator('img')).toBeVisible()
  expectVisible(await bar(card))
})

test('a changed request item shows the mark over its row, uncovered', async ({
  page,
}) => {
  await page.goto('/#/ws/default/requests/e2e-launch')
  const item = page.locator('li.item').first()
  await expect(item).toBeVisible()
  expectVisible(await bar(item))
  // the row inside it paints nothing over the bar
  expect(
    await item
      .locator('.row')
      .evaluate((row) => getComputedStyle(row).backgroundColor),
  ).toBe('rgba(0, 0, 0, 0)')
})

test('a live row keeps its bar while marked', async ({ page }) => {
  await page.goto('/#/ws/default/requests/e2e-launch')
  const row = page.locator('li.item .row').first()
  await expect(row).toBeVisible()
  await row.evaluate((node) => node.classList.add('row-live'))
  const marked = await bar(row)
  expectVisible(marked)
  const shadow = await row.evaluate((node) => getComputedStyle(node).boxShadow)
  expect(shadow).toMatch(/2px 0px 0px 0px inset/)
})

test("a tile's mark is one bar", async ({ page }) => {
  await page.goto('/#/gallery')
  const tile = page.locator('.cellwrap').first()
  await expect(tile).toBeVisible()
  expectVisible(await bar(tile))
  const shadow = await tile.evaluate((node) => getComputedStyle(node).boxShadow)
  expect(shadow).toBe('none')
})

// Two real changes to one card, 1.2 s apart (inside the 4 s mark). The
// second must restart the fade: the class goes off for a rendered frame and
// on again, so the ::after animation is a new Animation starting at zero.
// The change feed polls every 5 s; a tab that hides and shows polls at once,
// so the spec asks for it that way instead of waiting.

const SCRATCH = 'e2e-restart-mark'

async function pollFeed(page: Page) {
  await page.evaluate(() => {
    const state = (v: string) => {
      Object.defineProperty(document, 'visibilityState', {
        configurable: true,
        get: () => v,
      })
      document.dispatchEvent(new Event('visibilitychange'))
    }
    state('hidden')
    state('visible')
  })
}

function fade(card: Locator) {
  return card.evaluate((node) => {
    const w = window as unknown as { __fade?: Animation }
    const a = node
      .getAnimations({ subtree: true })
      .find((x) => (x.effect as KeyframeEffect).pseudoElement === '::after')
    if (!a) return null
    const same = w.__fade === a
    w.__fade = a
    return { same, time: Number(a.currentTime) }
  })
}

test('a card that changes twice flashes again: its fade restarts', async ({
  page,
  request,
}) => {
  const put = (description: string) =>
    request.put(`/api/workflows/${SCRATCH}`, {
      data: {
        workflow: {
          id: 'E2ERestartMark',
          description,
          variables: { line: 'x' },
          steps: [
            {
              name: 'card',
              task: {
                command: 'compose_text',
                arguments: { parts: ['variable:line'] },
              },
              result: { content_type: 'text/plain' },
            },
          ],
        },
      },
    })
  await page.goto('/#/ws/default/workflows')
  await expect(page.locator('a.card').first()).toBeVisible()
  const card = page.locator('a.card', { hasText: SCRATCH })
  try {
    expect((await put('first')).status()).toBe(200)
    await pollFeed(page)
    await expect(card).toHaveClass(/row-changed/, { timeout: 10_000 })
    await page.waitForTimeout(1200)
    const first = await fade(card)
    expect(first?.time).toBeGreaterThan(1000)

    expect((await put('second')).status()).toBe(200)
    await pollFeed(page)
    await expect
      .poll(async () => (await fade(card))?.same, { timeout: 10_000 })
      .toBe(false)
    const second = await fade(card)
    expect(second?.time).toBeLessThan(1000)
  } finally {
    await request.delete(`/api/workflows/${SCRATCH}`)
  }
})
