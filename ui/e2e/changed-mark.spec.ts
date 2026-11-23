import { expect, test, type Locator } from '@playwright/test'

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

test("a changed request item shows the row's inset bar, uncovered", async ({
  page,
}) => {
  await page.goto('/#/ws/default/requests/e2e-launch')
  const item = page.locator('li.item').first()
  await expect(item).toBeVisible()
  await item.evaluate((node) => node.classList.add('row-changed'))
  const seen = await item.evaluate((li) => ({
    shadow: getComputedStyle(li).boxShadow,
    // the row inside it paints nothing over the bar
    row: getComputedStyle(li.querySelector('.row')!).backgroundColor,
  }))
  expect(seen.shadow).toMatch(/3px 0px 0px 0px inset/)
  expect(seen.row).toBe('rgba(0, 0, 0, 0)')
})
