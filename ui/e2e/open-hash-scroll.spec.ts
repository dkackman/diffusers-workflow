import { expect, test, type Page } from '@playwright/test'

/** A link to a file below the fold lands on its tile (#1778). Real layout:
 * the unit test can only say scrollIntoView was asked for. */
async function opensOnItsTile(page: Page, view: string) {
  // short enough that the seeded files run past the fold
  await page.setViewportSize({ width: 1280, height: 300 })
  await page.goto(`/#/ws/default/${view}`)
  const picks = page.locator('.cellwrap .pick')
  await expect(picks.last()).toBeAttached()
  await expect(picks.last()).not.toBeInViewport()
  const label = await picks.last().getAttribute('aria-label')
  const name = label!.replace(/^select /, '')

  // a fresh page, as a link from elsewhere arrives
  await page.goto('about:blank')
  await page.goto(`/#/ws/default/${view}?open=${encodeURIComponent(name)}`)
  const tile = page.locator('.cellwrap .filetile.active')
  await expect(page.getByRole('dialog')).toBeVisible()
  await expect(tile).toBeInViewport()

  // closing the viewer leaves the page where it is
  await page.keyboard.press('Escape')
  await expect(page.getByRole('dialog')).toBeHidden()
  await expect(picks.last()).toBeInViewport()
}

test('a gallery open= scrolls to the tile', async ({ page }) => {
  await opensOnItsTile(page, 'gallery')
})

test('an assets open= scrolls to the tile', async ({ page }) => {
  await opensOnItsTile(page, 'assets')
})
