import { expect, test } from '@playwright/test'

/** Nothing should scroll the document sideways: wide content gets its own
 * scroller (the models table) or wraps. Regression guard for the header,
 * which is sticky and so slides off screen once the document overflows. */

const ROUTES = [
  '#/workflows',
  '#/workflows/models/z-image',
  '#/prompts',
  '#/prompt-edit/scenic_landscape',
  '#/jobs',
  '#/edit',
  '#/gallery',
  '#/models',
  '#/schema',
]

const WIDTHS = [1280, 900, 768, 640, 480, 375]

const overflowOf = (page: import('@playwright/test').Page) =>
  page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)

for (const width of WIDTHS) {
  for (const route of ROUTES) {
    test(`no horizontal overflow at ${width}px on ${route}`, async ({
      page,
    }) => {
      await page.setViewportSize({ width, height: 900 })
      await page.goto('/' + route)
      await page.waitForLoadState('networkidle')
      expect(await overflowOf(page)).toBeLessThanOrEqual(1)
    })
  }

  // The step/component/arguments grids only render once a step is expanded,
  // and they are the most deeply nested layout in the app
  test(`no horizontal overflow at ${width}px in the step editor`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 900 })
    await page.goto('/#/edit/models/flux-dev')
    await page.waitForLoadState('networkidle')
    // loaded steps default to the compact digest - switch to full to render
    // the deepest layout (the component/arguments grid)
    await page
      .locator('.step')
      .first()
      .getByRole('button', { name: 'full' })
      .click()
    await expect(page.locator('.step .grid').first()).toBeVisible()
    expect(await overflowOf(page)).toBeLessThanOrEqual(1)
  })
}

test('below the breakpoint the sidebar becomes a drawer opened from the header', async ({
  page,
}) => {
  // The top nav used to wrap onto a second header row at narrow widths;
  // it now lives in the sidebar, which is off-screen and inert below
  // 900px until the header's menu button opens it as a drawer.
  await page.setViewportSize({ width: 700, height: 900 })
  await page.goto('/')
  const nav = page.getByRole('complementary', { name: 'navigation' })
  const menu = page.getByRole('button', { name: 'open navigation' })
  await expect(menu).toBeVisible()
  await expect(nav).not.toBeInViewport()
  await menu.click()
  await expect(nav).toBeInViewport()
  await expect(nav.getByRole('link', { name: 'Overview' })).toBeVisible()
  // any navigation closes the drawer again
  await nav.getByRole('link', { name: 'Workflows' }).click()
  await expect(nav).not.toBeInViewport()
})
