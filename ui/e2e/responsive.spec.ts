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
  '#/ws/default/overview',
  '#/server/status',
  '#/ws/default/requests',
  '#/ws/default/assets',
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

test('the models table shows size at 390px', async ({ page }) => {
  // The listing is this test's own: the fixture's seeded repo is deleted by a
  // journey, so the test answers a one-row cache itself, whatever ran before.
  await page.route('**/api/models', async (route) => {
    if (route.request().method() !== 'GET') return route.fallback()
    await route.fulfill({
      json: {
        cache_dir: '/tmp/e2e-hub',
        disk_free: null,
        disk_total: null,
        size_on_disk: 4096,
        warnings: [],
        repos: [
          {
            repo_id: 'e2e/tiny',
            repo_type: 'model',
            size_on_disk: 4096,
            nb_files: 3,
            last_accessed: null,
            last_job_use: null,
            last_modified: null,
            revisions: [],
          },
        ],
      },
    })
  })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/#/models')
  await page.waitForLoadState('networkidle')
  const size = page.getByRole('columnheader', { name: 'size' })
  await expect(
    page.getByRole('cell', { name: /e2e\/tiny/ }).first(),
  ).toBeVisible()
  await expect(size).toBeInViewport()
  await expect(page.getByRole('columnheader', { name: 'files' })).toBeHidden()
  await expect(
    page.getByRole('columnheader', { name: 'last used' }),
  ).toBeHidden()
})

test("the Overview's long job id is clipped", async ({ page }) => {
  // The listing is this test's own: one finished job with an id no row holds
  const id = 'e2e' + 'x'.repeat(200)
  await page.route('**/api/jobs?**', async (route) => {
    const running = route.request().url().includes('status=running')
    const jobs = running
      ? []
      : [
          {
            id,
            workflow: 'text-to-image',
            workflow_name: 'text-to-image',
            status: 'failed',
            run_id: null,
            created_at: 1,
            started_at: 1,
            finished_at: 2,
            workspace: 'default',
          },
        ]
    await route.fulfill({ json: { jobs, total: jobs.length } })
  })
  await page.setViewportSize({ width: 900, height: 900 })
  await page.goto('/#/ws/default/overview')
  const row = page.getByTitle(`job ${id}`, { exact: true })
  const jid = row.locator('.jid')
  await expect(jid).toBeVisible()
  const rowBox = (await row.boundingBox())!
  const jidBox = (await jid.boundingBox())!
  expect(jidBox.x + jidBox.width).toBeLessThanOrEqual(
    rowBox.x + rowBox.width + 1,
  )
  // there is more id than box, and what does not fit is cut with an ellipsis
  expect(
    await jid.evaluate((el) => el.scrollWidth - el.clientWidth),
  ).toBeGreaterThan(0)
  await expect(jid).toHaveCSS('overflow-x', 'hidden')
  await expect(jid).toHaveCSS('text-overflow', 'ellipsis')
  expect(await overflowOf(page)).toBeLessThanOrEqual(1)
})

test('a row tile is 32 px', async ({ page }) => {
  // The suggestion rows of an image field, over the fixture's own stills
  await page.goto('/#/ws/default/workflows/templates/image-to-image')
  const field = page.locator('input.referencefield').first()
  await field.fill('')
  await field.pressSequentially('output:')
  const tile = page.locator('.ui-suggest [role="option"] .tile-row').first()
  await expect(tile).toBeVisible()
  expect((await tile.boundingBox())!.width).toBeCloseTo(32, 0)
})
