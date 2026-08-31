import { expect, test } from '@playwright/test'

test('the status strip carries worker state and docs, off the nav row', async ({
  page,
}) => {
  await page.goto('/')
  const strip = page.locator('header .statusbar')
  await expect(strip).toBeVisible()
  // worker state renders in the strip (fixture worker is idle at start)
  await expect(strip).toContainText(/idle|GB/)
  // docs links moved down out of the nav row
  await expect(
    strip.getByRole('link', { name: 'documentation on GitHub' }),
  ).toBeVisible()
  await expect(
    page.locator('header .navrow').getByRole('link', { name: 'Workflows' }),
  ).toBeVisible()
})

test('? opens the shortcuts overlay; Escape closes it; typing ? in a field does not', async ({
  page,
}) => {
  await page.goto('/')
  await page.keyboard.press('?')
  const dialog = page.getByRole('dialog', { name: 'keyboard shortcuts' })
  await expect(dialog).toBeVisible()
  await expect(dialog).toContainText('validate & run')
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
  // inside a text field the character types instead
  const filter = page.getByPlaceholder('filter…')
  await filter.click()
  await filter.press('?')
  await expect(dialog).toHaveCount(0)
  await expect(filter).toHaveValue('?')
})

test('tab order walks the nav row in reading order', async ({ page }) => {
  await page.goto('/')
  // From the top of the document, Tab lands on the nav links in their
  // visual order - the baseline "rational tab order" check on the chrome
  await page.keyboard.press('Tab')
  await expect(
    page.getByRole('link', { name: 'Workflows' }).first(),
  ).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(
    page.getByRole('link', { name: 'Prompts', exact: true }),
  ).toBeFocused()
})

test('saving surfaces a toast, not a pinned banner', async ({ page }) => {
  test.setTimeout(60_000)
  await page.goto('/#/edit/ZImage')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  // success text arrives as a toast...
  await expect(page.getByText(/Saved to .*ZImage\.json/)).toBeVisible({
    timeout: 30_000,
  })
  // ...and auto-dismisses instead of pinning the page down
  await expect(page.getByText(/Saved to .*ZImage\.json/)).toHaveCount(0, {
    timeout: 10_000,
  })
})

test('the status area opens a detail popover', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'server & worker status' }).click()
  const pop = page.getByRole('dialog', { name: 'server status' })
  await expect(pop).toBeVisible()
  // server health and worker lifecycle explained, not just blank space
  await expect(pop).toContainText('Server')
  await expect(pop).toContainText(/worker|spawns with the first job/i)
  await expect(pop).toContainText('queued')
  // Escape closes the popover like every other layer
  await page.keyboard.press('Escape')
  await expect(pop).toHaveCount(0)
  // click-outside closes it too
  await page.getByRole('button', { name: 'server & worker status' }).click()
  await expect(pop).toBeVisible()
  await page.locator('main').click()
  await expect(pop).toHaveCount(0)
})
