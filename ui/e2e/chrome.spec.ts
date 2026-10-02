import { expect, test } from '@playwright/test'

test('worker state and docs sit on the nav row, in one header row', async ({
  page,
}) => {
  await page.goto('/')
  // One header row: the state the old second row carried now sits at the
  // right of the nav, so no page spends a strip on the word "idle"
  const row = page.locator('header .navrow')
  await expect(row).toHaveCount(1)
  const state = page.locator('header .state')
  await expect(state).toBeVisible()
  // worker state renders there (fixture worker is idle at start)
  await expect(state).toContainText(/idle|GB/)
  await expect(
    state.getByRole('link', { name: 'documentation on GitHub' }),
  ).toBeVisible()
  // the nav row's own link is now the breadcrumb (the top nav moved to the
  // sidebar) - it still names where you are
  await expect(
    row.getByRole('link', { name: 'default', exact: true }),
  ).toBeVisible()
})

test('? opens the shortcuts overlay; Escape closes it; typing ? in a field does not', async ({
  page,
}) => {
  // the root is the overview now, which has no filter field - the workflows
  // list does
  await page.goto('/#/ws/default/workflows')
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

test('the shortcuts overlay traps focus and returns it to the trigger', async ({
  page,
}) => {
  await page.goto('/')
  // give a known element focus before opening, so "focus returns to the
  // trigger" has something concrete to check against
  const trigger = page
    .getByRole('complementary', { name: 'navigation' })
    .getByRole('link', { name: 'Workflows' })
  await trigger.focus()
  await page.keyboard.press('?')
  const dialog = page.getByRole('dialog', { name: 'keyboard shortcuts' })
  await expect(dialog).toBeVisible()
  // focus moved into the dialog on open, not left on whatever had it before
  await expect(dialog.getByRole('button', { name: 'close' })).toBeFocused()
  // Tab cycles within the dialog - it never escapes to the nav behind it
  await page.keyboard.press('Tab')
  await expect(dialog.getByRole('button', { name: 'close' })).toBeFocused()
  await page.keyboard.press('Shift+Tab')
  await expect(dialog.getByRole('button', { name: 'close' })).toBeFocused()
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
  // closing hands focus back to whatever had it before the dialog opened
  await expect(trigger).toBeFocused()
})

test('the status popover takes focus and returns it to its trigger button', async ({
  page,
}) => {
  await page.goto('/')
  const trigger = page.getByRole('button', { name: 'server & worker status' })
  await trigger.click()
  const pop = page.getByRole('dialog', { name: 'server status' })
  await expect(pop).toBeVisible()
  // A non-modal popover: focus moves into it (the idle fixture has nothing
  // focusable inside, so the panel itself takes it), it claims no
  // aria-modal, and the page behind it stays reachable
  await expect(pop).toBeFocused()
  await expect(pop).not.toHaveAttribute('aria-modal', 'true')
  await page.keyboard.press('Escape')
  await expect(pop).toHaveCount(0)
  // closing returns focus to the button that opened it
  await expect(trigger).toBeFocused()
})

test('tab order walks the sidebar nav in reading order', async ({ page }) => {
  await page.goto('/')
  // The top nav moved into the sidebar (an <aside> that sits before the
  // header in the document), so the baseline "rational tab order" check
  // now walks the sidebar rather than the header row. The wordmark is
  // still the first stop: it is a link home.
  await page.keyboard.press('Tab')
  await expect(page.locator('aside .brand')).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(
    page.getByRole('button', { name: 'collapse sidebar' }),
  ).toBeFocused()
  await page.keyboard.press('Tab')
  const nav = page.getByRole('complementary', { name: 'navigation' })
  await expect(nav.getByRole('link', { name: 'Overview' })).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(nav.getByRole('link', { name: 'Workflows' })).toBeFocused()
})

test('saving surfaces a toast, not a pinned banner', async ({ page }) => {
  test.setTimeout(60_000)
  await page.goto('/#/edit/models/z-image')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  // success text arrives as a toast...
  await expect(page.getByText('Saved to models/z-image')).toBeVisible({
    timeout: 30_000,
  })
  // ...and auto-dismisses instead of pinning the page down
  await expect(page.getByText('Saved to models/z-image')).toHaveCount(0, {
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
  // the trigger toggles: a second click closes it rather than reopening it
  const trigger = page.getByRole('button', { name: 'server & worker status' })
  await trigger.click()
  await expect(pop).toBeVisible()
  await trigger.click()
  await expect(pop).toHaveCount(0)
})

test('a class field suggests as you type, and keeps text it does not list', async ({
  page,
}) => {
  await page.goto('/#/edit')
  const field = page.getByLabel('pipeline').first()
  await field.fill('')
  await field.pressSequentially('Pipeline')
  const list = page.getByRole('listbox')
  await expect(list).toBeVisible()
  const first = list.getByRole('option').first()
  const chosen = (await first.textContent())?.trim() ?? ''
  await first.click()
  await expect(field).toHaveValue(chosen)
  await expect(list).toHaveCount(0)

  // a class the server does not list is still a value
  await field.fill('')
  await field.pressSequentially('MyOwnPipelineXyz')
  await expect(page.getByRole('listbox')).toHaveCount(0)
  await field.press('Tab')
  await expect(field).toHaveValue('MyOwnPipelineXyz')
})
