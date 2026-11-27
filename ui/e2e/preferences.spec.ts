import { expect, test } from './test'

test('the old status link lands on Activity and the sidebar names the Server pages', async ({
  page,
}) => {
  await page.goto('/#/server/status')
  await expect(page.getByRole('heading', { name: 'Activity' })).toBeVisible()
  await expect(page).toHaveURL(/#\/server\/activity/)
  const nav = page.getByRole('complementary', { name: 'navigation' })
  for (const name of ['Activity', 'Models', 'Connect', 'System']) {
    await expect(nav.getByRole('link', { name, exact: true })).toBeVisible()
  }
})

test('the top bar schema link opens the schema', async ({ page }) => {
  await page.goto('/#/server/activity')
  await page.getByRole('link', { name: 'workflow schema' }).click()
  await expect(page).toHaveURL(/#\/server\/schema/)
  await expect(page.getByRole('heading', { name: 'Schema' })).toBeVisible()
})

test('Models opens on Preferences and answers what runs here', async ({
  page,
}) => {
  await page.goto('/#/server/models')
  const card = page.getByRole('region', { name: 'preference text_to_image' })
  await expect(card).toBeVisible()
  // each tier answers: ready, or a download before its first job, or a refusal
  await expect(
    card
      .getByText(/ready|before its first job|Nothing fits this machine/)
      .first(),
  ).toBeVisible({ timeout: 20_000 })
})

test('editing a ladder re-checks it live and Discard restores it', async ({
  page,
}) => {
  await page.goto('/#/server/models/preferences')
  const card = page.getByRole('region', { name: 'preference text_to_image' })
  await card.getByRole('button', { name: 'Edit', exact: true }).click()

  // the first tier whose ladder holds two models: one chip can't be moved
  const lists = card.getByRole('list', { name: /in the order tried/ })
  const count = await lists.count()
  let at = -1
  for (let n = 0; n < count; n++) {
    if ((await lists.nth(n).locator('li.target').count()) >= 2) {
      at = n
      break
    }
  }
  expect(at, 'a ladder with two models').toBeGreaterThanOrEqual(0)
  const chain = lists.nth(at)
  const chips = chain.locator('li.target')
  const before = await chips.locator('code').allTextContents()

  await chips.first().focus()
  await page.keyboard.press('Alt+ArrowRight')
  await expect(chips.locator('code').first()).not.toHaveText(before[0]!)
  await expect(card.getByText(/Unsaved changes/)).toBeVisible()
  await expect(
    card.getByText(/Jobs already queued keep the model/),
  ).toBeVisible()

  // the dry run answers within 2 s of the edit: no chip is left unjudged
  await expect(
    chain.locator('li.target.good, li.target.warn, li.target.bad').first(),
  ).toBeVisible({ timeout: 2_000 })

  await card.getByRole('button', { name: 'Discard' }).click()
  await expect(
    card.getByRole('button', { name: 'Edit', exact: true }),
  ).toBeVisible()
  await card.getByRole('button', { name: 'Edit', exact: true }).click()
  await expect(
    card
      .getByRole('list', { name: /in the order tried/ })
      .nth(at)
      .locator('li.target code'),
  ).toHaveText(before)
})
