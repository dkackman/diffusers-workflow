import { expect, test } from '@playwright/test'

test('a legacy hash lands under the last-used workspace', async ({ page }) => {
  await page.goto('/#/gallery')
  await expect(page).toHaveURL(/#\/ws\/default\/gallery$/)
  await expect(page.getByRole('heading', { name: 'Gallery' })).toBeVisible()
})

test('the root lands on the overview and the sidebar marks it', async ({
  page,
}) => {
  await page.goto('/')
  await expect(page).toHaveURL(/#\/ws\/default\/overview$/)
  const nav = page.getByRole('complementary', { name: 'navigation' })
  await expect(nav.getByRole('link', { name: 'Overview' })).toHaveAttribute(
    'aria-current',
    'page',
  )
})

test('create a workspace, work in it, delete it', async ({ page }) => {
  await page.goto('/')
  const nav = page.getByRole('complementary', { name: 'navigation' })
  await nav.getByRole('button', { name: 'new workspace' }).click()
  await nav.getByPlaceholder('name').fill('e2e-ws')
  await nav.getByPlaceholder('name').press('Enter')
  await expect(page).toHaveURL(/#\/ws\/e2e-ws\/overview$/)
  // its gallery is scoped: the request carries the workspace
  const galleryRequest = page.waitForRequest(
    (r) =>
      r.url().includes('/api/gallery') && r.url().includes('workspace=e2e-ws'),
  )
  // the previous workspace's sections slide shut; click once they are gone
  await expect(nav.getByRole('link', { name: 'Gallery' })).toHaveCount(1)
  await nav.getByRole('link', { name: 'Gallery' }).click()
  await galleryRequest
  // back to default through the sidebar, and the scope follows
  const defaultRequest = page.waitForRequest(
    (r) => r.url().includes('/api/gallery') && !r.url().includes('workspace='),
  )
  await nav.getByRole('link', { name: /^default/ }).click()
  await expect(nav.getByRole('link', { name: 'Gallery' })).toHaveCount(1)
  await nav.getByRole('link', { name: 'Gallery' }).click()
  await defaultRequest
  // delete from its overview
  await page.goto('/#/ws/e2e-ws/overview')
  await page.getByRole('button', { name: 'Delete workspace' }).click()
  const confirm = page.getByRole('alertdialog')
  await expect(confirm).toContainText('Delete workspace "e2e-ws"?')
  await expect(confirm).toContainText('The prompt library and the shared')
  await confirm.getByRole('button', { name: 'Delete workspace' }).click()
  await expect(page).toHaveURL(/#\/ws\/default\/overview$/)
  await expect(nav.getByText('e2e-ws')).toHaveCount(0)
})

test('switching workspace from the gallery lands on the other gallery', async ({
  page,
}) => {
  const name = `e2e-switch-${Date.now().toString(36)}`
  const nav = page.getByRole('complementary', { name: 'navigation' })
  await page.goto('/')
  try {
    await nav.getByRole('button', { name: 'new workspace' }).click()
    await nav.getByPlaceholder('name').fill(name)
    await nav.getByRole('button', { name: 'Create' }).click()
    await expect(page).toHaveURL(new RegExp(`#/ws/${name}/overview$`))
    await page.goto('/#/ws/default/gallery')
    await expect(page.getByRole('heading', { name: 'Gallery' })).toBeVisible()
    // the other workspace's gallery is asked for, scoped to it
    const scoped = page.waitForRequest(
      (r) =>
        r.url().includes('/api/gallery') &&
        r.url().includes(`workspace=${name}`),
    )
    await nav.getByRole('link', { name: new RegExp(`^${name}`) }).click()
    await expect(page).toHaveURL(new RegExp(`#/ws/${name}/gallery$`))
    await scoped
    await expect(page.getByRole('heading', { name: 'Gallery' })).toBeVisible()
  } finally {
    await page.request.delete(
      `/api/workspaces/${encodeURIComponent(name)}?acknowledged=true`,
    )
  }
})

test('an unknown workspace falls back to default and says so', async ({
  page,
}) => {
  await page.goto('/#/ws/nope/gallery')
  await expect(page).toHaveURL(/#\/ws\/default\/overview$/)
  await expect(page.getByText(/No workspace named nope/)).toBeVisible()
})

test('shared prompts and server models are reachable from the sidebar', async ({
  page,
}) => {
  await page.goto('/')
  const nav = page.getByRole('complementary', { name: 'navigation' })
  await nav.getByRole('link', { name: 'Prompts' }).click()
  await expect(page).toHaveURL(/#\/shared\/prompts$/)
  await nav.getByRole('link', { name: 'Models' }).click()
  await expect(page).toHaveURL(/#\/server\/models$/)
})

// #1694: a row's other text once left its name no width at all
test('a sidebar row shows its whole name beside its size', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 })
  const name = `gilded-${Date.now().toString(36).slice(-5)}`
  const created = await page.request.post('/api/workspaces', {
    data: { name },
  })
  expect(created.ok()).toBe(true)
  try {
    await page.goto('/')
    const nav = page.getByRole('complementary', { name: 'navigation' })
    const row = nav.locator(`a[title="${name}"]`)
    await expect(row.locator('.size')).toBeVisible()
    const label = row.locator('.name')
    await expect(label).toHaveText(name)
    // neither squeezed away nor elided
    expect(
      await label.evaluate(
        (el) => el.clientWidth > 0 && el.scrollWidth <= el.clientWidth,
      ),
    ).toBe(true)
    // and nothing in the row runs past the sidebar's edge
    const edge = (await nav.boundingBox())!
    const size = (await row.locator('.size').boundingBox())!
    expect(size.x + size.width).toBeLessThanOrEqual(edge.x + edge.width)
  } finally {
    await page.request.delete(
      `/api/workspaces/${encodeURIComponent(name)}?acknowledged=true`,
    )
  }
})
