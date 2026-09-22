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
  await page
    .getByRole('alertdialog')
    .getByRole('button', { name: 'Delete' })
    .click()
  await expect(page).toHaveURL(/#\/ws\/default\/overview$/)
  await expect(nav.getByText('e2e-ws')).toHaveCount(0)
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
