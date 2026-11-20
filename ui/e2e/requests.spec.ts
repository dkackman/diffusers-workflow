import { expect, test } from '@playwright/test'

// The fixture seeds one request in the default workspace's ledger
// (e2e/serve_fixture.py): `hero` is done - its succeeded job left a run with
// an image - `alt` is planned, and `cut` was dropped from the plan after a
// job that failed was linked to it. The ledger and the jobs are the
// server's own, so what these read and write is the real contract.
const ID = 'e2e-launch'
const REQUEST = `/#/ws/default/requests/${ID}`

test('the requests list shows the request and what its items came to', async ({
  page,
}) => {
  await page.goto('/#/ws/default/overview')
  const nav = page.getByRole('complementary', { name: 'navigation' })
  await nav.getByRole('link', { name: 'Requests' }).click()
  await expect(page).toHaveURL(/#\/ws\/default\/requests$/)
  await expect(page.getByRole('heading', { name: 'Requests' })).toBeVisible()

  const row = page.locator('.rows li', { hasText: ID })
  await expect(row).toHaveCount(1)
  await expect(row.locator('.state')).toHaveText('open')
  await expect(row).toContainText('3 items')
  await expect(row).toContainText('1 done · 1 planned · 1 dropped')

  await row.getByRole('link', { name: 'E2E launch posters' }).click()
  await expect(page).toHaveURL(new RegExp(`#/ws/default/requests/${ID}$`))
  await expect(
    page.getByRole('heading', { name: 'E2E launch posters' }),
  ).toBeVisible()
})

test('a request shows each item with its status, attempts and result', async ({
  page,
}) => {
  await page.goto(REQUEST)
  await expect(
    page.getByText('make me two posters for the launch'),
  ).toBeVisible()
  const items = page.locator('ol.items > li')
  await expect(items).toHaveCount(3)

  // done: the linked job is its one attempt, and its image is the proof
  const hero = items.filter({ hasText: 'Hero poster' })
  await expect(hero.locator('.status')).toHaveText('done')
  await expect(hero.getByRole('link', { name: 'e2ereqdone01' })).toBeVisible()
  const proof = hero.getByRole('img', { name: 'result of Hero poster' })
  await expect(proof).toBeVisible()
  await expect
    .poll(() => proof.evaluate((img: HTMLImageElement) => img.naturalWidth))
    .toBeGreaterThan(0)

  // planned: nothing linked, nothing to show
  const alt = items.filter({ hasText: 'Alternate poster' })
  await expect(alt.locator('.status')).toHaveText('planned')
  await expect(alt.getByRole('img')).toHaveCount(0)
  await expect(alt.getByRole('button', { name: 'Add comment' })).toBeVisible()

  // dropped: kept with its failed attempt, and no comment is offered
  const cut = items.filter({ hasText: 'Cut scene' })
  await expect(cut.locator('.status')).toHaveText('dropped')
  await expect(cut.getByRole('link', { name: 'e2ereqcut001' })).toBeVisible()
  await expect(cut.getByRole('button', { name: /comment/i })).toHaveCount(0)
  await expect(cut.getByRole('textbox')).toHaveCount(0)
})

test('a comment saved on a live item is in the ledger after a reload', async ({
  page,
}) => {
  const text = `warmer light, please (${Date.now()})`
  await page.goto(REQUEST)
  const alt = page.locator('ol.items > li', { hasText: 'Alternate poster' })
  const before = await page.getByText(/^revision \d+$/).textContent()
  await alt.getByRole('button', { name: /^(Add|Edit) comment$/ }).click()
  await alt.getByRole('textbox', { name: 'Comment on alt' }).fill(text)
  const put = page.waitForResponse(
    (r) =>
      r.request().method() === 'PUT' &&
      r.url().endsWith(`/api/workspaces/default/requests/${ID}`),
  )
  await alt.getByRole('button', { name: 'Save' }).click()
  expect((await put).status()).toBe(200)
  await expect(alt.getByRole('textbox')).toHaveCount(0)
  await expect(alt.getByText(text)).toBeVisible()
  // the write was a revision of the plan, by the user
  await expect(page.getByText(/^revision \d+$/)).not.toHaveText(before ?? '')
  await expect(
    page.locator('.history li', { hasText: 'comment on alt' }).first(),
  ).toContainText('user')

  await page.reload()
  await expect(
    page
      .locator('ol.items > li', { hasText: 'Alternate poster' })
      .getByText(text),
  ).toBeVisible()
  // the other items came back as they were: the dropped one still dropped
  await expect(
    page.locator('ol.items > li', { hasText: 'Cut scene' }).locator('.status'),
  ).toHaveText('dropped')
})

test("a request's jobs link narrows the Jobs page to it, with a chip", async ({
  page,
}) => {
  await page.goto(REQUEST)
  await page.getByRole('link', { name: 'its jobs' }).click()
  await expect(page).toHaveURL(new RegExp(`#/ws/default/jobs\\?req=${ID}$`))
  const chip = page.getByTestId('req-filter')
  await expect(chip).toContainText(ID)
  // only the two linked jobs: the restart tombstones are another matter
  await expect(
    page.locator('.wschip', { hasText: `${ID} › hero` }),
  ).toHaveCount(1)
  await expect(page.locator('.wschip', { hasText: `${ID} › cut` })).toHaveCount(
    1,
  )
  await expect(page.locator('.chip.succeeded')).toHaveCount(1)
  await expect(page.locator('.chip.failed')).toHaveCount(1)
  await expect(page.locator('.chip.cancelled')).toHaveCount(0)

  // the chip leads back to the request, and clearing it widens the list
  await expect(chip.getByRole('link')).toHaveAttribute(
    'href',
    `#/ws/default/requests/${ID}`,
  )
  await chip.getByRole('button').click()
  await expect(chip).toHaveCount(0)
  await expect(page.locator('.chip.cancelled')).toHaveCount(2)
})
