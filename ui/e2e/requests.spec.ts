import { expect, test, type Locator, type Page } from '@playwright/test'

// The fixture seeds one request in the default workspace's ledger
// (e2e/serve_fixture.py): `hero` is done, with two succeeded jobs that each
// left a run with an image and neither accepted; `alt` is planned on `hero`
// alone, so a job could be queued for it; `sheet` is planned on `alt`, so it
// waits; and `cut` was dropped from the plan after a job that failed was
// linked to it. The ledger and the jobs are the server's own, so what these
// read and write is the real contract.
const ID = 'e2e-launch'
const REQUEST = `/#/ws/default/requests/${ID}`

// A row by its item's key: a title also turns up in the rows that name it
// as a dependency
const KEYS: Record<string, string> = {
  'Hero poster': 'hero',
  'Alternate poster': 'alt',
  'Contact sheet': 'sheet',
  'Cut scene': 'cut',
}
const item = (page: Page, title: string) =>
  page.locator(`li.item[data-key="${KEYS[title]}"]`)
const loaded = (img: Locator) =>
  expect
    .poll(() => img.evaluate((el: HTMLImageElement) => el.naturalWidth))
    .toBeGreaterThan(0)
/** Open a row that is not open: a row opens itself for some states */
async function expand(row: Locator, title: string) {
  const toggle = row.locator('.title')
  await expect(toggle).toHaveText(title)
  if ((await toggle.getAttribute('aria-expanded')) === 'false')
    await toggle.click()
  await expect(toggle).toHaveAttribute('aria-expanded', 'true')
}

test('the requests list says what the request wants and shows its latest results', async ({
  page,
}) => {
  await page.goto('/#/ws/default/overview')
  const nav = page.getByRole('complementary', { name: 'navigation' })
  await nav.getByRole('link', { name: 'Requests' }).click()
  await expect(page).toHaveURL(/#\/ws\/default\/requests$/)
  await expect(page.getByRole('heading', { name: 'Requests' })).toBeVisible()

  const row = page.locator('a.row', { hasText: ID })
  await expect(row).toHaveCount(1)
  await expect(row.locator('.tag')).toHaveText('open')
  // nothing of it is running, and one of its two planned items can be run
  await expect(row.locator('.attention')).toHaveText(
    'idle: 2 planned (1 ready), nothing running',
  )
  // one item of the three in the plan is done: the dropped one is not counted
  await expect(row.locator('.progress')).toContainText('1/3')
  await expect(
    row.getByRole('img', { name: '1 done, 2 not started' }),
  ).toBeVisible()
  // the proof: the result's own picture, not a count of them
  const cover = row.locator('.cover img')
  await expect(cover).toHaveCount(1)
  await loaded(cover)

  // the whole row is the link
  await row.click()
  await expect(page).toHaveURL(new RegExp(`#/ws/default/requests/${ID}$`))
  await expect(
    page.getByRole('heading', { name: 'E2E launch posters' }),
  ).toBeVisible()
})

test('the list hides closed requests until asked, and keeps the choice in the hash', async ({
  page,
}) => {
  await page.goto('/#/ws/default/requests')
  await expect(page.locator('a.row', { hasText: ID })).toBeVisible()
  await page.getByRole('combobox', { name: 'show' }).selectOption('all')
  await expect(page).toHaveURL(/#\/ws\/default\/requests\?show=all$/)
  await expect(page.locator('a.row', { hasText: ID })).toBeVisible()
})

test('a request leads with where it stands and what needs you', async ({
  page,
}) => {
  await page.goto(REQUEST)
  const head = page.locator('.reqhead')
  await expect(head.locator('.tag')).toHaveText('open')
  await expect(head.locator('.standing')).toHaveText(
    'Nothing is running. 1 ready to run. 1 of 3 done.',
  )
  await expect(head.locator('.needs')).toHaveText(
    'Needs you: 1 ready to run, 1 take to choose.',
  )
  // a segment of the bar is an item by name, and goes to its row
  await head.getByRole('button', { name: 'Contact sheet: waiting' }).click()
  await expect(
    item(page, 'Contact sheet').getByRole('button', {
      name: 'Contact sheet',
      exact: true,
    }),
  ).toBeFocused()
  // the ask, verbatim, and the plan's changes are beside the plan
  const rail = page.getByRole('complementary', { name: /The ask/ })
  await expect(
    rail.getByText('make me two posters for the launch'),
  ).toBeVisible()
  await expect(rail.getByText('dropped the cut scene')).toBeVisible()
})

test('each item says its state in words: ready apart from waiting, with what it waits on', async ({
  page,
}) => {
  await page.goto(REQUEST)
  await expect(page.locator('ol.items > li')).toHaveCount(3)

  // a job could be queued for it now
  const alt = item(page, 'Alternate poster')
  await expect(alt.locator('.row .chip')).toHaveText('ready')
  await expect(alt.getByRole('img')).toHaveCount(0)

  // planned too, but on an item that is not done: it names it
  const sheet = item(page, 'Contact sheet')
  await expect(sheet.locator('.row .chip')).toHaveText('waiting')
  await expect(sheet.locator('.row')).toContainText(
    'waits on Alternate poster (ready)',
  )
  await sheet.getByRole('button', { name: 'Alternate poster' }).click()
  await expect(alt.locator('.title')).toBeFocused()

  // dropped: out of the plan, in its own closed group, with its failed take
  const group = page.locator('details.droppedgroup')
  await expect(group.locator('summary')).toHaveText('Dropped (1)')
  const cut = item(page, 'Cut scene')
  await expect(cut).toBeHidden()
  await group.locator('summary').click()
  await expect(cut.locator('.row .chip')).toHaveText('dropped')
  await expand(cut, 'Cut scene')
  await expect(cut.locator('.strip .tile')).toHaveCount(1)
  await expect(cut.locator('.strip .chip')).toHaveText('failed')
  // the ledger keeps a dropped item as stored: no note is offered on one
  await expect(cut.getByRole('button', { name: /note/i })).toHaveCount(0)
  await expect(cut.getByRole('textbox')).toHaveCount(0)
})

test('a done item shows its takes as pictures, the latest marked as the result', async ({
  page,
}) => {
  await page.goto(REQUEST)
  const hero = item(page, 'Hero poster')
  await expect(hero.locator('.row .chip')).toHaveText('done')
  // the row's proof is the result, and leads to the job that made it
  const proof = hero.locator('.proofcell').getByRole('img', {
    name: 'result of Hero poster',
  })
  await loaded(proof)
  await expect(hero.locator('.proofcell a')).toHaveAttribute(
    'href',
    '#/ws/default/jobs/e2ereqdone02',
  )
  // two good takes and none accepted: it says so, and opens to show them
  await expect(hero.locator('.row')).toContainText('2 takes')
  await expect(hero.locator('.row')).toContainText('latest, none accepted')
  await expect(hero.locator('.title')).toHaveAttribute('aria-expanded', 'true')
  const takes = hero.locator('.strip .tile')
  await expect(takes).toHaveCount(2)
  await expect(takes.nth(0)).toContainText('take 1')
  await expect(takes.nth(0).locator('.mark')).toHaveCount(0)
  await expect(takes.nth(1).locator('.mark')).toHaveText(
    'shown as the result (latest)',
  )
  await loaded(takes.nth(0).getByRole('img'))
  await expect(takes.nth(0).getByRole('link')).toHaveAttribute(
    'href',
    '#/ws/default/jobs/e2ereqdone01',
  )
  // nothing here accepts a take yet
  await expect(hero.getByRole('button', { name: /accept/i })).toHaveCount(0)
})

test('a note for the agent says nothing is sent, and is in the ledger after a reload', async ({
  page,
}) => {
  const text = `warmer light, please (${Date.now()})`
  await page.goto(REQUEST)
  const alt = item(page, 'Alternate poster')
  await expand(alt, 'Alternate poster')
  const revision = page.locator('.rail .now')
  const before = await revision.textContent()
  await alt
    .getByRole('button', { name: /^(Leave a note for the agent|Edit note)$/ })
    .click()
  const box = alt.getByRole('textbox', {
    name: 'Note for the agent on Alternate poster',
  })
  // said before a word is written
  await expect(alt.getByText(/Nothing is sent to the agent/)).toBeVisible()
  await box.fill(text)
  const put = page.waitForResponse(
    (r) =>
      r.request().method() === 'PUT' &&
      r.url().endsWith(`/api/workspaces/default/requests/${ID}`),
  )
  await alt.getByRole('button', { name: 'Save note' }).click()
  expect((await put).status()).toBe(200)
  await expect(alt.getByRole('textbox')).toHaveCount(0)
  await expect(alt.getByText(text)).toBeVisible()
  // and said again under the saved note: when the agent will see it
  await expect(
    alt.getByText(
      /^Saved .+\. Nothing is sent to the agent\. It reads this the next time it updates the plan\.$/,
    ),
  ).toBeVisible()
  await expect(page.locator('.reqhead .figures')).toContainText(
    /\d+ notes? waiting/,
  )
  // the write was a revision of the plan, by the user
  await expect(revision).not.toHaveText(before ?? '')
  const rail = page.getByRole('complementary', { name: /The ask/ })
  await expect(rail.locator('.changes li').first()).toContainText('you')

  await page.reload()
  const again = item(page, 'Alternate poster')
  // a note the agent has not had the chance to read opens its row
  await expect(again.locator('.title')).toHaveAttribute('aria-expanded', 'true')
  await expect(again.getByText(text)).toBeVisible()
  // the other items came back as they were: the dropped one still dropped
  await expect(page.locator('details.droppedgroup summary')).toHaveText(
    'Dropped (1)',
  )
  await expect(item(page, 'Hero poster').locator('.row .chip')).toHaveText(
    'done',
  )
})

test("a note on the whole request is saved by an item note's rules", async ({
  page,
}) => {
  const text = `make it all warmer (${Date.now()})`
  await page.goto(REQUEST)
  const head = page.locator('.reqhead')
  await head
    .getByRole('button', { name: /^(Leave a note for the agent|Edit note)$/ })
    .click()
  await head
    .getByRole('textbox', { name: 'Note for the agent on this request' })
    .fill(text)
  const put = page.waitForResponse(
    (r) =>
      r.request().method() === 'PUT' &&
      r.url().endsWith(`/api/workspaces/default/requests/${ID}`),
  )
  await head.getByRole('button', { name: 'Save note' }).click()
  expect((await put).status()).toBe(200)
  await expect(head.getByText(text)).toBeVisible()
  await page.reload()
  await expect(page.locator('.reqhead').getByText(text)).toBeVisible()
  await expect(
    page.locator('.reqhead').getByText(/Nothing is sent to the agent/),
  ).toBeVisible()
})

test("a request's jobs link narrows the Jobs page to it, with a chip", async ({
  page,
}) => {
  await page.goto(REQUEST)
  await page.locator('.reqhead').getByRole('link', { name: 'its jobs' }).click()
  await expect(page).toHaveURL(new RegExp(`#/ws/default/jobs\\?req=${ID}$`))
  const chip = page.getByTestId('req-filter')
  await expect(chip).toContainText(ID)
  // only the three linked jobs: the restart tombstones are another matter
  await expect(
    page.locator('.wschip', { hasText: `${ID} › hero` }),
  ).toHaveCount(2)
  await expect(page.locator('.wschip', { hasText: `${ID} › cut` })).toHaveCount(
    1,
  )
  await expect(page.locator('.chip.succeeded')).toHaveCount(2)
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

test("an item's jobs link narrows the Jobs page to that item", async ({
  page,
}) => {
  await page.goto(REQUEST)
  const hero = item(page, 'Hero poster')
  await expand(hero, 'Hero poster')
  await hero.getByRole('link', { name: 'its jobs' }).click()
  await expect(page.getByTestId('req-filter')).toContainText(`${ID} › hero`)
  await expect(page.locator('.chip.succeeded')).toHaveCount(2)
  await expect(page.locator('.chip.failed')).toHaveCount(0)
})
