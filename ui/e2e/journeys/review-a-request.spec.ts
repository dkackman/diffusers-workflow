import { expect, test, type Locator, type Page } from '../test'

// Reviewing a request an agent planned, end to end against the real server:
// the list and the header say what it wants of the person, each part goes to
// its item, a failed take says why, a note stops a hold, the run dialog
// prices before it spends, Accept stars the take, what the agent changes is
// said, a row above its dependency says it still waits, and a closed request
// is read-only until reopened. The fixture's `e2e-review`
// (e2e/serve_fixture.py): `hero` done with two takes and neither accepted,
// `plate` failed once on a CUDA error, `comp` failed and waits on `plate`,
// `title` held for the next hour. One test: each step leaves the request as
// the next expects it.
const ID = 'e2e-review'
const REQUEST = `/#/ws/default/requests/${ID}`
const API = `/api/workspaces/default/requests/${ID}`
// The feed polls every 5 s (CHANGES_POLL_MS): two polls, and some slack
const FEED = { timeout: 15_000 }
const TAKE_2 = 'review-takes/20260909-093000-5eed5eed/hero-take.png'

const row = (page: Page, key: string) =>
  page.locator(`li.item[data-key="${key}"]`)
const written = (page: Page) =>
  page.waitForResponse(
    (r) => r.request().method() === 'PUT' && new URL(r.url()).pathname === API,
  )

/** Open a row that is not open: a row opens itself for some states */
async function expand(item: Locator) {
  const toggle = item.locator('.title')
  if ((await toggle.getAttribute('aria-expanded')) === 'false')
    await toggle.click()
  await expect(toggle).toHaveAttribute('aria-expanded', 'true')
}

type Stored = {
  key: string
  title: string
  dropped: boolean
  arguments?: Record<string, unknown>
}

/** Write the plan as the agent would: the real route, `by: "agent"` */
async function asAgent(
  request: Page['request'],
  why: string,
  change: (items: Stored[]) => Stored[],
) {
  const read = await (await request.get(API)).json()
  const response = await request.put(API, {
    data: {
      title: read.title,
      goal: read.goal,
      revision: read.revision,
      why,
      by: 'agent',
      items: change(read.items.filter((item: Stored) => !item.dropped)),
    },
  })
  expect(response.status()).toBe(200)
}

/** What a closed request takes away, and an open one has */
async function expectEditable(page: Page, editable: boolean) {
  const count = (locator: Locator) =>
    editable
      ? expect(locator.first()).toBeVisible()
      : expect(locator).toHaveCount(0)
  await count(page.locator('li.item .grip'))
  await count(page.getByRole('button', { name: /^Accept take/ }))
  await count(page.locator('li.item .act'))
  await count(page.getByRole('button', { name: 'Add a note for the agent' }))
  const closed = page.locator('.panel.closed')
  if (editable) await expect(closed).toHaveCount(0)
  else
    await expect(closed).toContainText(
      'This request is done. Reopen it to change the plan or leave a note.',
    )
}

async function markDone(page: Page) {
  const put = written(page)
  await page
    .locator('.reqhead')
    .getByRole('button', { name: 'More actions for this request' })
    .click()
  await page
    .getByRole('dialog')
    .getByRole('button', { name: 'Mark done' })
    .click()
  expect((await put).status()).toBe(200)
  await expect(page.locator('.reqhead .tag')).toHaveText('done')
}

test('review a request: what it needs, why it failed, a note, a price, a choice, the agent’s changes, and a reopen', async ({
  page,
  request,
}) => {
  test.setTimeout(120_000)
  const NEEDS = 'Needs you: 1 waiting for your OK, 1 failed, 1 take to choose.'

  // 1. The list says what it wants, and puts it first: it holds an item
  await page.goto('/#/ws/default/requests')
  // (held requests lead, in write order: other specs' held requests may
  // precede it, nothing that does not wait on the person may)
  const listed = page.locator('a.row', { hasText: ID })
  await expect(listed).toHaveCount(1)
  const rows = await page.locator('a.row').allInnerTexts()
  const at = rows.findIndex((text) => text.includes(ID))
  expect(at).toBeGreaterThanOrEqual(0)
  for (const text of rows.slice(0, at))
    expect(text).toContain('waiting for your OK')
  await expect(listed.locator('.attention')).toHaveText(NEEDS)
  await page.setViewportSize({ width: 390, height: 844 })
  await expect(listed.locator('.title')).toBeVisible()
  await expect(listed.locator('.title')).toHaveText('E2E review')
  await page.setViewportSize({ width: 1280, height: 900 })

  // 2. The header says the same, and each part goes to its item
  await page.goto(REQUEST)
  const head = page.locator('.reqhead')
  await expect(head.locator('.needs')).toHaveText(NEEDS)
  await head.locator('.needs').getByRole('button', { name: '1 failed' }).click()
  const plate = row(page, 'plate')
  await expect(plate.locator('.title')).toBeFocused()
  await expect(plate.locator('.title')).toHaveAttribute('aria-expanded', 'true')
  await expect(plate.locator('.row .sub.bad')).toHaveText(
    'Take 1 failed: CUDA out of memory. Tried to allocate 2.00 GiB',
  )
  await expect(row(page, 'comp').locator('.row')).toContainText(
    'Failed. It can run again once Background plate is done.',
  )

  // 3. A note on the request warns that it stops the hold, and does
  const box = head.locator('.notebox')
  await box.getByRole('button', { name: 'Add a note for the agent' }).click()
  await expect(
    box.getByText(
      'Saving this note stops 1 item waiting for your OK from starting on their own.',
    ),
  ).toBeVisible()
  await box
    .getByRole('textbox', { name: 'Note for the agent on this request' })
    .fill('warmer, please')
  await box.getByRole('button', { name: 'Save note' }).click()
  await expect(head.locator('.hold p')).toHaveText(
    "You left a note after these were held, so they won't start on their own, and the agent isn't connected. Run them here, or wait for the agent.",
  )
  const runNow = head.getByRole('button', { name: 'Run now…' })
  await expect(runNow).toBeVisible()
  await expect(row(page, 'title').locator('.chip.held')).toHaveText(
    'on hold: note left',
  )

  // 4. Run now prices the held item and spends nothing until asked. The
  // header's Run counts what its dialog offers: the held title and the
  // failed plate, which starts unticked
  await expect(head.getByRole('button', { name: 'Run 2…' })).toBeVisible()
  await runNow.click()
  const dialog = page.getByRole('dialog', { name: 'Run what can run now' })
  await expect(
    dialog.locator('.runrow[data-key="title"] .quote'),
  ).toContainText('No time estimate for this workflow.')
  await expect(
    dialog.getByRole('button', { name: 'Run 1 job', exact: true }),
  ).toBeEnabled()
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  await expect(dialog).toHaveCount(0)

  // 5. Accept a take: it is the result, and its file is starred
  const hero = row(page, 'hero')
  await expand(hero)
  const put = written(page)
  await hero
    .getByRole('button', { name: 'Accept take 2 for Hero still' })
    .click()
  expect((await put).status()).toBe(200)
  await expect(
    page.getByText('Accepted take 2 for Hero still and starred it'),
  ).toBeVisible()
  await expect
    .poll(async () => {
      const files: { name: string; starred?: boolean }[] = (
        await (await request.get('/api/gallery')).json()
      ).files
      return files.find((file) => file.name === TAKE_2)?.starred ?? false
    })
    .toBe(true)

  // 6. What the agent changes is said, and listed under its write
  await asAgent(request, 'wider hero, new plate line', (items) =>
    items.map((item) =>
      item.key === 'hero'
        ? { ...item, title: 'Hero still (wide)' }
        : item.key === 'plate'
          ? { ...item, arguments: { line: 'A warmer plate' } }
          : item,
    ),
  )
  const live = page.getByTestId('request-live')
  await expect(live).toContainText(
    'The agent renamed Hero still to Hero still (wide).',
    FEED,
  )
  await expect(live).toContainText(
    'The agent changed the arguments of Background plate.',
  )
  const newest = page.locator('.changes ol > li').first()
  await expect(newest).toContainText('wider hero, new plate line')
  await expect(newest.locator('ul.diff li')).toHaveText([
    'Renamed Hero still to Hero still (wide)',
    'Changed Background plate: its arguments (line)',
  ])

  // 7. A row moved above what it waits on says it still waits
  const moved = written(page)
  await page.locator('#req-item-comp-handle').focus()
  await page.keyboard.press('Home')
  expect((await moved).status()).toBe(200)
  await expect(row(page, 'comp').locator('.hint')).toHaveText(
    'Runs after Background plate whatever its place.',
  )

  // 8. The item menu's widest entry fits inside it
  await row(page, 'plate')
    .getByRole('button', { name: /^More actions for/ })
    .click()
  const menu = page.getByRole('dialog')
  const panel = (await menu.boundingBox())!
  const entries = menu.locator('.entries > li > :is(a, button)')
  expect(await entries.count()).toBeGreaterThan(1)
  for (const entry of await entries.all()) {
    const own = (await entry.boundingBox())!
    expect(own.x + own.width).toBeLessThan(panel.x + panel.width - 1)
    expect(await entry.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(
      true,
    )
  }
  await page.keyboard.press('Escape')
  await expect(menu).toHaveCount(0)

  // 9. Done is read-only and undoes; the closed line's Reopen opens it again
  await expectEditable(page, true)
  await markDone(page)
  await expectEditable(page, false)
  const undone = written(page)
  await page
    .getByRole('listitem')
    .filter({ hasText: 'Marked "E2E review" done' })
    .getByRole('button', { name: 'Undo' })
    .click()
  expect((await undone).status()).toBe(200)
  await expect(head.locator('.tag')).toHaveText('open')
  await expectEditable(page, true)

  await markDone(page)
  await expectEditable(page, false)
  const reopened = written(page)
  await page
    .locator('.panel.closed')
    .getByRole('button', { name: 'Reopen' })
    .click()
  expect((await reopened).status()).toBe(200)
  await expect(page.getByText('Reopened "E2E review"')).toBeVisible()
  await expectEditable(page, true)
  // a reload reads it open from the ledger; the accepted row starts closed
  await page.reload()
  await expect(page.locator('.reqhead .tag')).toHaveText('open')
  await expand(row(page, 'hero'))
  await expectEditable(page, true)

  // 10. Everything changed was seen here: a reload has nothing new to say
  await page.reload()
  await expect(page.locator('.reqhead .tag')).toHaveText('open')
  await expect(page.locator('.changes')).toBeVisible()
  await expect(page.getByText(/Since you last looked/)).toHaveCount(0)
})
