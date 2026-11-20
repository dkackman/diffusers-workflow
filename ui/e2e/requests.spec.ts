import {
  expect,
  test,
  type APIRequestContext,
  type Locator,
  type Page,
} from '@playwright/test'

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
  // each good take can be made the result; none is yet
  await expect(hero.getByRole('button', { name: /^Accept take/ })).toHaveCount(
    2,
  )
  await expect(
    hero.getByRole('button', { name: 'Accept take 1 for Hero poster' }),
  ).toHaveAttribute('aria-pressed', 'false')
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

// The specs below edit a plan, so they have a request of their own
// (e2e/serve_fixture.py): `board` is done with two takes and neither
// accepted, `shot` failed once, `cut` waits on `shot`, `title` has never
// run and is ready, `score` failed once and nothing depends on it. They run
// in order and each leaves the plan as the next expects it.
const BOARD_ID = 'e2e-board'
const BOARD = `/#/ws/default/requests/${BOARD_ID}`
const BOARD_API = `/api/workspaces/default/requests/${BOARD_ID}`
const boardRow = (page: Page, key: string) =>
  page.locator(`li.item[data-key="${key}"]`)
const written = (page: Page) =>
  page.waitForResponse(
    (r) => r.request().method() === 'PUT' && r.url().endsWith(BOARD_API),
  )
const order = (page: Page) =>
  page
    .locator('ol.items > li')
    .evaluateAll((rows) => rows.map((row) => (row as HTMLElement).dataset.key))
const ledger = async (request: APIRequestContext) =>
  (await request.get(BOARD_API)).json()
type Stored = { key: string; title: string; dropped: boolean }
/** Write the plan as the agent would, between the page's read and its
 * next write: the real route, `by: "agent"`. Answers the new revision. */
async function asAgent(
  request: APIRequestContext,
  why: string,
  change: (items: Stored[]) => Stored[],
): Promise<number> {
  const read = await ledger(request)
  const response = await request.put(BOARD_API, {
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
  return (await response.json()).revision
}
async function choose(page: Page, row: Locator, entry: string) {
  await row.getByRole('button', { name: /^More actions for/ }).click()
  await page.getByRole('dialog').getByRole('button', { name: entry }).click()
}

test('accepting a take is shown at once, is in the ledger, and the toast takes it back', async ({
  page,
  request,
}) => {
  await page.goto(BOARD)
  const board = boardRow(page, 'board')
  const first = board.getByRole('button', {
    name: 'Accept take 1 for Board sheet',
  })
  await expect(first).toHaveAttribute('aria-pressed', 'false')
  let put = written(page)
  await first.click()
  await expect(first).toHaveAttribute('aria-pressed', 'true')
  await expect(first).toHaveText('Accepted')
  expect((await put).status()).toBe(200)
  await expect(board.locator('.row')).toContainText('accepted')
  await expect(
    board.locator('.strip .tile').nth(0).locator('.mark'),
  ).toHaveText('accepted')
  // the row stays open under the button, and focus stays on it
  await expect(first).toBeFocused()
  let stored = await ledger(request)
  expect(stored.items[0].accepted.job).toBe('e2ebrdtake01')
  expect(stored.history.at(-1)).toMatchObject({
    by: 'user',
    why: 'accept e2ebrdtake01 for board',
  })

  put = written(page)
  await page
    .getByRole('listitem')
    .filter({ hasText: 'Accepted take 1 for Board sheet' })
    .getByRole('button', { name: 'Undo' })
    .click()
  expect((await put).status()).toBe(200)
  await expect(first).toHaveAttribute('aria-pressed', 'false')
  stored = await ledger(request)
  expect(stored.items[0].accepted).toBeNull()
  await page.reload()
  await expect(boardRow(page, 'board').locator('.row')).toContainText(
    'latest, none accepted',
  )
})

test('dropping an item others wait on says what it does to them, and Restore puts it all back', async ({
  page,
  request,
}) => {
  await page.goto(BOARD)
  const shot = boardRow(page, 'shot')
  const cut = boardRow(page, 'cut')
  await expect(cut.locator('.row .chip')).toHaveText('waiting')
  await expect(cut.locator('.row')).toContainText(
    'waits on Opening shot (failed)',
  )
  await choose(page, shot, 'Drop from plan')
  const confirm = page.getByRole('alertdialog')
  await expect(confirm).toContainText(
    'Final cut waits on this item. Dropping it removes that dependency, so it no longer waits for it.',
  )
  let put = written(page)
  await confirm.getByRole('button', { name: 'Drop' }).click()
  expect((await put).status()).toBe(200)
  expect(await order(page)).toEqual(['board', 'cut', 'title', 'score'])
  // the dependent no longer waits: with nothing else to wait on, it is ready
  await expect(cut.locator('.row .chip')).toHaveText('ready')
  await expect(cut.locator('.row')).not.toContainText('waits on')
  let stored = await ledger(request)
  expect(stored.items.find((i: Stored) => i.key === 'cut').depends_on).toEqual(
    [],
  )
  expect(stored.items.at(-1)).toMatchObject({ key: 'shot', dropped: true })

  // out of the plan, in the dropped group, with the way back
  const group = page.locator('details.droppedgroup')
  await expect(group.locator('summary')).toHaveText('Dropped (1)')
  await group.locator('summary').click()
  put = written(page)
  await shot.getByRole('button', { name: 'Restore' }).click()
  expect((await put).status()).toBe(200)
  await expect(group).toHaveCount(0)
  // its place and what waited on it, as they were
  expect(await order(page)).toEqual(['board', 'shot', 'cut', 'title', 'score'])
  await expect(cut.locator('.row')).toContainText(
    'waits on Opening shot (failed)',
  )
  stored = await ledger(request)
  expect(stored.items.find((i: Stored) => i.key === 'cut').depends_on).toEqual([
    'shot',
  ])
  await page.reload()
  expect(await order(page)).toEqual(['board', 'shot', 'cut', 'title', 'score'])
})

test('the handle reorders with the keys, in one write that a reload still shows', async ({
  page,
}) => {
  await page.goto(BOARD)
  const handle = page.locator('#req-item-title-handle')
  await expect(handle).toHaveAccessibleName(
    'Reorder Title card, position 4 of 5',
  )
  let puts = 0
  page.on('request', (r) => {
    if (r.method() === 'PUT' && r.url().endsWith(BOARD_API)) puts++
  })
  const put = written(page)
  await handle.focus()
  await page.keyboard.press('ArrowUp')
  await page.keyboard.press('ArrowUp')
  expect(await order(page)).toEqual(['board', 'title', 'shot', 'cut', 'score'])
  // focus stays on the handle, which now says where the item is
  await expect(handle).toBeFocused()
  await expect(handle).toHaveAccessibleName(
    'Reorder Title card, position 2 of 5',
  )
  await expect(page.getByTestId('request-live')).toHaveText(
    'Title card moved to position 2 of 5.',
  )
  expect((await put).status()).toBe(200)
  expect(puts).toBe(1)
  await page.reload()
  expect(await order(page)).toEqual(['board', 'title', 'shot', 'cut', 'score'])
})

test("an edit made over the agent's write goes on top of it, and the page says so", async ({
  page,
  request,
}) => {
  await page.goto(BOARD)
  await expect(boardRow(page, 'cut').locator('.name')).toHaveText('Final cut')
  // the agent rewrites the plan under the open page
  const theirs = await asAgent(request, 'tightened the cut', (items) =>
    items.map((item) =>
      item.key === 'cut' ? { ...item, title: 'Final cut, tighter' } : item,
    ),
  )
  const answers: number[] = []
  page.on('response', (r) => {
    if (r.request().method() === 'PUT' && r.url().endsWith(BOARD_API))
      answers.push(r.status())
  })
  await choose(page, boardRow(page, 'score'), 'Move to top')
  const notice = page.getByRole('status').filter({ hasText: 'applied on top' })
  await expect(notice).toContainText(
    `The agent changed the plan while you were editing (r${theirs}: 'tightened the cut'). Your change was applied on top.`,
  )
  // refused once as stale, then made over the plan as the agent left it
  expect(answers).toEqual([409, 200])
  expect(await order(page)).toEqual(['score', 'board', 'title', 'shot', 'cut'])
  await expect(boardRow(page, 'cut').locator('.name')).toHaveText(
    'Final cut, tighter',
  )
  const stored = await ledger(request)
  expect(stored.items.map((i: Stored) => i.key).slice(0, 2)).toEqual([
    'score',
    'board',
  ])
  expect(stored.items.find((i: Stored) => i.key === 'cut').title).toBe(
    'Final cut, tighter',
  )
  await notice.getByRole('button', { name: 'See changes' }).click()
  await expect(page.locator(`#plan-change-r${theirs}`)).toBeFocused()
  await notice.getByRole('button', { name: 'Dismiss' }).click()
  await expect(notice).toHaveCount(0)
})

test('a drop of an item the agent changed is held back until the user answers', async ({
  page,
  request,
}) => {
  await page.goto(BOARD)
  const score = boardRow(page, 'score')
  await expect(score.locator('.name')).toHaveText('Score')
  const theirs = await asAgent(request, 'strings only', (items) =>
    items.map((item) =>
      item.key === 'score' ? { ...item, title: 'Score, strings only' } : item,
    ),
  )
  // a job is linked to it, nothing waits on it, nothing is running: no
  // confirm - the drop goes straight out, and is refused as stale
  await choose(page, score, 'Drop from plan')
  const card = score.getByRole('alert')
  await expect(card).toContainText(
    `While you were dropping this, the agent changed it (r${theirs}: 'strings only'): it is now called Score, strings only.`,
  )
  // not dropped, and nothing written
  await expect(score.locator('.name')).toHaveText('Score, strings only')
  expect(await order(page)).toContain('score')
  expect((await ledger(request)).revision).toBe(theirs)

  const put = written(page)
  await card.getByRole('button', { name: 'Drop anyway' }).click()
  expect((await put).status()).toBe(200)
  await expect(card).toHaveCount(0)
  expect(await order(page)).toEqual(['board', 'title', 'shot', 'cut'])
  const stored = await ledger(request)
  expect(stored.items.at(-1)).toMatchObject({
    key: 'score',
    title: 'Score, strings only',
    dropped: true,
  })
})

test('the run dialog prices what is ready, says what will not run, and spends nothing until asked', async ({
  page,
}) => {
  let submitted = 0
  page.on('request', (r) => {
    if (r.method() === 'POST' && new URL(r.url()).pathname === '/api/jobs')
      submitted++
  })
  await page.goto(BOARD)
  const run = page.getByRole('button', { name: 'Run 1 ready…' })
  const priced = page.waitForResponse(
    (r) => new URL(r.url()).pathname === '/api/validate',
  )
  await run.click()
  const dialog = page.getByRole('dialog', { name: 'Run what is ready' })
  // a dialog that spends starts on Cancel
  await expect(dialog.getByRole('button', { name: 'Cancel' })).toBeFocused()
  expect((await priced).status()).toBe(200)
  const row = dialog.locator('.runrow[data-key="title"]')
  await expect(row).toContainText('Title card')
  await expect(row.locator('.flow')).toHaveText('e2e-card')
  // the server's own quote: a task-only workflow has no figure, and says so
  await expect(row.locator('.quote')).toContainText('no figure')
  await expect(dialog.locator('.total')).toHaveText('total: no figure')
  await expect(dialog.getByRole('button', { name: 'Run 1' })).toBeEnabled()
  await expect(dialog.locator('.wont li')).toHaveText([
    'Opening shot: failed: run it again from its row',
    'Final cut, tighter: waits on Opening shot (failed)',
  ])
  // Cancel: nothing queued, and focus is back on what opened it
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  await expect(dialog).toHaveCount(0)
  await expect(run).toBeFocused()
  expect(submitted).toBe(0)

  // one item, whose stored workflow the server cannot reach: it would be
  // refused, so there is nothing to run
  await boardRow(page, 'shot')
    .getByRole('button', { name: 'Run again…' })
    .click()
  const single = page.getByRole('dialog', { name: 'Run Opening shot' })
  await expect(single.locator('.wont li')).toContainText(
    'Opening shot: would be refused: workflow_path must name a workflow the server can reach',
  )
  await expect(single.getByRole('checkbox')).toHaveCount(0)
  await expect(single.getByRole('button', { name: 'Run 0' })).toBeDisabled()
  await single.getByRole('button', { name: 'Cancel' }).click()
  expect(submitted).toBe(0)
})

test('running a ready item queues a job bound to its quote, and the job appears on the item', async ({
  page,
  request,
}) => {
  await page.goto(BOARD)
  const title = boardRow(page, 'title')
  await expect(title.locator('.row .chip')).toHaveText('ready')
  await page.getByRole('button', { name: 'Run 1 ready…' }).click()
  const dialog = page.getByRole('dialog', { name: 'Run what is ready' })
  const submit = page.waitForResponse(
    (r) =>
      r.request().method() === 'POST' &&
      new URL(r.url()).pathname === '/api/jobs',
  )
  await dialog.getByRole('button', { name: 'Run 1' }).click()
  const answer = await submit
  expect(answer.status()).toBe(201)
  const sent = answer.request().postDataJSON()
  expect(sent).toMatchObject({
    workflow_path: 'e2e-card',
    arguments: { line: 'The opening' },
    request: BOARD_ID,
    item: 'title',
  })
  // the validate plan itself is the acknowledgement
  expect(sent.acknowledged_cost.fingerprint).toMatch(/^sha256:/)
  const job = await answer.json()
  expect(job).toMatchObject({
    acknowledged: 'bound',
    request: BOARD_ID,
    item: 'title',
  })
  await expect(page.getByText('Queued 1 job')).toBeVisible()
  await expect(dialog).toHaveCount(0)
  // read again at once: the item carries the job, in whatever state it
  // has reached, and is no longer something to run
  await expect(title.locator('.row .chip')).toHaveText(/queued|running|done/)
  await expect(page.getByRole('button', { name: /ready…$/ })).toHaveCount(0)
  const stored = await ledger(request)
  const item = stored.items.find((i: Stored) => i.key === 'title')
  expect(item.attempts.map((a: { job: string }) => a.job)).toEqual([job.id])
  // a text task with no model: it finishes, and the page follows it there
  await expect(title.locator('.row .chip')).toHaveText('done', {
    timeout: 25_000,
  })
  await expect(title.locator('.proofcell a')).toHaveAttribute(
    'href',
    `#/ws/default/jobs/${job.id}`,
  )
  // a done item can take another run, from its menu
  await title.getByRole('button', { name: /^More actions for/ }).click()
  await expect(
    page.getByRole('dialog').getByRole('button', { name: 'Run another take…' }),
  ).toBeVisible()
})

test('dragging the handle moves an item, in one write', async ({ page }) => {
  await page.goto(BOARD)
  expect(await order(page)).toEqual(['board', 'title', 'shot', 'cut'])
  const handle = page.locator('#req-item-board-handle')
  const target = boardRow(page, 'title')
  const from = await handle.boundingBox()
  const to = await target.boundingBox()
  if (!from || !to) throw new Error('no boxes')
  const put = written(page)
  await page.mouse.move(from.x + from.width / 2, from.y + from.height / 2)
  await page.mouse.down()
  // past the middle of the row below
  await page.mouse.move(from.x + from.width / 2, to.y + to.height * 0.8, {
    steps: 8,
  })
  await expect(target).toHaveClass(/insert-after/)
  await page.mouse.up()
  expect(await order(page)).toEqual(['title', 'board', 'shot', 'cut'])
  const body = (await put).request().postDataJSON()
  expect(body.why).toBe('reorder board')
  await page.reload()
  expect(await order(page)).toEqual(['title', 'board', 'shot', 'cut'])
})

test('the request can be marked done and reopened from its menu, and deleted once closed', async ({
  page,
  request,
}) => {
  await page.goto(BOARD)
  const head = page.locator('.reqhead')
  const menu = head.getByRole('button', {
    name: 'More actions for this request',
  })
  await expect(head.locator('.tag')).toHaveText('open')
  let put = written(page)
  await menu.click()
  // an open request has no Delete
  await expect(
    page.getByRole('dialog').getByRole('button', { name: 'Delete…' }),
  ).toHaveCount(0)
  await page
    .getByRole('dialog')
    .getByRole('button', { name: 'Mark done' })
    .click()
  expect((await put).status()).toBe(200)
  await expect(head.locator('.tag')).toHaveText('done')
  let stored = await ledger(request)
  expect(stored.state).toBe('done')
  expect(stored.history.at(-1)).toMatchObject({
    by: 'user',
    why: 'marked done',
    state: 'done',
  })
  // closed: nothing here runs
  await expect(page.getByRole('button', { name: /^Run/ })).toHaveCount(0)

  put = written(page)
  await menu.click()
  await page.getByRole('dialog').getByRole('button', { name: 'Reopen' }).click()
  expect((await put).status()).toBe(200)
  await expect(head.locator('.tag')).toHaveText('open')
  stored = await ledger(request)
  expect(stored.history.at(-1)).toMatchObject({
    why: 'reopened',
    state: 'open',
  })
  await expect(
    boardRow(page, 'shot').getByRole('button', { name: 'Run again…' }),
  ).toBeVisible()

  // abandoned, then deleted: its jobs stay, the ledger's record goes
  put = written(page)
  await menu.click()
  await page
    .getByRole('dialog')
    .getByRole('button', { name: 'Abandon' })
    .click()
  expect((await put).status()).toBe(200)
  await expect(head.locator('.tag')).toHaveText('abandoned')
  await menu.click()
  await page
    .getByRole('dialog')
    .getByRole('button', { name: 'Delete…' })
    .click()
  const confirm = page.getByRole('alertdialog')
  await expect(confirm).toContainText('Delete "E2E storyboard"?')
  await expect(confirm).toContainText('Its jobs and their outputs are kept.')
  const removed = page.waitForResponse(
    (r) => r.request().method() === 'DELETE' && r.url().includes(BOARD_API),
  )
  await confirm.getByRole('button', { name: 'Delete' }).click()
  expect((await removed).status()).toBe(200)
  await expect(page).toHaveURL(/#\/ws\/default\/requests$/)
  await expect(page.getByText('Deleted E2E storyboard')).toBeVisible()
  await page.getByRole('combobox', { name: 'show' }).selectOption('all')
  await expect(page.locator('a.row', { hasText: ID })).toBeVisible()
  await expect(page.locator('a.row', { hasText: BOARD_ID })).toHaveCount(0)
  expect((await request.get(BOARD_API)).status()).toBe(404)
})
