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
  const toggle = row.getByTestId('item-title')
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

  const row = page.getByTestId('request-row').filter({ hasText: ID })
  await expect(row).toHaveCount(1)
  await expect(row.getByTestId('state-tag')).toHaveText('open')
  // nothing of it is running, and one of its two planned items can be run
  await expect(row.locator('.attention')).toHaveText(
    'Needs you: 1 ready to run, 1 take to choose.',
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
  await expect(
    page.getByTestId('request-row').filter({ hasText: ID }),
  ).toBeVisible()
  await page.getByRole('combobox', { name: 'show' }).selectOption('all')
  await expect(page).toHaveURL(/#\/ws\/default\/requests\?show=all$/)
  await expect(
    page.getByTestId('request-row').filter({ hasText: ID }),
  ).toBeVisible()
})

test('a request leads with where it stands and what needs you', async ({
  page,
}) => {
  await page.goto(REQUEST)
  const head = page.getByTestId('req-head')
  await expect(head.getByTestId('state-tag')).toHaveText('open')
  await expect(head.locator('.standing')).toHaveText(
    'Nothing is running. 1 of 3 done.',
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
  await expect(alt.getByTestId('item-chip')).toHaveText('ready')
  await expect(alt.getByRole('img')).toHaveCount(0)

  // planned too, but on an item that is not done: it names it
  const sheet = item(page, 'Contact sheet')
  await expect(sheet.getByTestId('item-chip')).toHaveText('waiting')
  await expect(sheet.getByTestId('item-row')).toContainText(
    'waits on Alternate poster (ready)',
  )
  await sheet.getByRole('button', { name: 'Alternate poster' }).click()
  await expect(alt.getByTestId('item-title')).toBeFocused()

  // dropped: out of the plan, in its own closed group, with its failed take
  const group = page.getByTestId('dropped-group')
  await expect(group.locator(':scope > summary')).toHaveText('Dropped (1)')
  const cut = item(page, 'Cut scene')
  await expect(cut).toBeHidden()
  await group.locator(':scope > summary').click()
  await expect(cut.getByTestId('item-chip')).toHaveText('dropped')
  await expand(cut, 'Cut scene')
  await expect(
    cut.getByRole('list', { name: /^Takes of/ }).locator('.tile'),
  ).toHaveCount(1)
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
  await expect(hero.getByTestId('item-chip')).toHaveText('done')
  // two good takes and none accepted: it says so, and opens to show them;
  // open, the row's own picture steps aside for the takes
  await expect(hero.getByTestId('item-row')).toContainText('2 takes')
  await expect(hero.getByTestId('item-row')).toContainText(
    'latest, none accepted',
  )
  await expect(hero.getByTestId('item-title')).toHaveAttribute(
    'aria-expanded',
    'true',
  )
  await expect(hero.getByTestId('item-proof')).toHaveClass(/under/)
  const takes = hero.getByRole('list', { name: /^Takes of/ }).locator('.tile')
  await expect(takes).toHaveCount(2)
  await expect(takes.nth(0)).toContainText('take 1')
  await expect(takes.nth(0).locator('.mark')).toHaveCount(0)
  await expect(takes.nth(1).locator('.mark')).toHaveText('latest take')
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
  // closed, the row's proof is the result, and leads to the job that made it
  await hero.getByTestId('item-title').click()
  const proof = hero.getByTestId('item-proof').getByRole('img', {
    name: 'result of Hero poster',
  })
  await loaded(proof)
  await expect(hero.getByTestId('item-proof').locator('a')).toHaveAttribute(
    'href',
    '#/ws/default/jobs/e2ereqdone02',
  )
})

test('a note for the agent says whether the agent is connected, and is in the ledger after a reload', async ({
  page,
}) => {
  const text = `warmer light, please (${Date.now()})`
  await page.goto(REQUEST)
  const alt = item(page, 'Alternate poster')
  await expand(alt, 'Alternate poster')
  const revision = page.locator('.rail .now')
  const before = await revision.textContent()
  await alt.getByRole('button', { name: 'Add a note for the agent' }).click()
  const box = alt.getByRole('textbox', {
    name: 'Note for the agent on Alternate poster',
  })
  // said before a word is written: no agent session polls this server
  await expect(
    alt.getByText(/The agent isn't connected\. It reads/),
  ).toBeVisible()
  await box.fill(text)
  const post = page.waitForResponse(
    (r) =>
      r.request().method() === 'POST' &&
      r.url().endsWith('/api/workspaces/default/request-notes'),
  )
  await alt.getByRole('button', { name: 'Save note' }).click()
  expect((await post).status()).toBe(200)
  await expect(alt.getByRole('textbox')).toHaveCount(0)
  await expect(alt.getByText(text)).toBeVisible()
  // and said again under the saved note: when the agent will see it
  await expect(
    alt.getByText(
      /^written .+\. The agent isn't connected\. It reads this the next time it updates the plan\.$/,
    ),
  ).toBeVisible()
  await expect(page.getByTestId('req-head').locator('.figures')).toContainText(
    /\d+ notes? waiting/,
  )
  // the write was a revision of the request, by the user
  await expect(revision).not.toHaveText(before ?? '')
  const rail = page.getByRole('complementary', { name: /The ask/ })
  await expect(rail.locator('.changes li').first()).toContainText('you')

  await page.reload()
  const again = item(page, 'Alternate poster')
  // a note the agent has not had the chance to read opens its row
  await expect(again.getByTestId('item-title')).toHaveAttribute(
    'aria-expanded',
    'true',
  )
  await expect(again.getByText(text)).toBeVisible()
  // the other items came back as they were: the dropped one still dropped
  await expect(
    page.locator('[data-testid="dropped-group"] > summary'),
  ).toHaveText('Dropped (1)')
  await expect(item(page, 'Hero poster').getByTestId('item-chip')).toHaveText(
    'done',
  )
})

test("a note on the whole request is saved by an item note's rules", async ({
  page,
}) => {
  const text = `make it all warmer (${Date.now()})`
  await page.goto(REQUEST)
  const head = page.getByTestId('req-head')
  await head.getByRole('button', { name: 'Add a note for the agent' }).click()
  await head
    .getByRole('textbox', { name: 'Note for the agent on this request' })
    .fill(text)
  const post = page.waitForResponse(
    (r) =>
      r.request().method() === 'POST' &&
      r.url().endsWith('/api/workspaces/default/request-notes'),
  )
  await head.getByRole('button', { name: 'Save note' }).click()
  expect((await post).status()).toBe(200)
  await expect(head.getByText(text)).toBeVisible()
  await page.reload()
  await expect(page.getByTestId('req-head').getByText(text)).toBeVisible()
  await expect(
    page
      .getByTestId('req-head')
      .getByText(/The agent isn't connected\. It reads/)
      .first(),
  ).toBeVisible()
})

test("a request's jobs link narrows the Jobs page to it, with a chip", async ({
  page,
}) => {
  await page.goto(REQUEST)
  await page
    .getByTestId('req-head')
    .getByRole('link', { name: 'its jobs' })
    .click()
  await expect(page).toHaveURL(new RegExp(`#/ws/default/jobs\\?req=${ID}$`))
  const chip = page.getByTestId('req-filter')
  // the link's title, not its text: the request's own title replaces the id
  // there once the chip's listing lands
  await expect(chip.getByRole('link')).toHaveAttribute(
    'title',
    `${ID} - open this request`,
  )
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
  // the item's menu holds its jobs and outputs, not the take strip
  await hero.getByRole('button', { name: /^More actions for/ }).click()
  await page.getByRole('dialog').getByRole('link', { name: 'Its jobs' }).click()
  const chip = page.getByTestId('req-filter')
  await expect(chip.getByRole('link')).toHaveAttribute(
    'title',
    `${ID} - open this request`,
  )
  await expect(chip).toContainText('› hero')
  await expect(page.locator('.chip.succeeded')).toHaveCount(2)
  await expect(page.locator('.chip.failed')).toHaveCount(0)
})

test("the item menu's widest entry fits inside it", async ({ page }) => {
  await page.goto(REQUEST)
  const hero = item(page, 'Hero poster')
  await hero.getByRole('button', { name: /^More actions for/ }).click()
  const menu = page.getByRole('dialog')
  const box = await menu.boundingBox()
  const entries = menu.locator('.entries > li > :is(a, button)')
  expect(await entries.count()).toBeGreaterThan(1)
  for (const entry of await entries.all()) {
    const own = await entry.boundingBox()
    // inside the panel's padding, and its words inside its own box
    expect(own!.x + own!.width).toBeLessThan(box!.x + box!.width - 1)
    expect(await entry.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(
      true,
    )
  }
})

test('a row at 1300 px keeps its title on one line', async ({ page }) => {
  await page.setViewportSize({ width: 1300, height: 900 })
  // the storyboard has open rows with takes and a held one
  await page.goto('/#/ws/default/requests/e2e-board')
  const lines = page.locator('li.item .titleline')
  await expect(lines.first()).toBeVisible()
  for (const line of await lines.all()) {
    const title = (await line.getByTestId('item-title').boundingBox())!
    const key = (await line.locator('.key').boundingBox())!
    // the key beside the title, not under it, and the title one line
    expect(key.y).toBeLessThan(title.y + title.height)
    expect(
      await line
        .getByTestId('item-name')
        .evaluate((el) => el.getClientRects().length),
    ).toBe(1)
  }
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
// The page asks with `?reasons=true`: match the path, not the whole URL
const isBoard = (url: string) => new URL(url).pathname === BOARD_API
const written = (page: Page) =>
  page.waitForResponse(
    (r) => r.request().method() === 'PUT' && isBoard(r.url()),
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
  await expect(board.getByTestId('item-row')).toContainText('accepted')
  await expect(
    board
      .getByRole('list', { name: /^Takes of/ })
      .locator('.tile')
      .nth(0)
      .locator('.mark'),
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
  await expect(boardRow(page, 'board').getByTestId('item-row')).toContainText(
    'latest, none accepted',
  )
})

test('Back closes the take viewer', async ({ page }) => {
  await page.goto(BOARD)
  const board = boardRow(page, 'board')
  await board
    .getByRole('button', { name: 'take 1 of Board sheet', exact: true })
    .click()
  const viewer = page.getByRole('dialog')
  await expect(viewer).toBeVisible()
  await expect(page).toHaveURL(/open=board/)
  // Back closes the viewer and stays on the request
  await page.goBack()
  await expect(viewer).toHaveCount(0)
  await expect(page).not.toHaveURL(/open=/)
  await expect(page).toHaveURL(new RegExp(BOARD_ID))
  await expect(
    board.getByRole('list', { name: /^Takes of/ }).locator('.tile'),
  ).toHaveCount(2)
})

test('dropping an item others wait on says what it does to them, and Restore puts it all back', async ({
  page,
  request,
}) => {
  await page.goto(BOARD)
  const shot = boardRow(page, 'shot')
  const cut = boardRow(page, 'cut')
  await expect(cut.getByTestId('item-chip')).toHaveText('waiting')
  await expect(cut.getByTestId('item-row')).toContainText(
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
  await expect(cut.getByTestId('item-chip')).toHaveText('ready')
  await expect(cut.getByTestId('item-row')).not.toContainText('waits on')
  let stored = await ledger(request)
  expect(stored.items.find((i: Stored) => i.key === 'cut').depends_on).toEqual(
    [],
  )
  expect(stored.items.at(-1)).toMatchObject({ key: 'shot', dropped: true })

  // out of the plan, in the dropped group, with the way back
  const group = page.getByTestId('dropped-group')
  await expect(group.locator(':scope > summary')).toHaveText('Dropped (1)')
  await group.locator(':scope > summary').click()
  put = written(page)
  await shot.getByRole('button', { name: 'Restore' }).click()
  expect((await put).status()).toBe(200)
  await expect(group).toHaveCount(0)
  // its place and what waited on it, as they were
  expect(await order(page)).toEqual(['board', 'shot', 'cut', 'title', 'score'])
  await expect(cut.getByTestId('item-row')).toContainText(
    'waits on Opening shot (failed)',
  )
  stored = await ledger(request)
  expect(stored.items.find((i: Stored) => i.key === 'cut').depends_on).toEqual([
    'shot',
  ])
  await page.reload()
  await expect
    .poll(() => order(page))
    .toEqual(['board', 'shot', 'cut', 'title', 'score'])
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
    if (r.method() === 'PUT' && isBoard(r.url())) puts++
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
  await expect
    .poll(() => order(page))
    .toEqual(['board', 'title', 'shot', 'cut', 'score'])
})

test("an edit made over the agent's write goes on top of it, and the page says so", async ({
  page,
  request,
}) => {
  await page.goto(BOARD)
  await expect(boardRow(page, 'cut').getByTestId('item-name')).toHaveText(
    'Final cut',
  )
  // the agent rewrites the plan under the open page
  const theirs = await asAgent(request, 'tightened the cut', (items) =>
    items.map((item) =>
      item.key === 'cut' ? { ...item, title: 'Final cut, tighter' } : item,
    ),
  )
  const answers: number[] = []
  page.on('response', (r) => {
    if (r.request().method() === 'PUT' && isBoard(r.url()))
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
  await expect(boardRow(page, 'cut').getByTestId('item-name')).toHaveText(
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
  await expect(score.getByTestId('item-name')).toHaveText('Score')
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
  await expect(score.getByTestId('item-name')).toHaveText('Score, strings only')
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
  const run = page.getByRole('button', { name: 'Run 2…' })
  const priced = page.waitForResponse(
    (r) =>
      new URL(r.url()).pathname === '/api/validate' &&
      r.request().postDataJSON()?.workflow_path === 'e2e-title',
  )
  await run.click()
  const dialog = page.getByRole('dialog', { name: 'Run what can run now' })
  // a dialog that spends starts on Cancel
  await expect(dialog.getByRole('button', { name: 'Cancel' })).toBeFocused()
  expect((await priced).status()).toBe(200)
  const row = dialog.locator('[data-testid="run-row"][data-key="title"]')
  await expect(row).toContainText('Title card')
  await expect(row.locator('.flow')).toHaveText('e2e-title')
  // the server's own quote: a task-only workflow that has never run (the
  // fixture keeps `e2e-title` for this) has no figure, and says so, the engine's reason one click away
  await expect(row.locator('.quote')).toContainText(
    'No time estimate for this workflow.',
  )
  await expect(row.locator('.quote details summary')).toHaveText(
    "Why there's no estimate",
  )
  await expect(dialog.locator('.total')).toHaveText('total: no figure')
  await expect(
    dialog.getByRole('button', { name: 'Run 1 job', exact: true }),
  ).toBeEnabled()
  // the failed shot is offered again, but its workflow cannot be reached
  await expect(dialog.locator('.wont li')).toContainText([
    'Opening shot: would be refused: workflow_path must name a workflow the server can reach',
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
  await expect(
    single.getByRole('button', { name: 'Run 0 jobs', exact: true }),
  ).toBeDisabled()
  await single.getByRole('button', { name: 'Cancel' }).click()
  expect(submitted).toBe(0)
})

test('a held item says so, counts down, runs only when asked, and a note turns it back to the agent', async ({
  page,
}) => {
  let submitted = 0
  page.on('request', (r) => {
    if (r.method() === 'POST' && new URL(r.url()).pathname === '/api/jobs')
      submitted++
  })
  await page.goto(BOARD)
  // the fixture holds `title` for an hour, and no session listens here
  const banner = page.getByTestId('req-head').locator('.hold p')
  await expect(banner).toHaveText(
    /^1 item waiting for your OK\. The agent isn't connected, so nothing runs at .+ unless it reconnects\. Run it here, or leave a note\.$/,
  )
  const title = boardRow(page, 'title')
  const chip = title.locator('.chip.held')
  await expect(chip).toHaveText(/^waiting for your OK · \d\d:\d\d$/)
  const first = await chip.textContent()
  await expect(chip).not.toHaveText(first ?? '', { timeout: 5000 })

  await page.getByRole('button', { name: 'Run now…' }).click()
  const dialog = page.getByRole('dialog', { name: 'Run what can run now' })
  await expect(
    dialog.locator('[data-testid="run-row"][data-key="title"] .held span'),
  ).toHaveText(
    /^Waiting for your OK until .+\. Running it now is your go-ahead\.$/,
  )
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  await expect(dialog).toHaveCount(0)
  expect(submitted).toBe(0)

  await expand(title, 'Title card')
  await title.getByRole('button', { name: 'Add a note for the agent' }).click()
  await expect(
    title.getByText(
      'Saving this note stops 1 item waiting for your OK from starting on their own.',
    ),
  ).toBeVisible()
  await title
    .getByRole('textbox', { name: 'Note for the agent on Title card' })
    .fill('make the title bolder')
  await title.getByRole('button', { name: 'Save note' }).click()
  await expect(banner).toHaveText(
    "You left a note after these were held, so they won't start on their own, and the agent isn't connected. Run them here, or wait for the agent.",
  )
  await expect(chip).toHaveText('on hold: note left')
  await expect(page.getByRole('button', { name: 'Run now…' })).toBeVisible()
  expect(submitted).toBe(0)
})

test('running a ready item queues a job bound to its quote, and the job appears on the item', async ({
  page,
  request,
}) => {
  await page.goto(BOARD)
  const title = boardRow(page, 'title')
  await expect(title.getByTestId('item-chip')).toHaveText('ready')
  // the header counts what the dialog offers: the ready title and every
  // failed item that can run again, however many earlier tests left
  const header = page.getByRole('button', { name: /^Run \d+…$/ })
  const offered = Number(/\d+/.exec((await header.textContent()) ?? '')?.[0])
  await header.click()
  const dialog = page.getByRole('dialog', { name: 'Run what can run now' })
  await expect(dialog.getByTestId('run-row')).toHaveCount(offered)
  await expect(
    dialog.locator('[data-testid="run-row"][data-key="title"]'),
  ).toHaveCount(1)
  const submit = page.waitForResponse(
    (r) =>
      r.request().method() === 'POST' &&
      new URL(r.url()).pathname === '/api/jobs',
  )
  await dialog.getByRole('button', { name: 'Run 1 job', exact: true }).click()
  const answer = await submit
  expect(answer.status()).toBe(201)
  const sent = answer.request().postDataJSON()
  expect(sent).toMatchObject({
    workflow_path: 'e2e-title',
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
  await expect(title.getByTestId('item-chip')).toHaveText(/queued|running|done/)
  // the title is no longer offered: whatever else was, still is
  if (offered > 1)
    await expect(
      page.getByRole('button', { name: `Run ${offered - 1}…` }),
    ).toBeVisible()
  else await expect(header).toHaveCount(0)
  const stored = await ledger(request)
  const item = stored.items.find((i: Stored) => i.key === 'title')
  expect(item.attempts.map((a: { job: string }) => a.job)).toEqual([job.id])
  // a text task with no model: it finishes, and the page follows it there
  await expect(title.getByTestId('item-chip')).toHaveText('done', {
    timeout: 25_000,
  })
  await expect(title.getByTestId('item-proof').locator('a')).toHaveAttribute(
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
  await expect
    .poll(() => order(page))
    .toEqual(['title', 'board', 'shot', 'cut'])
})

test('the request can be marked done and reopened from its menu, and deleted once closed', async ({
  page,
  request,
}) => {
  await page.goto(BOARD)
  const head = page.getByTestId('req-head')
  const menu = head.getByRole('button', {
    name: 'More actions for this request',
  })
  await expect(head.getByTestId('state-tag')).toHaveText('open')
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
  await expect(head.getByTestId('state-tag')).toHaveText('done')
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
  await expect(head.getByTestId('state-tag')).toHaveText('open')
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
  await expect(head.getByTestId('state-tag')).toHaveText('abandoned')
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
  await expect(
    page.getByTestId('request-row').filter({ hasText: ID }),
  ).toBeVisible()
  await expect(page.locator('a.row', { hasText: BOARD_ID })).toHaveCount(0)
  expect((await request.get(BOARD_API)).status()).toBe(404)
})
