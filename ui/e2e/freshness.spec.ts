import { expect, test } from '@playwright/test'

// A page of server data reloads from the change feed (src/lib/changes.svelte.ts),
// never from a reload: what an agent writes through the API while the page is
// open shows up on it. The real server and the real route, so the counter the
// write moves is the server's own.
const ID = 'e2e-fresh'
const API = `/api/workspaces/default/requests/${ID}`

test('a request written through the API from outside appears on the open Requests page without a reload', async ({
  page,
  request,
}) => {
  await page.goto('/#/ws/default/requests')
  await expect(page.getByRole('heading', { name: 'Requests' })).toBeVisible()
  // the seeded request is listed: the page has made its first read
  await expect(page.locator('a.row', { hasText: 'e2e-launch' })).toHaveCount(1)
  await expect(page.locator('a.row', { hasText: ID })).toHaveCount(0)

  let navigated = 0
  page.on('framenavigated', () => navigated++)
  const created = await request.put(API, {
    data: {
      title: 'Made by an agent',
      goal: 'Shows up without a reload',
      revision: 0,
      why: 'e2e: written from outside the page',
      by: 'agent',
      items: [{ key: 'one', title: 'One thing' }],
    },
  })
  expect(created.status()).toBe(200)

  // one feed read (CHANGES_POLL_MS, 5 s) and the quiet reload after it
  const row = page.locator('a.row', { hasText: ID })
  await expect(row).toHaveCount(1, { timeout: 15_000 })
  await expect(row).toContainText('Made by an agent')
  expect(navigated).toBe(0)

  // leave the ledger as the fixture made it for the other specs
  const read = await (await request.get(API)).json()
  const removed = await request.delete(`${API}?revision=${read.revision}`)
  expect(removed.status()).toBe(200)
  await expect(row).toHaveCount(0, { timeout: 15_000 })
})
