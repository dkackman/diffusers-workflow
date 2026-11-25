import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import RequestsPage from './RequestsPage.svelte'
import { summary } from '../request/fixtures'
import { workspace } from '../workspaceState.svelte'
import { api } from '../test/apiMock'
import { feed } from '../test/feedMock'
import { connection } from '../connection.svelte'
import { formatExact, formatRelative } from '../format'
import { quiet, useQuietClock } from '../test/quiet'

useQuietClock()

const data = vi.hoisted(() => ({
  requests: [] as unknown[],
  unreadable: undefined as string | undefined,
  error: null as Error | null,
  requestsFor: null as ((ws: string) => Promise<unknown>) | null,
}))
vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
api.listRequests.mockImplementation((ws: string) => {
  if (data.requestsFor) return data.requestsFor(ws)
  if (data.error) return Promise.reject(data.error)
  return Promise.resolve({
    workspace: 'studio',
    requests: data.requests,
    unreadable: data.unreadable,
  })
})

vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))

beforeEach(() => {
  workspace.current = 'studio'
  data.requestsFor = null
  data.error = null
  data.unreadable = undefined
  history.replaceState(null, '', '#/ws/studio/requests')
})
afterEach(cleanup)

const rows = () =>
  [...document.querySelectorAll('.list a.row')] as HTMLAnchorElement[]
const titles = () => rows().map((r) => r.querySelector('.title')?.textContent)

it('lists a request as one row: its pictures, what it wants, how far and what it cost', async () => {
  data.requests = [
    summary({
      cover: [
        { name: 'run/b.mp4', kind: 'video' },
        { name: 'run/a.png', kind: 'image' },
      ],
      ready: 1,
      updated_at: '2026-01-02T03:04:05+00:00',
    }),
  ]
  render(RequestsPage)
  const link = await screen.findByRole('link', { name: /Launch posters/ })
  expect(link.getAttribute('href')).toBe('#/ws/studio/requests/r1')
  expect(link.classList.contains('row')).toBe(true)
  expect(
    [...link.querySelectorAll('.cover img')].map((i) => i.getAttribute('src')),
  ).toEqual([
    '/api/gallery/run/b.mp4/thumbnail?workspace=studio',
    '/api/gallery/run/a.png/thumbnail?workspace=studio',
  ])
  expect(link.querySelectorAll('.cover .tile-mini')).toHaveLength(2)
  expect(link.querySelectorAll('.cover .videobadge')).toHaveLength(1)
  expect(screen.getByText('r1')).toBeTruthy()
  expect(link.querySelector('.tag')?.textContent).toBe('open')
  const line = link.querySelector('.attention') as HTMLElement
  expect(line.textContent?.trim()).toBe('1 planned, nothing running')
  expect(line.classList.contains('idle')).toBe(true)
  expect(
    screen.getByRole('img', { name: '2 done, 1 not started' }),
  ).toBeTruthy()
  expect(screen.getByText('2/3')).toBeTruthy()
  // measured minutes carry no tilde; what is left is an estimate
  expect(screen.getByText('4 min spent')).toBeTruthy()
  expect(screen.getByText('about 2 min left')).toBeTruthy()
  expect(link.querySelector('.updated')?.textContent).not.toBe('')
})

it('marks a request with a job running as the live row, and a failure in words', async () => {
  data.requests = [
    summary({ id: 'run', title: 'Running', status_counts: { running: 1 } }),
    summary({ id: 'bad', title: 'Broken', status_counts: { failed: 2 } }),
  ]
  render(RequestsPage)
  await screen.findByRole('link', { name: /Running/ })
  const [bad, run] = rows()
  expect(bad.querySelector('.attention')?.textContent?.trim()).toBe('2 failed')
  expect(bad.querySelector('.attention')?.classList.contains('failed')).toBe(
    true,
  )
  expect(bad.classList.contains('row-live')).toBe(false)
  expect(run.classList.contains('row-live')).toBe(true)
  expect(run.querySelector('.attention .pulse-dot')).toBeTruthy()
  expect(run.querySelector('.attention')?.textContent?.trim()).toBe('running')
})

it('puts what needs the user first, then what is in flight, then the newest', async () => {
  data.requests = [
    summary({ id: 'a', title: 'Quiet', status_counts: { done: 1 } }),
    summary({ id: 'b', title: 'Queued', status_counts: { queued: 1 } }),
    summary({
      id: 'c',
      title: 'Ready',
      status_counts: { planned: 1 },
      ready: 1,
    }),
  ]
  render(RequestsPage)
  await screen.findByRole('link', { name: /Quiet/ })
  expect(titles()).toEqual(['Ready', 'Queued', 'Quiet'])
})

it('shows open requests, and the closed ones when asked, kept in the hash', async () => {
  data.requests = [
    summary({ id: 'o', title: 'Open ask' }),
    summary({ id: 'd', title: 'Old ask', state: 'done' }),
  ]
  render(RequestsPage)
  await screen.findByRole('link', { name: /Open ask/ })
  expect(titles()).toEqual(['Open ask'])
  await fireEvent.change(screen.getByRole('combobox', { name: 'show' }), {
    target: { value: 'all' },
  })
  await waitFor(() => expect(titles()).toEqual(['Open ask', 'Old ask']))
  expect(location.hash).toBe('#/ws/studio/requests?show=all')
  // a closed request's line is its state tag: there is nothing it wants
  expect(rows()[1].querySelector('.tag')?.textContent).toBe('done')
  expect(rows()[1].querySelector('.attention')?.textContent?.trim()).toBe('')
})

it('reads the filter from the hash it was opened with', async () => {
  history.replaceState(null, '', '#/ws/studio/requests?show=all')
  data.requests = [summary({ id: 'd', title: 'Old ask', state: 'abandoned' })]
  render(RequestsPage)
  await screen.findByRole('link', { name: /Old ask/ })
})

it('with only closed requests, says there are none open and offers them', async () => {
  data.requests = [
    summary({ id: 'd', title: 'Old ask', state: 'done' }),
    summary({ id: 'e', title: 'Older ask', state: 'abandoned' }),
  ]
  render(RequestsPage)
  await screen.findByText(/No open requests\./)
  await fireEvent.click(screen.getByRole('button', { name: 'show closed (2)' }))
  await waitFor(() => expect(rows()).toHaveLength(2))
})

it('says how a request comes to be when there are none, in no tool names', async () => {
  data.requests = []
  render(RequestsPage)
  const empty = await screen.findByText(/No requests in this workspace yet\./)
  expect(empty.textContent?.replace(/\s+/g, ' ')).toContain(
    'One appears when you ask your agent for something and it writes its plan here.',
  )
  expect(empty.textContent).not.toMatch(/set_request/)
})

it('a list that fails to load says why, and Retry asks again', async () => {
  data.error = new Error('boom')
  render(RequestsPage)
  expect((await screen.findByRole('alert')).textContent).toContain(
    "Couldn't load requests",
  )
  data.error = null
  data.requests = [summary()]
  await fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await screen.findByRole('link', { name: /Launch posters/ })
})

it('recovers when the server returns, by the connection epoch rising', async () => {
  data.error = new Error('down')
  render(RequestsPage)
  await screen.findByText(/Couldn't load requests/)
  data.error = null
  data.requests = [summary()]
  connection.epoch++
  await screen.findByRole('link', { name: /Launch posters/ })
  expect(screen.queryByText(/Couldn't load requests/)).toBeNull()
})

it('says when the requests file is there and cannot be read', async () => {
  data.requests = []
  data.unreadable = 'requests.json is not a ledger'
  render(RequestsPage)
  expect((await screen.findByRole('alert')).textContent).toContain(
    'requests.json is not a ledger',
  )
})

it("drops another workspace's list that lands after a switch", async () => {
  let land: (reply: unknown) => void = () => {}
  data.requestsFor = (ws) =>
    ws === 'studio'
      ? new Promise((resolve) => (land = resolve))
      : Promise.resolve({
          workspace: ws,
          requests: [summary({ id: 'o1', title: 'Other ask' })],
        })
  render(RequestsPage)
  workspace.current = 'other'
  await screen.findByRole('link', { name: /Other ask/ })
  land({
    workspace: 'studio',
    requests: [summary({ id: 's1', title: 'Studio ask' })],
  })
  await quiet()
  expect(screen.queryByText('Studio ask')).toBeNull()
  expect(screen.getByRole('link', { name: /Other ask/ })).toBeTruthy()
})

it('a request written from outside shows when the feed says requests moved, with no spinner', async () => {
  data.requests = [summary()]
  render(RequestsPage)
  await screen.findByRole('link', { name: /Launch posters/ })
  expect(feed.watched('studio')).toEqual(['requests'])
  data.requests = [
    summary(),
    summary({
      id: 'r2',
      title: 'Second launch',
      updated_at: '2026-02-02T00:00:00+00:00',
    }),
  ]
  let answer = () => {}
  data.requestsFor = () =>
    new Promise((resolve) => {
      answer = () => resolve({ workspace: 'studio', requests: data.requests })
    })
  feed.move('studio', 'requests')
  // the list stays while it asks: no skeleton, no blanking
  expect(screen.queryByLabelText('loading requests')).toBeNull()
  expect(rows()).toHaveLength(1)
  answer()
  await waitFor(() => expect(rows()).toHaveLength(2))
  expect(titles()).toContain('Second launch')
})

it('a quiet reload that fails keeps the list under a refresh line', async () => {
  data.requests = [summary()]
  render(RequestsPage)
  await screen.findByRole('link', { name: /Launch posters/ })
  data.error = new Error('503')
  feed.move('studio', 'requests')
  await new Promise((resolve) => setTimeout(resolve, 0))
  expect(rows()).toHaveLength(1)
  expect((await screen.findByRole('alert')).textContent).toContain(
    "Couldn't refresh requests",
  )
})

it('another kind moving, or another workspace, reloads nothing', async () => {
  let calls = 0
  data.requestsFor = () => {
    calls++
    return Promise.resolve({ workspace: 'studio', requests: [summary()] })
  }
  render(RequestsPage)
  await screen.findByRole('link', { name: /Launch posters/ })
  feed.move('studio', 'gallery')
  feed.move('other', 'requests')
  expect(calls).toBe(1)
})

it('watches the workspace it switched to, and only it', async () => {
  data.requests = [summary()]
  render(RequestsPage)
  await screen.findByRole('link', { name: /Launch posters/ })
  workspace.current = 'other'
  await waitFor(() => expect(feed.watched('other')).toEqual(['requests']))
  expect(feed.watched('studio')).toEqual([])
})

it('a request an agent edited is marked', async () => {
  data.requests = [
    summary({ id: 'r1', title: 'Launch posters' }),
    summary({ id: 'r2', title: 'Second launch' }),
  ]
  render(RequestsPage)
  await screen.findByRole('link', { name: /Second launch/ })
  expect(document.querySelectorAll('.row-changed')).toHaveLength(0)
  data.requests = [
    summary({ id: 'r1', title: 'Launch posters' }),
    summary({ id: 'r2', title: 'Second launch, recut' }),
  ]
  feed.move('studio', 'requests')
  await screen.findByRole('link', { name: /Second launch, recut/ })
  const marked = [...document.querySelectorAll('.list a.row-changed')]
  expect(marked.map((r) => r.getAttribute('href'))).toEqual([
    '#/ws/studio/requests/r2',
  ])
})

it('a row names its waiting notes', async () => {
  data.requests = [
    summary({
      id: 'n',
      title: 'Noted',
      needs: {
        held: 0,
        ready: 1,
        failed: 0,
        revised: 0,
        choose: 0,
        accepted_missing: 0,
        notes_waiting: 2,
      },
    }),
    summary({ id: 'm', title: 'Plain' }),
  ]
  render(RequestsPage)
  await screen.findByRole('link', { name: /Noted/ })
  const [noted, plain] = rows()
  expect(noted.querySelector('.figures')?.textContent).toMatch(
    /2\s+notes waiting for the agent/,
  )
  expect(noted.querySelector('.attention')?.textContent?.trim()).toBe(
    'Needs you: 1 ready to run.',
  )
  expect(plain.querySelector('.figures')?.textContent).not.toMatch(/waiting/)
})

it('a row says when it was last written, and the hover says it is the last write', async () => {
  const at = '2026-01-02T03:04:05+00:00'
  data.requests = [summary({ updated_at: at })]
  render(RequestsPage)
  await screen.findByRole('link', { name: /Launch posters/ })
  const time = rows()[0].querySelector('.updated time') as HTMLElement
  expect(time.textContent?.trim()).toBe(
    `changed ${formatRelative(Date.parse(at) / 1000, Date.now() / 1000)}`,
  )
  expect(time.getAttribute('title')).toBe(
    `last written ${formatExact(Date.parse(at) / 1000)}`,
  )
})

it('a row whose stored time does not parse still renders', async () => {
  data.requests = [summary({ updated_at: 'yesterday-ish' })]
  render(RequestsPage)
  await screen.findByRole('link', { name: /Launch posters/ })
  const updated = rows()[0].querySelector('.updated') as HTMLElement
  expect(updated.textContent?.trim()).toBe('unknown time')
  expect(updated.querySelector('[datetime]')).toBeNull()
})
