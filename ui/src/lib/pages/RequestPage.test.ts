import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import RequestPage from './RequestPage.svelte'
import { workspace } from '../workspaceState.svelte'
import { ACTIVE_MS, IDLE_MS } from '../request/refresh'
import { doneItem, entry, item, request } from '../request/fixtures'
import type { LedgerRequest } from '../types'
import { feed } from '../test/feedMock'
import { readSnapshot, snapshotOf, writeSnapshot } from '../request/planDiff'

const data = vi.hoisted(() => ({
  request: null as unknown,
  getError: null as { status: number; message: string } | null,
  putRequest: vi.fn(),
  addNote: vi.fn(),
  notesChannel: vi.fn(),
}))
vi.mock('../api', async () => ({
  ApiError: (await import('../apiErrors')).ApiError,
  api: {
    getRequest: vi.fn(async () => {
      if (data.getError) {
        const { ApiError } = await import('../api')
        throw new ApiError(data.getError.message, data.getError.status)
      }
      return data.request
    }),
    putRequest: (...args: unknown[]) => data.putRequest(...args),
    addNote: (...args: unknown[]) => data.addNote(...args),
    notesChannel: (...args: unknown[]) => data.notesChannel(...args),
  },
}))
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))
vi.mock('../toast', () => ({
  notify: {
    error: vi.fn(),
    failure: vi.fn(),
    success: vi.fn(),
    undo: vi.fn(),
    news: vi.fn(),
  },
}))

const proof = { name: 'run/a.png', kind: 'image' as const }
const make = (): LedgerRequest =>
  request(
    [
      doneItem('a', {
        title: 'First poster',
        estimate_minutes: 2,
        workflow: 'poster',
        result: { job: 'job-a', run_id: 'run-a', output: proof },
        spent_minutes: 1.5,
      }),
      item('b', {
        title: 'Second poster',
        depends_on: ['a'],
        workflow: 'poster',
        ready: true,
        estimate_minutes: 4,
      }),
      item('x', { title: 'Cut scene', dropped: true, status: 'dropped' }),
    ],
    {
      prompt: 'make me\n  three posters',
      spent_minutes: 5,
      remaining_minutes: 7,
      history: [
        entry(1, 'agent', 'opened', { goal: 'First goal', state: 'open' }),
        entry(2, 'agent', 'the ask grew', { goal: 'Three finished posters' }),
        entry(3, 'user', 'note added'),
      ],
      agent_wrote_at: entry(2, 'agent', '').at,
    },
  )

beforeEach(() => {
  sessionStorage.clear()
  workspace.current = 'studio'
  data.request = make()
  data.getError = null
  data.putRequest.mockReset()
  data.addNote.mockReset()
  data.notesChannel.mockReset()
  data.notesChannel.mockResolvedValue({ listening: false, last_seen: null })
  vi.clearAllMocks()
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

const loaded = () => screen.findByRole('heading', { name: 'Launch posters' })
const row = (title: string) =>
  screen.getByRole('button', { name: title }).closest('li') as HTMLElement
const live = () => screen.getByTestId('request-live').textContent
// The revision is in the rail, with the plan's changes
const revision = () => document.querySelector('.rail .now')?.textContent
const atRevision = (n: number) =>
  waitFor(() => expect(revision()).toBe(`now at r${n}`))
const chips = () =>
  [...document.querySelectorAll('ol.items .row .chip')].map(
    (e) => e.textContent,
  )

/** Open the note box of the item called `title`, and return it */
async function openNote(title: string) {
  await loaded()
  await fireEvent.click(screen.getByRole('button', { name: title }))
  await fireEvent.click(
    within(row(title)).getByRole('button', {
      name: 'Add a note for the agent',
    }),
  )
  return screen.getByLabelText(
    `Note for the agent on ${title}`,
  ) as HTMLTextAreaElement
}
const saveNote = (title: string) =>
  fireEvent.click(within(row(title)).getByRole('button', { name: 'Save note' }))

it('leads with where the request stands, then the plan, the ask and its changes', async () => {
  const { container } = render(RequestPage, { requestId: 'r1' })
  await loaded()
  expect(container.querySelector('.standing')?.textContent).toBe(
    'Nothing is running. 1 of 2 done.',
  )
  expect(container.querySelector('.needs')?.textContent).toBe(
    'Needs you: 1 ready to run.',
  )
  const titles = [...container.querySelectorAll('ol.items .title .name')].map(
    (e) => e.textContent,
  )
  expect(titles).toEqual(['First poster', 'Second poster'])
  expect(chips()).toEqual(['done', 'ready'])
  expect(
    [...container.querySelectorAll('ol.items .figure')].map(
      (e) => e.textContent,
    ),
  ).toEqual(['1.5 min', 'est. about 4 min'])
  expect(
    container
      .querySelector('.figures')
      ?.textContent?.replace(/\s+/g, ' ')
      .trim(),
  ).toBe('5 min spent about 7 min left')
  // the proof comes from the ledger's own output: no job is asked for
  expect(
    screen
      .getByRole('img', { name: 'result of First poster' })
      .getAttribute('src'),
  ).toBe('/api/gallery/run/a.png/thumbnail?workspace=studio')
  const rail = container.querySelector('.rail') as HTMLElement
  expect(within(rail).getByText(/three posters/).textContent).toBe(
    'make me\n  three posters',
  )
  expect(
    within(rail).getAllByText('Three finished posters').length,
  ).toBeGreaterThan(0)
  expect(
    [...rail.querySelectorAll('.changes ol > li')].map(
      (e) => e.querySelector('.num')?.textContent,
    ),
  ).toEqual(['r3', 'r2', 'r1'])
  expect(rail.querySelector('.now')?.textContent).toBe('now at r3')
})

it('sets the tab title to the request, with Requests as its back link', async () => {
  render(RequestPage, { requestId: 'r1' })
  await loaded()
  expect(document.title).toBe('Launch posters · diffusers-workflow')
  expect(
    screen.getByRole('link', { name: '← Requests' }).getAttribute('href'),
  ).toMatch(/\/requests$/)
  expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1)
})

it('keeps dropped items in a closed group under the plan', async () => {
  const { container } = render(RequestPage, { requestId: 'r1' })
  await loaded()
  const group = container.querySelector(
    'details.droppedgroup',
  ) as HTMLDetailsElement
  expect(group.open).toBe(false)
  expect(group.querySelector('summary')?.textContent).toBe('Dropped (1)')
  expect(group.querySelector('.name')?.textContent).toBe('Cut scene')
  expect(container.querySelectorAll('ol.items > li')).toHaveLength(2)
})

it('says so when the agent has planned nothing yet', async () => {
  data.request = request([], { history: [entry(1, 'agent', 'opened')] })
  const { container } = render(RequestPage, { requestId: 'r1' })
  await loaded()
  expect(
    screen.getByText(
      'The agent has not planned any items for this request yet.',
    ),
  ).toBeTruthy()
  expect(container.querySelector('.list')).toBeNull()
})

it('says a missing request no longer exists, with the way back', async () => {
  data.getError = { status: 404, message: 'not found' }
  render(RequestPage, { requestId: 'gone' })
  await screen.findByText(
    /This request no longer exists\. The agent may have removed it\./,
  )
  expect(
    screen.getByRole('link', { name: '← Requests' }).getAttribute('href'),
  ).toBe('#/ws/studio/requests')
  expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1)
  expect(screen.getByRole('heading', { level: 1 }).textContent).toContain(
    'Request',
  )
  expect(document.title).toBe('Request · diffusers-workflow')
})

it('names itself and links back while the request loads', async () => {
  const { api } = await import('../api')
  let land: (reply: unknown) => void = () => {}
  vi.mocked(api.getRequest).mockImplementationOnce(
    () => new Promise((resolve) => (land = resolve)) as never,
  )
  render(RequestPage, { requestId: 'r1' })
  expect(screen.getByLabelText('loading request')).toBeTruthy()
  expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1)
  expect(
    screen.getByRole('link', { name: '← Requests' }).getAttribute('href'),
  ).toBe('#/ws/studio/requests')
  expect(document.title).toBe('Request · diffusers-workflow')
  land(make())
  await loaded()
  expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1)
})

// The engine's own words, which only the folded Details may show
const RAW = 'ledger read failed: sqlite3.OperationalError database is locked'
const outsideDetails = (el: Element) => {
  const copy = el.cloneNode(true) as Element
  copy.querySelectorAll('details').forEach((d) => d.remove())
  return copy.textContent ?? ''
}

it('a first load that fails says why, and Retry asks again', async () => {
  data.getError = { status: 503, message: RAW }
  render(RequestPage, { requestId: 'r1' })
  const alert = await screen.findByRole('alert')
  expect(alert.textContent).toContain(
    "Couldn't load this request: The server hit an error",
  )
  expect(outsideDetails(alert)).not.toContain(RAW)
  expect(alert.querySelector('details pre')?.textContent).toBe(RAW)
  expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1)
  expect(
    screen.getByRole('link', { name: '← Requests' }).getAttribute('href'),
  ).toBe('#/ws/studio/requests')
  expect(document.title).toBe('Request · diffusers-workflow')
  data.getError = null
  await fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await loaded()
  expect(screen.queryByRole('alert')).toBeNull()
})

it('a segment of the progress bar goes to its row', async () => {
  render(RequestPage, { requestId: 'r1' })
  await loaded()
  await fireEvent.click(
    screen.getByRole('button', { name: 'Second poster: ready' }),
  )
  expect(document.activeElement).toBe(
    screen.getByRole('button', { name: 'Second poster' }),
  )
})

/** The plan as read after the user's note on `key` (null: the request) */
function noted(key: string | null, text: string, revision = 4): LedgerRequest {
  const next = make()
  next.revision = revision
  const note = { id: 'n1', text, created: AT, at: AT }
  if (key === null) next.notes = [note]
  else
    next.items = next.items.map((i) =>
      i.key === key ? { ...i, notes: [note] } : i,
    )
  return next
}
const AT = '2026-01-04T00:00:00+00:00'

it('saves a note through the note route, never the plan, and reads the plan again', async () => {
  const { api } = await import('../api')
  data.addNote.mockImplementation(async () => {
    data.request = noted('b', 'needs more red')
    return { revision: 4 }
  })
  render(RequestPage, { requestId: 'r1' })
  const box = await openNote('Second poster')
  await fireEvent.input(box, { target: { value: 'needs more red' } })
  const reads = vi.mocked(api.getRequest).mock.calls.length
  await saveNote('Second poster')
  await atRevision(4)
  expect(data.addNote).toHaveBeenCalledWith('studio', {
    request: 'r1',
    item: 'b',
    text: 'needs more red',
  })
  expect(data.putRequest).not.toHaveBeenCalled()
  expect(vi.mocked(api.getRequest).mock.calls.length).toBe(reads + 1)
  await within(row('Second poster')).findByText('needs more red')
})

it("saves the request's own note with no item", async () => {
  data.addNote.mockImplementation(async () => {
    data.request = noted(null, 'make it all warmer')
    return { revision: 4 }
  })
  const { container } = render(RequestPage, { requestId: 'r1' })
  await loaded()
  const head = container.querySelector('.reqhead') as HTMLElement
  await fireEvent.click(
    within(head).getByRole('button', { name: 'Add a note for the agent' }),
  )
  await fireEvent.input(
    screen.getByLabelText('Note for the agent on this request'),
    { target: { value: 'make it all warmer' } },
  )
  await fireEvent.click(within(head).getByRole('button', { name: 'Save note' }))
  await within(head).findByText('make it all warmer')
  await waitFor(() =>
    expect(
      screen.queryByLabelText('Note for the agent on this request'),
    ).toBeNull(),
  )
  expect(data.addNote.mock.calls[0][1]).toEqual({
    request: 'r1',
    text: 'make it all warmer',
  })
  expect(data.putRequest).not.toHaveBeenCalled()
  expect(
    within(head).getByText(/The agent isn't connected\. It reads/),
  ).toBeTruthy()
})

it('a note the server refuses says why beside the draft, with a reload', async () => {
  const { ApiError } = await import('../api')
  data.addNote.mockRejectedValue(
    new ApiError("item 'b' of request 'r1' was dropped", 422, 'item_dropped'),
  )
  render(RequestPage, { requestId: 'r1' })
  const box = await openNote('Second poster')
  await fireEvent.input(box, { target: { value: 'my draft' } })
  await saveNote('Second poster')
  expect((await screen.findByRole('alert')).textContent).toContain(
    'This item was dropped or removed from the plan, so the note is not saved.',
  )
  expect(screen.getByRole('button', { name: 'Reload the plan' })).toBeTruthy()
  expect(box.value).toBe('my draft')
})

it('links to its jobs and outputs through the req filter', async () => {
  const { container } = render(RequestPage, { requestId: 'r1' })
  await loaded()
  const head = within(container.querySelector('.reqhead') as HTMLElement)
  expect(head.getByText('its jobs').getAttribute('href')).toBe(
    `#/ws/${workspace.current}/jobs?req=r1`,
  )
  expect(head.getByText('its outputs').getAttribute('href')).toBe(
    `#/ws/${workspace.current}/gallery?req=r1`,
  )
})

it('drops a request that lands after the page moved to another', async () => {
  const { api } = await import('../api')
  let land: (reply: unknown) => void = () => {}
  vi.mocked(api.getRequest).mockImplementationOnce(
    () => new Promise((resolve) => (land = resolve)) as never,
  )
  const view = render(RequestPage, { requestId: 'r1' })
  data.request = { ...make(), id: 'r2', title: 'Second ask' }
  await view.rerender({ requestId: 'r2' })
  await screen.findByText('Second ask')
  land(make())
  await new Promise((r) => setTimeout(r, 10))
  expect(screen.queryByText('Launch posters')).toBeNull()
  expect(screen.getByText('Second ask')).toBeTruthy()
})

// Only the interval is faked, so findBy/waitFor keep their own real timers
const fakeInterval = () =>
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })

/** The plan after the agent queued the first job of item b, at revision 4 */
function queued(): LedgerRequest {
  const next = make()
  next.revision = 4
  next.items[1] = {
    ...next.items[1],
    status: 'queued',
    ready: false,
    attempts: [{ job: 'job-b', status: 'queued', minutes: null, output: null }],
  }
  return next
}

it('refreshes an idle open request, and a note draft survives it', async () => {
  fakeInterval()
  const { api } = await import('../api')
  render(RequestPage, { requestId: 'r1' })
  const box = await openNote('Second poster')
  await fireEvent.input(box, { target: { value: 'half a thought' } })
  const next = queued()
  data.request = next
  const calls = vi.mocked(api.getRequest).mock.calls.length
  vi.advanceTimersByTime(IDLE_MS)
  await atRevision(4)
  expect(vi.mocked(api.getRequest).mock.calls.length).toBe(calls + 1)
  await waitFor(() => expect(chips()).toEqual(['done', 'queued']))
  expect(
    within(row('Second poster'))
      .getByRole('link', { name: 'Open job' })
      .getAttribute('href'),
  ).toBe('#/ws/studio/jobs/job-b')
  expect(box.value).toBe('half a thought')
  expect(document.body.contains(box)).toBe(true)
})

it('tells a screen reader what a refresh changed, and nothing on first load', async () => {
  fakeInterval()
  render(RequestPage, { requestId: 'r1' })
  await loaded()
  const region = screen.getByTestId('request-live')
  expect(region.getAttribute('aria-live')).toBe('polite')
  expect(live()).toBe('')
  data.request = queued()
  vi.advanceTimersByTime(IDLE_MS)
  await waitFor(() => expect(live()).toBe('Second poster was queued.'))
  const done = queued()
  done.items[1] = doneItem('b', {
    title: 'Second poster',
    depends_on: ['a'],
    workflow: 'poster',
  })
  data.request = done
  vi.advanceTimersByTime(ACTIVE_MS)
  await waitFor(() => expect(live()).toBe('Second poster finished.'))
})

// One resource reads the request: the feed's `requests` kind always, and a
// timer beside it while the request can change on its own (refreshInterval)
it('an open request refreshes on its interval and a closed one on the feed', async () => {
  fakeInterval()
  const { api } = await import('../api')
  const reads = () => vi.mocked(api.getRequest).mock.calls.length
  data.request = { ...make(), state: 'done', revision: 4 }
  render(RequestPage, { requestId: 'r1' })
  await loaded()
  await waitFor(() => expect(feed.watched('studio')).toEqual(['requests']))
  // closed with nothing in flight: no timer
  let calls = reads()
  vi.advanceTimersByTime(IDLE_MS * 3)
  expect(reads()).toBe(calls)
  // another kind moving asks nothing
  feed.move('studio', 'gallery')
  expect(reads()).toBe(calls)
  // an agent reopens it: the feed says requests moved
  data.request = { ...make(), state: 'open', revision: 5 }
  feed.move('studio', 'requests')
  await atRevision(5)
  // open again, it is on its idle timer, and still on the feed
  calls = reads()
  vi.advanceTimersByTime(IDLE_MS)
  expect(reads()).toBe(calls + 1)
  expect(feed.watched('studio')).toEqual(['requests'])
  // a job in flight: the active interval
  data.request = { ...queued(), revision: 6 }
  feed.move('studio', 'requests')
  await atRevision(6)
  calls = reads()
  vi.advanceTimersByTime(ACTIVE_MS)
  expect(reads()).toBe(calls + 1)
})

it('a feed read of a closed request at an older revision does not replace it', async () => {
  fakeInterval()
  data.request = { ...make(), state: 'done', revision: 6 }
  render(RequestPage, { requestId: 'r1' })
  await atRevision(6)
  await waitFor(() => expect(feed.watched('studio')).toEqual(['requests']))
  data.request = { ...make(), state: 'open', revision: 5 }
  feed.move('studio', 'requests')
  await new Promise((resolve) => setTimeout(resolve, 0))
  expect(revision()).toBe('now at r6')
})

it('a refresh answered after a save does not put the old revision back', async () => {
  const { api } = await import('../api')
  render(RequestPage, { requestId: 'r1' })
  const box = await openNote('Second poster')
  // The feed's read goes out and hangs
  let land: (reply: unknown) => void = () => {}
  vi.mocked(api.getRequest).mockImplementationOnce(
    () => new Promise((resolve) => (land = resolve)) as never,
  )
  feed.move('studio', 'requests')
  // The save lands first, and the read after it is at revision 4
  data.addNote.mockImplementation(async () => {
    data.request = noted('b', 'more red')
    return { revision: 4 }
  })
  await fireEvent.input(box, { target: { value: 'more red' } })
  await saveNote('Second poster')
  await atRevision(4)
  // then the feed's read, still carrying revision 3 and no note
  land(make())
  await new Promise((r) => setTimeout(r, 10))
  expect(revision()).toBe('now at r4')
  expect(within(row('Second poster')).getByText('more red')).toBeTruthy()
})

it('an item an agent edited is marked', async () => {
  render(RequestPage, { requestId: 'r1' })
  await loaded()
  // nothing is marked on the first read
  expect(document.querySelector('li.row-changed')).toBeNull()
  const next = make()
  next.revision = 4
  next.items[1] = { ...next.items[1], title: 'Second poster, in red' }
  data.request = next
  feed.move('studio', 'requests')
  await atRevision(4)
  await waitFor(() =>
    expect(row('Second poster, in red').classList).toContain('row-changed'),
  )
  expect(row('First poster').classList).not.toContain('row-changed')
})

it('a switch to another request marks against that request, never the one left', async () => {
  const first = make()
  first.revision = 6
  data.request = first
  const view = render(RequestPage, { requestId: 'r1' })
  await atRevision(6)
  // another request, younger: its revision is below the one left
  const second = { ...make(), id: 'r2', title: 'Second ask', revision: 1 }
  data.request = second
  await view.rerender({ requestId: 'r2' })
  await screen.findByText('Second ask')
  await atRevision(1)
  // nothing is marked by the switch itself
  expect(document.querySelector('li.row-changed')).toBeNull()
  const edited = structuredClone(second)
  edited.revision = 2
  edited.items[1] = { ...edited.items[1], title: 'Second poster, in red' }
  data.request = edited
  feed.move('studio', 'requests')
  await atRevision(2)
  await waitFor(() =>
    expect(row('Second poster, in red').classList).toContain('row-changed'),
  )
  expect(document.querySelectorAll('li.row-changed')).toHaveLength(1)
})

it('a switch away from a request that failed shows the skeleton, not its failure', async () => {
  data.getError = { status: 404, message: 'no such request' }
  const view = render(RequestPage, { requestId: 'gone' })
  await screen.findByText(/This request no longer exists/)
  data.getError = null
  const { api } = await import('../api')
  let land: (reply: unknown) => void = () => {}
  vi.mocked(api.getRequest).mockImplementationOnce(
    () => new Promise((resolve) => (land = resolve)) as never,
  )
  await view.rerender({ requestId: 'r1' })
  expect(screen.queryByText(/This request no longer exists/)).toBeNull()
  expect(screen.getByLabelText('loading request')).toBeTruthy()
  land(make())
  await loaded()
})

it('a refresh sent before a save and answered after it does not undo the save', async () => {
  fakeInterval()
  const { api } = await import('../api')
  render(RequestPage, { requestId: 'r1' })
  const box = await openNote('Second poster')
  // The refresh goes out and hangs
  let land: (reply: unknown) => void = () => {}
  vi.mocked(api.getRequest).mockImplementationOnce(
    () => new Promise((resolve) => (land = resolve)) as never,
  )
  vi.advanceTimersByTime(IDLE_MS)
  // The save lands first, and the read after it is at revision 4
  data.addNote.mockImplementation(async () => {
    data.request = noted('b', 'more red')
    return { revision: 4 }
  })
  await fireEvent.input(box, { target: { value: 'more red' } })
  await saveNote('Second poster')
  await atRevision(4)
  // then the refresh, still carrying revision 3 and no note
  land(make())
  await new Promise((r) => setTimeout(r, 10))
  expect(revision()).toBe('now at r4')
  expect(within(row('Second poster')).getByText('more red')).toBeTruthy()
})

it('a refresh that fails keeps the plan and the draft, and says how old it is', async () => {
  fakeInterval()
  const { notify } = await import('../toast')
  render(RequestPage, { requestId: 'r1' })
  const box = await openNote('Second poster')
  await fireEvent.input(box, { target: { value: 'my draft' } })
  data.getError = { status: 503, message: RAW }
  vi.advanceTimersByTime(IDLE_MS)
  const line = await screen.findByRole('status')
  expect(line.textContent).toMatch(
    /Showing the plan as of .+\. Couldn't refresh: The server hit an error/,
  )
  expect(outsideDetails(line)).not.toContain(RAW)
  expect(line.querySelector('details pre')?.textContent).toBe(RAW)
  expect(notify.error).not.toHaveBeenCalled()
  expect(box.value).toBe('my draft')
  expect(screen.getByRole('heading', { name: 'Launch posters' })).toBeTruthy()
  // Retry reads again; a read that works takes the line away
  data.getError = null
  await fireEvent.click(within(line).getByRole('button', { name: 'Retry' }))
  await waitFor(() => expect(screen.queryByRole('status')).toBeNull())
  expect(box.value).toBe('my draft')
})

it('says once when the agent answers a waiting note, and the toast goes to it', async () => {
  fakeInterval()
  const { notify } = await import('../toast')
  // b's note waits; a's was answered before the page opened
  const waiting = noted('b', 'more red', 3)
  waiting.items[0].notes = [
    {
      id: 'n0',
      text: 'old',
      created: '2026-01-01T00:00:00+00:00',
      at: '2026-01-01T00:00:00+00:00',
    },
  ]
  data.request = waiting
  render(RequestPage, { requestId: 'r1' })
  await loaded()
  expect(notify.news).not.toHaveBeenCalled()
  // the agent writes after it
  const answered = structuredClone(waiting)
  answered.revision = 4
  answered.history.push(entry(4, 'agent', 'retook b with more red'))
  answered.agent_wrote_at = '2026-01-05T00:00:00+00:00'
  data.request = answered
  vi.advanceTimersByTime(IDLE_MS)
  await atRevision(4)
  await waitFor(() => expect(notify.news).toHaveBeenCalledTimes(1))
  const [text, action] = vi.mocked(notify.news).mock.calls[0]
  expect(text).toBe(
    'The agent updated Launch posters after your note: “retook b with more red”',
  )
  // a later read says nothing more
  data.request = { ...answered, revision: 5 }
  vi.advanceTimersByTime(IDLE_MS)
  await atRevision(5)
  expect(notify.news).toHaveBeenCalledTimes(1)
  // the toast opens the row and lands on the note
  expect(
    screen
      .getByRole('button', { name: 'Second poster' })
      .getAttribute('aria-expanded'),
  ).toBe('false')
  action.onClick()
  // focus is no mutation for waitFor to see, and its interval is faked
  await new Promise((r) => setTimeout(r, 20))
  expect(document.activeElement?.id).toBe('request-note-n1')
  expect(
    screen
      .getByRole('button', { name: 'Second poster' })
      .getAttribute('aria-expanded'),
  ).toBe('true')
})

it('tells again for the same note id on another request, and never on its first read', async () => {
  fakeInterval()
  const { notify } = await import('../toast')
  const answerOf = (r: LedgerRequest, rev: number, what: string) => {
    const a = structuredClone(r)
    a.revision = rev
    a.history.push(entry(rev, 'agent', what))
    a.agent_wrote_at = '2026-01-05T00:00:00+00:00'
    return a
  }
  const first = noted('b', 'more red', 3)
  data.request = first
  const view = render(RequestPage, { requestId: 'r1' })
  await loaded()
  data.request = answerOf(first, 4, 'retook b')
  vi.advanceTimersByTime(IDLE_MS)
  await atRevision(4)
  await waitFor(() => expect(notify.news).toHaveBeenCalledTimes(1))
  // another request, whose first note has the same id, already waiting
  const second = {
    ...noted('b', 'more blue', 3),
    id: 'r2',
    title: 'Second ask',
  }
  data.request = second
  await view.rerender({ requestId: 'r2' })
  await screen.findByText('Second ask')
  expect(notify.news).toHaveBeenCalledTimes(1)
  // the agent answers it: told, though n1 was told for r1
  data.request = answerOf(second, 4, 'retook b in blue')
  vi.advanceTimersByTime(IDLE_MS)
  await atRevision(4)
  await waitFor(() => expect(notify.news).toHaveBeenCalledTimes(2))
})

it('a save that fails for the network says so, and keeps the draft', async () => {
  const { notify } = await import('../toast')
  data.addNote.mockRejectedValue(new TypeError('Failed to fetch'))
  render(RequestPage, { requestId: 'r1' })
  const box = await openNote('Second poster')
  await fireEvent.input(box, { target: { value: 'my draft' } })
  await saveNote('Second poster')
  await waitFor(() =>
    expect(notify.failure).toHaveBeenCalledWith(
      'save the note',
      expect.any(TypeError),
    ),
  )
  expect(box.value).toBe('my draft')
  expect(document.body.contains(box)).toBe(true)
  expect(screen.getByRole('heading', { name: 'Launch posters' })).toBeTruthy()
})

// What the agent changed: under its write while the page is open, and
// since this browser last looked, from a snapshot per workspace and id
const diffOf = (revision: number) =>
  [...document.querySelectorAll(`#plan-change-r${revision} .diff li`)].map(
    (e) => e.textContent,
  )
const sinceLines = () =>
  [...document.querySelectorAll('.since li')].map((e) => e.textContent)
/** An agent write at r4 that renames `key`'s item */
function agentRename(index: number, title: string): LedgerRequest {
  const next = make()
  next.revision = 4
  next.items[index] = { ...next.items[index], title }
  next.history = [...next.history, entry(4, 'agent', 'a clearer name')]
  return next
}
/** The request as an earlier visit saw it, at r1, its second item seeded */
function seenAtR1(ws = 'studio', revision = 1) {
  const earlier = make()
  earlier.revision = revision
  earlier.items[1] = { ...earlier.items[1], arguments: { seed: 1 } }
  writeSnapshot(ws, 'r1', snapshotOf(earlier))
}

it('a write read while open lists its changes under it', async () => {
  localStorage.clear()
  render(RequestPage, { requestId: 'r1' })
  await loaded()
  expect(document.querySelector('.diff')).toBeNull()
  data.request = agentRename(1, 'Second poster, in red')
  feed.move('studio', 'requests')
  await atRevision(4)
  await waitFor(() =>
    expect(diffOf(4)).toEqual([
      'Renamed Second poster to Second poster, in red',
    ]),
  )
  // the user's own write brings no lines
  expect(diffOf(3)).toEqual([])
})

it('since you last looked lists the changes from the snapshot', async () => {
  localStorage.clear()
  seenAtR1()
  render(RequestPage, { requestId: 'r1' })
  await loaded()
  expect(
    await screen.findByText('Since you last looked (r1 to r3)'),
  ).toBeTruthy()
  expect(sinceLines()).toEqual(['Changed Second poster: its arguments (seed)'])
  // the read was taken as the new snapshot
  expect(readSnapshot('studio', 'r1')?.revision).toBe(3)
  await fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }))
  expect(document.querySelector('.since')).toBeNull()
  expect(readSnapshot('studio', 'r1')?.revision).toBe(3)
})

it('a stored revision above the current one is discarded', async () => {
  localStorage.clear()
  seenAtR1('studio', 9)
  render(RequestPage, { requestId: 'r1' })
  await loaded()
  await atRevision(3)
  expect(document.querySelector('.since')).toBeNull()
  expect(readSnapshot('studio', 'r1')?.revision).toBe(3)
})

it('storage that throws shows no diff', async () => {
  const get = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
    throw new Error('forbidden')
  })
  const set = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
    throw new Error('forbidden')
  })
  try {
    render(RequestPage, { requestId: 'r1' })
    await loaded()
    await atRevision(3)
    expect(document.querySelector('.since')).toBeNull()
    // and what the agent writes while the page is open is still listed
    data.request = agentRename(0, 'Poster one')
    feed.move('studio', 'requests')
    await atRevision(4)
    await waitFor(() =>
      expect(diffOf(4)).toEqual(['Renamed First poster to Poster one']),
    )
  } finally {
    get.mockRestore()
    set.mockRestore()
  }
})

it('snapshots are per workspace', async () => {
  localStorage.clear()
  seenAtR1('other')
  render(RequestPage, { requestId: 'r1' })
  await loaded()
  await atRevision(3)
  expect(document.querySelector('.since')).toBeNull()
  expect(readSnapshot('other', 'r1')?.revision).toBe(1)
  expect(readSnapshot('studio', 'r1')?.revision).toBe(3)
})

it('a diff is taken against the read the page held, not the snapshot', async () => {
  localStorage.clear()
  seenAtR1()
  render(RequestPage, { requestId: 'r1' })
  await screen.findByText('Since you last looked (r1 to r3)')
  data.request = agentRename(0, 'Poster one')
  feed.move('studio', 'requests')
  await atRevision(4)
  await waitFor(() =>
    expect(diffOf(4)).toEqual(['Renamed First poster to Poster one']),
  )
  // the panel still says what it said: the visit's own snapshot is r3
  expect(sinceLines()).toEqual(['Changed Second poster: its arguments (seed)'])
})
