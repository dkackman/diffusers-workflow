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

const data = vi.hoisted(() => ({
  request: null as unknown,
  getError: null as { status: number; message: string } | null,
  putRequest: vi.fn(),
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
    galleryThumbnailUrl: (name: string, ws?: string) =>
      `/thumb/${name}?ws=${ws}`,
    putRequest: (...args: unknown[]) => data.putRequest(...args),
  },
}))
vi.mock('../toast', () => ({
  notify: { error: vi.fn(), success: vi.fn(), undo: vi.fn() },
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
        entry(3, 'user', 'comment on a'),
      ],
    },
  )

beforeEach(() => {
  sessionStorage.clear()
  workspace.current = 'studio'
  data.request = make()
  data.getError = null
  data.putRequest.mockReset()
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
      name: 'Leave a note for the agent',
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
    'Nothing is running. 1 ready to run. 1 of 2 done.',
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
  ).toEqual(['1.5 min', 'est. ~4 min'])
  expect(
    container
      .querySelector('.figures')
      ?.textContent?.replace(/\s+/g, ' ')
      .trim(),
  ).toBe('5 min spent ~7 min left')
  // the proof comes from the ledger's own output: no job is asked for
  expect(
    screen
      .getByRole('img', { name: 'result of First poster' })
      .getAttribute('src'),
  ).toBe('/thumb/run/a.png?ws=studio')
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
    screen.getByRole('link', { name: 'See all requests' }).getAttribute('href'),
  ).toBe('#/ws/studio/requests')
})

it('a first load that fails says why, and Retry asks again', async () => {
  data.getError = { status: 503, message: 'busy' }
  render(RequestPage, { requestId: 'r1' })
  expect((await screen.findByRole('alert')).textContent).toContain(
    'Could not load this request: busy',
  )
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

it('saves a note at the read revision, changing only that item', async () => {
  data.putRequest.mockImplementation((_ws, _id, body) =>
    Promise.resolve({ ...make(), revision: body.revision + 1 }),
  )
  render(RequestPage, { requestId: 'r1' })
  const box = await openNote('Second poster')
  await fireEvent.input(box, { target: { value: 'needs more red' } })
  await saveNote('Second poster')
  await waitFor(() => expect(data.putRequest).toHaveBeenCalled())
  const [ws, id, body] = data.putRequest.mock.calls[0]
  expect(ws).toBe('studio')
  expect(id).toBe('r1')
  expect(body).not.toHaveProperty('prompt')
  // the request's own note is left out, so the server keeps it
  expect(body).not.toHaveProperty('comment')
  expect(body).toMatchObject({
    title: 'Launch posters',
    goal: 'Three finished posters',
    revision: 3,
    why: 'comment on b',
    state: 'open',
  })
  const expected = make().items
  expect(body.items[0]).toEqual(expected[0])
  expect(body.items[1]).toEqual({
    ...expected[1],
    comment: { text: 'needs more red' },
  })
  expect(body.items[2]).toEqual(expected[2])
  await atRevision(4)
})

it("saves the request's own note, every item as it was read", async () => {
  data.putRequest.mockImplementation((_ws, _id, body) =>
    Promise.resolve({
      ...make(),
      revision: body.revision + 1,
      comment: { text: body.comment.text, at: '2026-01-04T00:00:00+00:00' },
    }),
  )
  const { container } = render(RequestPage, { requestId: 'r1' })
  await loaded()
  const head = container.querySelector('.reqhead') as HTMLElement
  await fireEvent.click(
    within(head).getByRole('button', { name: 'Leave a note for the agent' }),
  )
  await fireEvent.input(
    screen.getByLabelText('Note for the agent on this request'),
    { target: { value: 'make it all warmer' } },
  )
  await fireEvent.click(within(head).getByRole('button', { name: 'Save note' }))
  await waitFor(() => expect(data.putRequest).toHaveBeenCalled())
  const body = data.putRequest.mock.calls[0][2]
  expect(body).toMatchObject({
    revision: 3,
    why: 'comment on the request',
    comment: { text: 'make it all warmer' },
  })
  expect(body.items).toEqual(make().items)
  await within(head).findByText('make it all warmer')
  expect(within(head).getByText(/Nothing is sent to the agent/)).toBeTruthy()
})

it('a write refused as stale over and over says so in plain words and keeps the draft', async () => {
  const { ApiError } = await import('../api')
  data.putRequest.mockRejectedValue(
    new ApiError(
      'at revision 5, not 3 - read it again and rewrite the plan against revision 5',
      409,
      'stale_revision',
    ),
  )
  render(RequestPage, { requestId: 'r1' })
  const box = await openNote('Second poster')
  await fireEvent.input(box, { target: { value: 'my draft' } })
  await saveNote('Second poster')
  expect((await screen.findByRole('alert')).textContent).toContain(
    'The plan changed while you were writing',
  )
  expect(screen.getByRole('button', { name: 'Reload the plan' })).toBeTruthy()
  expect(box.value).toBe('my draft')
  // tried over a fresh read three more times, then left to the user
  expect(data.putRequest).toHaveBeenCalledTimes(4)
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
  // and the save goes at the refreshed revision, not the one first read
  data.putRequest.mockImplementation((_ws, _id, body) =>
    Promise.resolve({ ...next, revision: body.revision + 1 }),
  )
  await saveNote('Second poster')
  await waitFor(() => expect(data.putRequest).toHaveBeenCalled())
  expect(data.putRequest.mock.calls[0][2].revision).toBe(4)
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
  done.items[1] = doneItem('b', { title: 'Second poster', depends_on: ['a'] })
  data.request = done
  vi.advanceTimersByTime(ACTIVE_MS)
  await waitFor(() => expect(live()).toBe('Second poster finished.'))
})

it('does not refresh a closed request with nothing in flight', async () => {
  fakeInterval()
  const { api } = await import('../api')
  data.request = { ...make(), state: 'done' }
  render(RequestPage, { requestId: 'r1' })
  await loaded()
  const calls = vi.mocked(api.getRequest).mock.calls.length
  vi.advanceTimersByTime(IDLE_MS * 3)
  expect(vi.mocked(api.getRequest).mock.calls.length).toBe(calls)
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
  // The save lands first, at revision 4
  const saved = make()
  saved.revision = 4
  saved.items[1].comment = { text: 'more red', at: '2026-01-04T00:00:00Z' }
  data.putRequest.mockResolvedValue(saved)
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
  data.getError = { status: 503, message: 'busy' }
  vi.advanceTimersByTime(IDLE_MS)
  const line = await screen.findByRole('status')
  expect(line.textContent).toMatch(
    /Showing the plan as of .+\. Could not refresh: busy/,
  )
  expect(notify.error).not.toHaveBeenCalled()
  expect(box.value).toBe('my draft')
  expect(screen.getByRole('heading', { name: 'Launch posters' })).toBeTruthy()
  // Retry reads again; a read that works takes the line away
  data.getError = null
  await fireEvent.click(within(line).getByRole('button', { name: 'Retry' }))
  await waitFor(() => expect(screen.queryByRole('status')).toBeNull())
  expect(box.value).toBe('my draft')
})

it('a save over a plan the agent changed is made again on top, and says so', async () => {
  const { ApiError, api } = await import('../api')
  data.putRequest.mockRejectedValueOnce(
    new ApiError('now revision 4', 409, 'stale_revision'),
  )
  render(RequestPage, { requestId: 'r1' })
  const box = await openNote('Second poster')
  await fireEvent.input(box, { target: { value: 'more red' } })
  // the agent has written since the page read the plan
  const theirs = make()
  theirs.revision = 4
  theirs.history.push(entry(4, 'agent', 'added shot 7'))
  data.request = theirs
  const saved = structuredClone(theirs)
  saved.revision = 5
  saved.history.push(entry(5, 'user', 'comment on b'))
  saved.items[1].comment = { text: 'more red', at: '2026-01-04T00:00:00Z' }
  data.putRequest.mockResolvedValue(saved)
  const reads = vi.mocked(api.getRequest).mock.calls.length
  await saveNote('Second poster')
  await atRevision(5)
  // read again once, and the write made over that read: never the stale plan
  expect(vi.mocked(api.getRequest).mock.calls.length).toBe(reads + 1)
  expect(data.putRequest.mock.calls.map((c) => c[2].revision)).toEqual([3, 4])
  await within(row('Second poster')).findByText('more red')
  const notice = screen.getByRole('status')
  expect(notice.textContent).toContain(
    "The agent changed the plan while you were editing (r4: 'added shot 7'). Your change was applied on top.",
  )
  // the saved note's box has closed and handed focus back
  await waitFor(() =>
    expect(document.activeElement?.textContent).toBe('Edit note'),
  )
  await fireEvent.click(
    within(notice).getByRole('button', { name: 'See changes' }),
  )
  expect(document.activeElement?.id).toBe('plan-change-r4')
  await fireEvent.click(within(notice).getByRole('button', { name: 'Dismiss' }))
  expect(screen.queryByRole('status')).toBeNull()
})

it('a read that fails after a stale save keeps the request and the draft', async () => {
  const { ApiError } = await import('../api')
  const { notify } = await import('../toast')
  data.putRequest.mockRejectedValue(
    new ApiError('now revision 5', 409, 'stale_revision'),
  )
  render(RequestPage, { requestId: 'r1' })
  const box = await openNote('Second poster')
  await fireEvent.input(box, { target: { value: 'my draft' } })
  data.getError = { status: 503, message: 'busy' }
  await saveNote('Second poster')
  await waitFor(() => expect(notify.error).toHaveBeenCalledWith('busy'))
  expect(data.putRequest).toHaveBeenCalledTimes(1)
  expect(box.value).toBe('my draft')
  expect(document.body.contains(box)).toBe(true)
  expect(screen.getByRole('heading', { name: 'Launch posters' })).toBeTruthy()
})
