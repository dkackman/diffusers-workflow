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
import { ApiError } from '../apiErrors'
import { fakeLedger } from '../request/fakeLedger'
import { entry, item, request } from '../request/fixtures'
import { api } from '../test/apiMock'
import { useQuietClock } from '../test/quiet'

// The agent's proposal to close a request, answered on its page: Close
// writes the proposed state, Keep open declines with an optional note
useQuietClock()

const world = vi.hoisted(() => ({
  ledger: null as unknown as import('../request/fakeLedger').FakeLedger,
  notify: {
    error: vi.fn(),
    failure: vi.fn(),
    success: vi.fn(),
    undo: vi.fn(),
    dismiss: vi.fn(),
    news: vi.fn(),
  },
}))
vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
api.getRequest.mockImplementation(() => world.ledger.get())
api.putRequest.mockImplementation((_ws, _id, body) => world.ledger.put(body))
api.notesChannel.mockImplementation(async () => ({
  listening: false,
  last_seen: null,
}))
vi.mock('../toast', () => ({ notify: world.notify }))
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))

beforeEach(() => {
  sessionStorage.clear()
  vi.clearAllMocks()
  workspace.current = 'studio'
  world.ledger = fakeLedger(
    request([item('a', { title: 'Shot 1' })], {
      history: [entry(3, 'agent', 'opened')],
    }),
  )
  world.ledger.propose('every shot has a keeper')
})
afterEach(() => {
  cleanup()
  document.body.innerHTML = ''
})

const start = async () => {
  render(RequestPage, { requestId: 'r1' })
  await screen.findByRole('heading', { name: 'Launch posters' })
}
const banner = () => screen.queryByTestId('close-proposal')
const declined = () => screen.queryByTestId('close-declined')
const last = () => world.ledger.puts.at(-1)!

it('shows a pending proposal with its reason, Close and Keep open', async () => {
  await start()
  const shown = within(banner()!)
  expect(banner()!.textContent).toContain(
    'The agent proposes to mark this done: every shot has a keeper',
  )
  expect(shown.getByRole('button', { name: 'Close' })).toBeTruthy()
  expect(shown.getByRole('button', { name: 'Keep open' })).toBeTruthy()
  expect(declined()).toBeNull()
})

it('says abandon for a proposal to abandon', async () => {
  world.ledger.propose('the user moved on', 'abandoned')
  await start()
  expect(banner()!.textContent).toContain(
    'The agent proposes to abandon this: the user moved on',
  )
})

it('shows nothing with no proposal', async () => {
  world.ledger.current = { ...world.ledger.current, close_proposal: undefined }
  await start()
  expect(banner()).toBeNull()
  expect(declined()).toBeNull()
})

it('Close writes the proposed state as the user, and the banner goes', async () => {
  await start()
  await fireEvent.click(
    within(banner()!).getByRole('button', { name: 'Close' }),
  )
  await waitFor(() => expect(world.ledger.current.state).toBe('done'))
  expect(last().state).toBe('done')
  expect(last().close_proposal).toBeUndefined()
  expect(world.ledger.current.close_proposal?.outcome).toBe('accepted')
  await waitFor(() => expect(banner()).toBeNull())
  expect(declined()).toBeNull()
  await waitFor(() => expect(world.notify.undo).toHaveBeenCalledTimes(1))
})

it('Close is sent as the user', async () => {
  const { api: real } = await vi.importActual<typeof import('../api')>('../api')
  const fetch = vi
    .spyOn(globalThis, 'fetch')
    .mockResolvedValue(new Response('{}', { status: 200 }))
  await real.putRequest('studio', 'r1', {
    revision: 4,
    title: 'T',
    goal: 'g',
    state: 'done',
    items: [],
    why: 'marked done',
  })
  const body = JSON.parse(String(fetch.mock.calls[0][1]?.body))
  expect(body).toMatchObject({ by: 'user', state: 'done' })
  fetch.mockRestore()
})

it('a stale Close says the plan changed, not nothing', async () => {
  await start()
  vi.spyOn(world.ledger, 'put').mockRejectedValue(
    new ApiError('the request is at revision 9', 409, 'stale_revision'),
  )
  await fireEvent.click(
    within(banner()!).getByRole('button', { name: 'Close' }),
  )
  await waitFor(() =>
    expect(world.notify.error).toHaveBeenCalledWith(
      'The plan changed since this page read it. Look again, then try that.',
    ),
  )
  expect(world.notify.failure).not.toHaveBeenCalled()
  expect(banner()).toBeTruthy()
})

it('Keep open with a note declines with it, and the page says so', async () => {
  await start()
  await fireEvent.click(
    within(banner()!).getByRole('button', { name: 'Keep open' }),
  )
  const field = within(banner()!).getByRole('textbox')
  await waitFor(() => expect(document.activeElement).toBe(field))
  await fireEvent.input(field, {
    target: { value: 'the second shot still flickers' },
  })
  await fireEvent.click(
    within(banner()!).getByRole('button', { name: 'Keep open' }),
  )
  await waitFor(() => expect(banner()).toBeNull())
  expect(last().close_proposal).toEqual({
    outcome: 'declined',
    note: 'the second shot still flickers',
  })
  expect(last().state ?? 'open').toBe('open')
  expect(world.ledger.current.state).toBe('open')
  expect(world.ledger.current.close_proposal?.outcome).toBe('declined')
  expect(declined()!.textContent?.trim()).toBe(
    'You kept this open: the second shot still flickers',
  )
})

it('Keep open with no note still declines', async () => {
  await start()
  await fireEvent.click(
    within(banner()!).getByRole('button', { name: 'Keep open' }),
  )
  await fireEvent.click(
    within(banner()!).getByRole('button', { name: 'Keep open' }),
  )
  await waitFor(() => expect(banner()).toBeNull())
  expect(last().close_proposal).toEqual({ outcome: 'declined' })
  expect(declined()!.textContent?.trim()).toBe('You kept this open.')
})

it('Cancel and Escape leave the proposal pending, unsent', async () => {
  await start()
  const open = () =>
    fireEvent.click(
      within(banner()!).getByRole('button', { name: 'Keep open' }),
    )
  await open()
  await fireEvent.click(
    within(banner()!).getByRole('button', { name: 'Cancel' }),
  )
  expect(within(banner()!).queryByRole('textbox')).toBeNull()
  await open()
  await fireEvent.keyDown(within(banner()!).getByRole('textbox'), {
    key: 'Escape',
  })
  expect(within(banner()!).queryByRole('textbox')).toBeNull()
  expect(world.ledger.puts).toHaveLength(0)
})

it('the kept-open line gives way to the next proposal', async () => {
  world.ledger.put({
    revision: world.ledger.current.revision,
    title: world.ledger.current.title,
    goal: world.ledger.current.goal,
    items: world.ledger.current.items,
    why: 'kept open',
    close_proposal: { outcome: 'declined', note: 'not yet' },
  })
  await waitFor(() =>
    expect(world.ledger.current.close_proposal?.outcome).toBe('declined'),
  )
  world.ledger.propose('both shots are clean now')
  await start()
  expect(declined()).toBeNull()
  expect(banner()!.textContent).toContain('both shots are clean now')
})
