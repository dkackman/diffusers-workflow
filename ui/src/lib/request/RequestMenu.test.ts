import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import { ApiError } from '../apiErrors'
import RequestMenu from './RequestMenu.svelte'
import type { LedgerRequest } from '../types'
import { fakeLedger } from './fakeLedger'
import { item, request } from './fixtures'
import { PlanStore } from './planStore.svelte'

const notify = vi.hoisted(() => ({
  error: vi.fn(),
  failure: vi.fn(),
  success: vi.fn(),
  undo: vi.fn(),
}))
vi.mock('../toast', () => ({ notify }))
const api = vi.hoisted(() => ({ deleteRequest: vi.fn() }))
vi.mock('../api', async (orig) => ({
  ...(await orig<typeof import('../api')>()),
  api,
}))
vi.mock('../confirm.svelte', () => ({
  confirmDialog: vi.fn().mockResolvedValue(true),
}))

afterEach(() => {
  cleanup()
  document.body.innerHTML = ''
  vi.clearAllMocks()
})

it('a refused state change is a failure toast, not the raw message', async () => {
  const refused = new ApiError('the ledger is busy', 503)
  const push = vi.fn().mockRejectedValue(refused)
  render(RequestMenu, {
    props: {
      request: { id: 'r1', title: 'T', state: 'open' } as LedgerRequest,
      store: { push, read: null } as never,
      workspace: 'default',
      onreload: vi.fn(),
      ondeleted: vi.fn(),
    },
  })
  await fireEvent.click(
    screen.getByRole('button', { name: 'More actions for this request' }),
  )
  await fireEvent.click(
    await screen.findByRole('button', { name: 'Mark done' }),
  )
  await vi.waitFor(() => expect(notify.failure).toHaveBeenCalled())
  expect(notify.failure).toHaveBeenCalledWith('save that', refused)
  expect(notify.error).not.toHaveBeenCalled()
})

it('a delete refused for an unnamed reason is a failure toast, not the raw message', async () => {
  const refused = new ApiError('sqlite: database is locked', 503)
  api.deleteRequest.mockRejectedValue(refused)
  render(RequestMenu, {
    props: {
      request: { id: 'r1', title: 'T', state: 'done' } as LedgerRequest,
      store: { push: vi.fn(), read: { revision: 2 } } as never,
      workspace: 'default',
      onreload: vi.fn(),
      ondeleted: vi.fn(),
    },
  })
  await fireEvent.click(
    screen.getByRole('button', { name: 'More actions for this request' }),
  )
  await fireEvent.click(await screen.findByRole('button', { name: 'Delete…' }))
  await vi.waitFor(() => expect(notify.failure).toHaveBeenCalled())
  expect(notify.failure).toHaveBeenCalledWith('delete that', refused)
  expect(notify.error).not.toHaveBeenCalled()
})

/** The menu over a real store on a ledger that refuses a stale write */
function realStore(state: LedgerRequest['state'] = 'open') {
  const ledger = fakeLedger(request([item('a')], { state }))
  const store = new PlanStore({
    get: () => ledger.get(),
    put: (body) => ledger.put(body),
  })
  store.take(ledger.current)
  render(RequestMenu, {
    props: {
      request: ledger.current,
      store,
      workspace: 'default',
      onreload: vi.fn(),
      ondeleted: vi.fn(),
    },
  })
  return { ledger, store }
}

async function pick(name: string) {
  await fireEvent.click(
    screen.getByRole('button', { name: 'More actions for this request' }),
  )
  await fireEvent.click(await screen.findByRole('button', { name }))
}

it('Mark done acts at once and Undo reopens', async () => {
  const { ledger } = realStore()
  await pick('Mark done')
  // No question first: it is written, then offered back
  await vi.waitFor(() => expect(notify.undo).toHaveBeenCalledTimes(1))
  expect(ledger.current.state).toBe('done')
  expect(notify.undo.mock.calls[0][0]).toBe('Marked "Launch posters" done')
  notify.undo.mock.calls[0][1]()
  await vi.waitFor(() => expect(ledger.current.state).toBe('open'))
  expect(ledger.current.history.at(-1)?.why).toBe('reopened')
  expect(notify.failure).not.toHaveBeenCalled()
  expect(notify.error).not.toHaveBeenCalled()
})

it('Abandon offers Undo', async () => {
  const { ledger } = realStore()
  await pick('Abandon')
  await vi.waitFor(() => expect(notify.undo).toHaveBeenCalledTimes(1))
  expect(ledger.current.state).toBe('abandoned')
  expect(notify.undo.mock.calls[0][0]).toBe('Abandoned "Launch posters"')
  notify.undo.mock.calls[0][1]()
  await vi.waitFor(() => expect(ledger.current.state).toBe('open'))
})

it("an Undo refused as stale goes through the store's conflict path", async () => {
  const { ledger, store } = realStore()
  await pick('Mark done')
  await vi.waitFor(() => expect(notify.undo).toHaveBeenCalledTimes(1))
  // An agent reopens the request (and moves the plan on) before Undo
  ledger.current = {
    ...ledger.current,
    state: 'open',
    revision: ledger.current.revision + 1,
  }
  const puts = ledger.puts.length
  const gets = ledger.gets
  notify.undo.mock.calls[0][1]()
  // The Undo's write is refused as stale, the store reads the plan again
  // and applies the reopen to it: already open, so nothing is sent over it
  await vi.waitFor(() => expect(ledger.gets).toBe(gets + 1))
  await vi.waitFor(() => expect(store.queue).toHaveLength(0))
  expect(ledger.puts).toHaveLength(puts + 1)
  expect(ledger.current.state).toBe('open')
  expect(store.read?.revision).toBe(ledger.current.revision)
  expect(notify.failure).not.toHaveBeenCalled()
})

it('an Undo the store gives up on says the plan changed, not a raw error', async () => {
  const stale = new ApiError('at revision 9, not 4', 409, 'stale_revision')
  const push = vi
    .fn()
    .mockResolvedValueOnce({ status: 'saved', stripped: [] })
    .mockRejectedValueOnce(stale)
  render(RequestMenu, {
    props: {
      request: { id: 'r1', title: 'T', state: 'open' } as LedgerRequest,
      store: { push, read: null } as never,
      workspace: 'default',
      onreload: vi.fn(),
      ondeleted: vi.fn(),
    },
  })
  await pick('Abandon')
  await vi.waitFor(() => expect(notify.undo).toHaveBeenCalledTimes(1))
  notify.undo.mock.calls[0][1]()
  await vi.waitFor(() => expect(notify.error).toHaveBeenCalled())
  expect(push).toHaveBeenLastCalledWith({ kind: 'state', state: 'open' })
  expect(notify.error.mock.calls[0][0]).toMatch(/^The plan changed/)
  expect(notify.failure).not.toHaveBeenCalled()
})
