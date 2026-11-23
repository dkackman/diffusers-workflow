import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import { ApiError } from '../apiErrors'
import RequestMenu from './RequestMenu.svelte'
import type { LedgerRequest } from '../types'

const notify = vi.hoisted(() => ({
  error: vi.fn(),
  failure: vi.fn(),
  success: vi.fn(),
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
