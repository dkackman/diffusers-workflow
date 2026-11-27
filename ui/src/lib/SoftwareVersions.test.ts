import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import SoftwareVersions from './SoftwareVersions.svelte'
import { api } from './test/apiMock'
import { feed } from './test/feedMock'
import { workspace } from './workspaceState.svelte'
import { confirmDialog } from './confirm.svelte'

const state = vi.hoisted(() => ({
  // how many times the diffusers status was read
  diffusersReads: 0,
  // the diffusers status the component reads; null stays pending
  diffusers: null as unknown,
  // the dw update state the component reads; null stays pending
  dw: null as unknown,
}))
vi.mock('./confirm.svelte', () => ({ confirmDialog: vi.fn() }))
vi.mock('./api', async () => (await import('./test/apiMock')).apiModule())
vi.mock('./changes.svelte', async () => ({
  changes: (await import('./test/feedMock')).feed.changes,
}))
api.diffusersStatus.mockImplementation(() => {
  state.diffusersReads++
  return state.diffusers
    ? Promise.resolve(state.diffusers)
    : new Promise(() => {})
})
api.updateDiffusers.mockImplementation(() => Promise.resolve({}))
api.dwStatus.mockImplementation(() =>
  state.dw ? Promise.resolve(state.dw) : new Promise(() => {}),
)
api.updateDw.mockImplementation(() => Promise.resolve({}))

afterEach(() => {
  cleanup()
  state.diffusersReads = 0
  state.diffusers = null
  state.dw = null
  vi.mocked(confirmDialog).mockReset()
  api.updateDiffusers.mockClear()
  api.updateDw.mockClear()
})

it('a models move reads the diffusers status again', async () => {
  render(SoftwareVersions)
  await waitFor(() => expect(state.diffusersReads).toBe(1))
  // An update started by another tab or an agent bumps models
  feed.move(workspace.current, 'models')
  await waitFor(() => expect(state.diffusersReads).toBe(2))
})

it('Update diffusers asks first and runs only on yes', async () => {
  state.diffusers = { version: '0.30', commit: null, status: 'idle' }
  render(SoftwareVersions)
  const button = await screen.findByRole('button', {
    name: /Update from GitHub/,
  })
  vi.mocked(confirmDialog).mockResolvedValueOnce(false)
  await fireEvent.click(button)
  await waitFor(() => expect(confirmDialog).toHaveBeenCalledTimes(1))
  expect(api.updateDiffusers).not.toHaveBeenCalled()
  vi.mocked(confirmDialog).mockResolvedValueOnce(true)
  await fireEvent.click(button)
  await waitFor(() => expect(api.updateDiffusers).toHaveBeenCalledTimes(1))
  expect(vi.mocked(confirmDialog).mock.calls[1]).toEqual([
    "This changes the server's Python environment and restarts the idle worker.",
    {
      title: 'Update diffusers from GitHub?',
      confirmLabel: 'Update diffusers',
      cancelLabel: undefined,
      danger: false,
    },
  ])
})

const dwState = (over: Record<string, unknown>) => ({
  version: '0.23.0',
  latest: '0.24.0',
  newer: true,
  managed: true,
  command: 'curl -LsSf https://example.test/bootstrap.sh | sh',
  status: 'idle',
  error: null,
  log: null,
  ...over,
})

it('an unmanaged server shows the install command and no update button', async () => {
  state.dw = dwState({ managed: false })
  render(SoftwareVersions)
  await screen.findByText(/bootstrap\.sh/)
  expect(screen.queryByRole('button', { name: /Update dw/ })).toBeNull()
  expect(screen.getByTitle('copy the update command')).toBeTruthy()
})

it('Update dw on a managed server with a newer release asks, then runs', async () => {
  state.dw = dwState({})
  render(SoftwareVersions)
  const button = await screen.findByRole('button', { name: /Update dw/ })
  vi.mocked(confirmDialog).mockResolvedValueOnce(true)
  await fireEvent.click(button)
  await waitFor(() => expect(api.updateDw).toHaveBeenCalledTimes(1))
})

it('a managed server already on the latest offers no update', async () => {
  state.dw = dwState({ latest: '0.23.0', newer: false })
  render(SoftwareVersions)
  await screen.findByText('up to date')
  expect(screen.queryByRole('button', { name: /Update dw/ })).toBeNull()
})

it('the server decides newer: 0.10.0 over 0.9.0 offers the update', async () => {
  state.dw = dwState({ version: '0.9.0', latest: '0.10.0', newer: true })
  render(SoftwareVersions)
  await screen.findByRole('button', { name: /Update dw/ })
  expect(screen.getByText('0.10.0 is available')).toBeTruthy()
})

it('newer false offers no update even when latest reads newer as a string', async () => {
  state.dw = dwState({ version: '1.0.0', latest: '1.0.0rc1', newer: false })
  render(SoftwareVersions)
  await screen.findByText('up to date')
  expect(screen.queryByRole('button', { name: /Update dw/ })).toBeNull()
})

it('a null newer offers no update and claims nothing', async () => {
  state.dw = dwState({ latest: 'not a version', newer: null })
  render(SoftwareVersions)
  await screen.findByText('0.23.0')
  expect(screen.queryByRole('button', { name: /Update dw/ })).toBeNull()
  expect(screen.queryByText(/is available/)).toBeNull()
  expect(screen.queryByText('up to date')).toBeNull()
})

it('an unmanaged server with a null newer names no available release', async () => {
  state.dw = dwState({ managed: false, latest: null, newer: null })
  render(SoftwareVersions)
  await screen.findByText(/bootstrap\.sh/)
  expect(screen.queryByText(/is available/)).toBeNull()
})

it('a running dw update shows progress and keeps polling', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  try {
    state.dw = dwState({ status: 'running' })
    render(SoftwareVersions)
    await screen.findByText(/the server restarts/)
    expect(screen.queryByRole('button', { name: /Update dw/ })).toBeNull()
    const reads = api.dwStatus.mock.calls.length
    await vi.advanceTimersByTimeAsync(2100)
    await waitFor(() =>
      expect(api.dwStatus.mock.calls.length).toBeGreaterThan(reads),
    )
  } finally {
    vi.useRealTimers()
  }
})
