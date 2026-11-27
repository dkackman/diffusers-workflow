import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import ActiveDownloads from './ActiveDownloads.svelte'
import { api } from './test/apiMock'
import { feed } from './test/feedMock'
import { workspace } from './workspaceState.svelte'

const state = vi.hoisted(() => ({ downloads: [] as unknown[], reads: 0 }))
vi.mock('./api', async () => (await import('./test/apiMock')).apiModule())
vi.mock('./changes.svelte', async () => ({
  changes: (await import('./test/feedMock')).feed.changes,
}))
api.listDownloads.mockImplementation(() => {
  state.reads++
  return Promise.resolve({ downloads: state.downloads })
})
api.cancelDownload.mockImplementation(() => Promise.resolve({}))

const download = (over: Record<string, unknown> = {}) => ({
  id: 'd1',
  repo_id: 'org/big-model',
  status: 'downloading',
  downloaded: 1024 ** 3,
  total: 4 * 1024 ** 3,
  error: null,
  finished_at: null,
  ...over,
})

afterEach(() => {
  cleanup()
  state.downloads = []
  state.reads = 0
  api.cancelDownload.mockClear()
})

it('shows an in-flight download with its progress', async () => {
  state.downloads = [download()]
  const { container } = render(ActiveDownloads)
  await screen.findByText('org/big-model')
  expect(container.textContent).toContain('1.0 / 4.0 GB')
  expect(
    container.querySelector('.dlbar > span')?.getAttribute('style'),
  ).toContain('width: 25%')
})

it('renders nothing when no download is active', async () => {
  const { container } = render(ActiveDownloads)
  await waitFor(() => expect(state.reads).toBe(1))
  expect(container.querySelector('.dl')).toBeNull()
  expect(container.textContent?.trim()).toBe('')
})

it('Cancel sends the download id', async () => {
  state.downloads = [download()]
  render(ActiveDownloads)
  await fireEvent.click(
    await screen.findByRole('button', {
      name: 'cancel downloading org/big-model',
    }),
  )
  await waitFor(() => expect(api.cancelDownload).toHaveBeenCalledWith('d1'))
})

it('a recently failed download names its error', async () => {
  state.downloads = [
    download({
      status: 'failed',
      error: 'disk full',
      finished_at: Date.now() / 1000 - 5,
    }),
  ]
  render(ActiveDownloads)
  await screen.findByText('disk full')
})

it('a models move reads the downloads again', async () => {
  render(ActiveDownloads)
  await waitFor(() => expect(state.reads).toBe(1))
  feed.move(workspace.current, 'models')
  await waitFor(() => expect(state.reads).toBe(2))
})

it('onfinished fires when an active download completes', async () => {
  const onfinished = vi.fn()
  state.downloads = [download()]
  render(ActiveDownloads, { onfinished })
  await screen.findByText('org/big-model')
  expect(onfinished).not.toHaveBeenCalled()
  state.downloads = [
    download({ status: 'done', finished_at: Date.now() / 1000 }),
  ]
  feed.move(workspace.current, 'models')
  await waitFor(() => expect(onfinished).toHaveBeenCalledTimes(1))
})
