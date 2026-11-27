import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import ModelsPage from './ModelsPage.svelte'
import { api } from '../test/apiMock'

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

const repo = (repo_id: string) => ({
  repo_id,
  repo_type: 'model',
  size_on_disk: 1024,
  nb_files: 3,
  last_accessed: null,
  last_job_use: null,
  last_modified: null,
  revisions: [],
})
api.listModels.mockResolvedValue({
  cache_dir: '/hub',
  disk_free: null,
  disk_total: null,
  size_on_disk: 0,
  warnings: [],
  repos: [repo('org/a'), repo('org/b')],
})
api.listDownloads.mockResolvedValue({ downloads: [] })
api.getPreferences.mockResolvedValue({ preferences: {}, resolved: {} })
api.listWorkflows.mockResolvedValue({ workflows: [] })

afterEach(cleanup)

it('Preferences tab is current, links both tabs and shows the preferences', async () => {
  render(ModelsPage, { tab: 'preferences' })
  const prefs = screen.getByRole('link', { name: 'Preferences' })
  expect(prefs.getAttribute('aria-current')).toBe('page')
  expect(prefs.getAttribute('href')).toBe('#/server/models')
  const downloaded = await screen.findByRole('link', {
    name: 'Downloaded · 2',
  })
  expect(downloaded.getAttribute('href')).toBe('#/server/models/downloaded')
  expect(downloaded.getAttribute('aria-current')).toBeNull()
  await waitFor(() =>
    expect(api.getPreferences).toHaveBeenCalledWith({ resolve: true }),
  )
  expect(screen.queryByLabelText('Hugging Face repo to download')).toBeNull()
})

it('Downloaded tab is current and shows the download form', async () => {
  render(ModelsPage, { tab: 'downloaded' })
  expect(screen.getByLabelText('Hugging Face repo to download')).toBeTruthy()
  const downloaded = await screen.findByRole('link', {
    name: 'Downloaded · 2',
  })
  expect(downloaded.getAttribute('aria-current')).toBe('page')
  expect(
    screen.getByRole('link', { name: 'Preferences' }).getAttribute('href'),
  ).toBe('#/server/models')
})
