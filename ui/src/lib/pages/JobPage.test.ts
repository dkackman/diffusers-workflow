import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import JobPage from './JobPage.svelte'
import type { JobDetail, JobEvent } from '../types'

const detail = vi.hoisted(() => ({ job: null as JobDetail | null }))
// The live stream's callback, captured so a test can push step_end events
const stream = vi.hoisted(() => ({
  onEvent: null as ((event: JobEvent) => void) | null,
}))

vi.mock('../api', () => ({
  ApiError: class ApiError extends Error {},
  api: {
    getJob: vi.fn(() => Promise.resolve(detail.job)),
    getJobWorkflow: vi.fn(() =>
      Promise.resolve({ definition: null, seed_variable: null }),
    ),
  },
  outputUrl: (path: string) => `/outputs/${path}`,
  streamJobEvents: (
    _jobId: string,
    _after: number,
    onEvent: (event: JobEvent) => void,
  ) => {
    stream.onEvent = onEvent
    return () => {}
  },
}))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

const job = (manifest: JobDetail['manifest']): JobDetail => ({
  id: 'j1',
  workflow: 'templates/minimax/storyboard',
  status: 'succeeded',
  created_at: 1,
  started_at: 1,
  finished_at: 2,
  workspace: 'default',
  arguments: {},
  warnings: [],
  manifest,
  error: null,
  traceback: null,
  event_count: 0,
})

afterEach(() => {
  cleanup()
  stream.onEvent = null
})

it('groups a foldered run under final/ and intermediate/ headings, final first', async () => {
  detail.job = job([
    { step: 'board_1', files: ['intermediate/b1.png'], subfolder: 'intermediate' },
    { step: 'voyage', files: ['final/voyage.mp4'], subfolder: 'final' },
  ])
  render(JobPage, { jobId: 'j1' })
  const headings = await waitFor(() => {
    const found = screen.getAllByRole('heading', { level: 3 })
    expect(found.length).toBeGreaterThanOrEqual(2)
    return found
  })
  expect(headings.map((h) => h.textContent?.trim())).toEqual(['final/', 'intermediate/'])
  // Step names drop to h4 under a subfolder heading
  expect(
    screen.getAllByRole('heading', { level: 4 }).map((h) => h.textContent?.trim()),
  ).toEqual(['voyage', 'board_1'])
})

it('renders an unfoldered run exactly as before: step headings, no subfolder heading', async () => {
  detail.job = job([
    { step: 'generate', files: ['a.png'] },
    { step: 'upscale', files: ['a_big.png'], subfolder: '' },
  ])
  render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(screen.getByText('upscale')).toBeTruthy())
  expect(
    screen.getAllByRole('heading', { level: 3 }).map((h) => h.textContent?.trim()),
  ).toEqual(['generate', 'upscale'])
  expect(screen.queryByRole('heading', { level: 4 })).toBeNull()
  expect(screen.queryByText('(run root)')).toBeNull()
})

it('places a live step_end under its subfolder before the manifest arrives', async () => {
  detail.job = { ...job([]), status: 'running', finished_at: null }
  render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(stream.onEvent).not.toBeNull())
  stream.onEvent!({
    seq: 1,
    event: 'step_end',
    step: 'episode',
    files: ['final/e.mp4'],
    subfolder: 'final',
  })
  await waitFor(() =>
    expect(screen.getByRole('heading', { level: 3, name: 'final/' })).toBeTruthy(),
  )
})
