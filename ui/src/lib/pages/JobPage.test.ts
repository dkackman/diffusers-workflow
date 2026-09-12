import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import JobPage from './JobPage.svelte'
import { api } from '../api'
import type { JobDetail, JobEvent } from '../types'

const detail = vi.hoisted(() => ({ job: null as JobDetail | null }))
// The live stream's callback, captured so a test can push step_end events
const stream = vi.hoisted(() => ({
  onEvent: null as ((event: JobEvent) => void) | null,
}))
// What the metadata route answers for each file, by name
const metadata = vi.hoisted(() => ({
  byFile: {} as Record<string, Record<string, unknown>>,
}))

vi.mock('../api', () => ({
  ApiError: class ApiError extends Error {},
  api: {
    getJob: vi.fn(() => Promise.resolve(detail.job)),
    getJobWorkflow: vi.fn(() =>
      Promise.resolve({ definition: null, seed_variable: null }),
    ),
    galleryMetadata: vi.fn((name: string) =>
      Promise.resolve({
        name,
        metadata: metadata.byFile[name] ?? null,
        job: null,
      }),
    ),
    outputDownloadUrl: (name: string) => `/api/gallery/${name}/download`,
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
  metadata.byFile = {}
  vi.mocked(api.galleryMetadata).mockClear()
})

it('shows each image beside what made it, with a download link, and never probes a video', async () => {
  metadata.byFile['a.png'] = {
    model_name: 'org/model',
    seed: 7,
    arguments: { prompt: 'a cat', negative_prompt: 'a dog' },
  }
  detail.job = job([{ step: 'generate', files: ['a.png', 'clip.mp4'] }])
  render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(screen.getByText('org/model')).toBeTruthy())
  expect(screen.getByText('7')).toBeTruthy()
  expect(screen.getByText('a cat')).toBeTruthy()
  expect(screen.getByText('a dog')).toBeTruthy()
  expect(
    screen
      .getAllByRole('link', { name: 'Download' })
      .map((a) => a.getAttribute('href')),
  ).toEqual(['/api/gallery/a.png/download', '/api/gallery/clip.mp4/download'])
  // The metadata route decodes a video whole to probe it, and the page
  // shows nothing a probe would report
  expect(vi.mocked(api.galleryMetadata).mock.calls.map((c) => c[0])).toEqual([
    'a.png',
  ])
})

it('groups a foldered run under final/ and intermediate/ headings, final first', async () => {
  detail.job = job([
    {
      step: 'board_1',
      files: ['intermediate/b1.png'],
      subfolder: 'intermediate',
    },
    { step: 'voyage', files: ['final/voyage.mp4'], subfolder: 'final' },
  ])
  render(JobPage, { jobId: 'j1' })
  const headings = await waitFor(() => {
    const found = screen.getAllByRole('heading', { level: 3 })
    expect(found.length).toBeGreaterThanOrEqual(2)
    return found
  })
  expect(headings.map((h) => h.textContent?.trim())).toEqual([
    'final/',
    'intermediate/',
  ])
  // Step names drop to h4 under a subfolder heading
  expect(
    screen
      .getAllByRole('heading', { level: 4 })
      .map((h) => h.textContent?.trim()),
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
    screen
      .getAllByRole('heading', { level: 3 })
      .map((h) => h.textContent?.trim()),
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
    expect(
      screen.getByRole('heading', { level: 3, name: 'final/' }),
    ).toBeTruthy(),
  )
})
