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
// The definition the job ran, for the flow view and the unsaved reasons
const ran = vi.hoisted(() => ({
  definition: null as Record<string, any> | null,
}))

vi.mock('../api', () => ({
  ApiError: class ApiError extends Error {},
  api: {
    getJob: vi.fn(() => Promise.resolve(detail.job)),
    getJobWorkflow: vi.fn(() =>
      Promise.resolve({ definition: ran.definition, seed_variable: null }),
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
  ran.definition = null
  vi.mocked(api.galleryMetadata).mockClear()
})

/** The flow view's box for one step of the definition. */
function nodeFor(container: HTMLElement, name: string) {
  return [...container.querySelectorAll('g.node')].find((node) =>
    node.getAttribute('aria-label')?.startsWith(`step ${name},`),
  )!
}

/** The page's "wrote nothing" rows, whitespace flattened. */
function unsavedRows(container: HTMLElement) {
  return [...container.querySelectorAll('.unsaved li')].map((li) =>
    li.textContent?.replace(/\s+/g, ' ').trim(),
  )
}

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

it('lights the for_each step in the flow chart while one of its members runs', async () => {
  // The graph is drawn from the definition, where for_each is one step; the
  // engine reports the members, so the two only meet at the group name
  ran.definition = {
    steps: [
      {
        name: 'shot',
        pipeline: { configuration: { component_type: 'Fake' } },
        for_each: 'variable:shots',
      },
      { name: 'episode', task: { command: 'mux' } },
    ],
  }
  detail.job = { ...job([]), status: 'running', finished_at: null }
  const { container } = render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(stream.onEvent).not.toBeNull())
  stream.onEvent!({
    seq: 1,
    event: 'workflow_start',
    steps: ['shot@open', 'shot@reveal', 'episode'],
  })
  stream.onEvent!({ seq: 2, event: 'step_start', step: 'shot@open' })
  await waitFor(() =>
    expect(nodeFor(container, 'shot').classList.contains('active')).toBe(true),
  )

  // One member of two down: the group is not behind us yet
  stream.onEvent!({ seq: 3, event: 'step_end', step: 'shot@open', files: [] })
  await waitFor(() =>
    expect(nodeFor(container, 'shot').classList.contains('active')).toBe(true),
  )
  expect(nodeFor(container, 'shot').classList.contains('done')).toBe(false)

  stream.onEvent!({ seq: 4, event: 'step_end', step: 'shot@reveal', files: [] })
  stream.onEvent!({ seq: 5, event: 'step_start', step: 'episode' })
  await waitFor(() =>
    expect(nodeFor(container, 'shot').classList.contains('done')).toBe(true),
  )
  expect(nodeFor(container, 'episode').classList.contains('active')).toBe(true)
})

it('marks the entries a for_each step runs as chips inside its box', async () => {
  // The realized workflow keeps for_each and holds the run's actual list
  // in its variables, so the flow view can name the members
  ran.definition = {
    variables: { shots: [{ name: 'open' }, { name: 'reveal' }] },
    steps: [
      {
        name: 'shot',
        for_each: 'variable:shots',
        pipeline: { configuration: { component_type: 'Fake' } },
      },
    ],
  }
  detail.job = { ...job([]), status: 'running', finished_at: null }
  const { container } = render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(stream.onEvent).not.toBeNull())
  stream.onEvent!({
    seq: 1,
    event: 'workflow_start',
    steps: ['shot@open', 'shot@reveal'],
  })
  stream.onEvent!({ seq: 2, event: 'step_start', step: 'shot@open' })
  const chip = (key: string) =>
    [...nodeFor(container, 'shot').querySelectorAll('g.member')].find(
      // the label is "member shot@open" plus ", done"/", active" when styled
      (m) =>
        m.getAttribute('aria-label')?.split(',')[0] === `member shot@${key}`,
    )!
  await waitFor(() =>
    expect(chip('open').classList.contains('active')).toBe(true),
  )

  // One entry down, one to go: the finished chip greens while the group
  // box itself stays amber
  stream.onEvent!({ seq: 3, event: 'step_end', step: 'shot@open', files: [] })
  await waitFor(() =>
    expect(chip('open').classList.contains('done')).toBe(true),
  )
  expect(chip('reveal').classList.contains('done')).toBe(false)
  expect(nodeFor(container, 'shot').classList.contains('active')).toBe(true)
})

it('keeps the Progress list on the composed step while its child runs', async () => {
  ran.definition = {
    steps: [
      { name: 'shot1', workflow: { path: 'child.json' } },
      { name: 'episode', task: { command: 'mux' } },
    ],
  }
  detail.job = { ...job([]), status: 'running', finished_at: null }
  const { container } = render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(stream.onEvent).not.toBeNull())
  stream.onEvent!({
    seq: 1,
    event: 'workflow_start',
    steps: ['shot1', 'episode'],
  })
  // What a child emits: its own step name, with the parent's alongside
  stream.onEvent!({
    seq: 2,
    event: 'step_start',
    step: 'reference_to_video_audio',
    parent_step: 'shot1',
  })
  await waitFor(() =>
    expect(nodeFor(container, 'shot1').classList.contains('active')).toBe(true),
  )
  const dotFor = (name: string) =>
    [...container.querySelectorAll('.step')]
      .find(
        (row) =>
          row.querySelector('span:nth-child(2)')?.textContent?.trim() === name,
      )!
      .querySelector('.dot')!
  expect(dotFor('shot1').classList.contains('active')).toBe(true)
  expect(dotFor('episode').classList.contains('active')).toBe(false)
})

it('says why a step wrote nothing, so a deliberate non-output is not a missing one', async () => {
  ran.definition = {
    steps: [
      { name: 'base', pipeline: {}, result: { save: false } },
      { name: 'edit', task: { command: 'mux' } },
      { name: 'film', pipeline: {}, result: { content_type: 'video/mp4' } },
    ],
  }
  detail.job = job([
    { step: 'base', files: [] },
    { step: 'edit', files: [] },
    { step: 'film', files: ['final/film.mp4'], subfolder: 'final' },
  ])
  const { container } = render(JobPage, { jobId: 'j1' })
  await waitFor(() =>
    expect(screen.getByText('Steps that wrote nothing')).toBeTruthy(),
  )
  expect(unsavedRows(container)).toEqual([
    'base result.save is false, so the step is kept in memory and never written',
    'edit result.content_type is not declared, so there is no file type to write',
  ])
  // The step that did write is in the results above, not in this list
  expect(screen.getByRole('heading', { level: 3, name: 'final/' })).toBeTruthy()
})

it('explains a run that wrote nothing at all rather than showing no results', async () => {
  ran.definition = { steps: [{ name: 'base', result: { save: false } }] }
  detail.job = job([{ step: 'base', files: [] }])
  const { container } = render(JobPage, { jobId: 'j1' })
  await waitFor(() =>
    expect(screen.getByText('Steps that wrote nothing')).toBeTruthy(),
  )
  expect(
    screen.getByRole('heading', { level: 2, name: 'Results' }),
  ).toBeTruthy()
  expect(unsavedRows(container)).toEqual([
    'base result.save is false, so the step is kept in memory and never written',
  ])
})

it('says when the run was queued with an acknowledgement bound to its plan', async () => {
  detail.job = {
    ...job([]),
    acknowledged: 'bound',
    acknowledged_cost: {
      fingerprint: 'sha256:abcdef0123456789',
      minutes: 38,
      downloads: [],
    },
  }
  render(JobPage, { jobId: 'j1' })
  const line = await waitFor(() => screen.getByText(/acknowledged/))
  expect(line.textContent?.replace(/\s+/g, ' ').trim()).toBe(
    'acknowledged at 38 min',
  )
})

it('says nothing about an acknowledgement a run did not carry', async () => {
  detail.job = { ...job([]), acknowledged: 'none' }
  render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(screen.getByText('j1')).toBeTruthy())
  expect(screen.queryByText(/acknowledged/)).toBeNull()
})
