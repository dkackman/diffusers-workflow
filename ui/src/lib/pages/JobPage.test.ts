import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import JobPage from './JobPage.svelte'
import '../router.svelte'
import { api, setNote } from '../api'
import { ApiError, NetworkError } from '../apiErrors'
import { connection } from '../connection.svelte'
import type { JobDetail, JobEvent } from '../types'
import { feed } from '../test/feedMock'

const detail = vi.hoisted(() => ({ job: null as JobDetail | null }))
// The live stream's callback, captured so a test can push step_end events
const stream = vi.hoisted(() => ({
  onEvent: null as ((event: JobEvent) => void) | null,
  onDrop: null as (() => void) | null,
  // the `after` each stream was opened from, and how many were stopped
  opened: [] as number[],
  stopped: 0,
}))
// What the metadata route answers for each file, by name
const metadata = vi.hoisted(() => ({
  byFile: {} as Record<string, Record<string, unknown>>,
  // The run inputs the route reports beside a file's metadata
  inputs: [] as Array<{ reference: string; kind: 'asset' | 'output' }>,
}))
// The definition the job ran, for the flow view and the unsaved reasons,
// and whether it is the realized copy or the definition as submitted
const ran = vi.hoisted(() => ({
  definition: null as Record<string, any> | null,
  realized: true,
}))
// What the event-log route answers: one page, or a status to fail with
const eventLog = vi.hoisted(() => ({
  events: [] as JobEvent[],
  note: null as string | null,
  status: 0,
}))

// Monaco cannot boot in jsdom; the page's JSON view is tested through this
// stub, which renders the value it was handed as plain text
vi.mock('../editor/JsonEditor.svelte', async () => ({
  default: (await import('./JsonEditorStub.svelte')).default,
}))

vi.mock('../api', () => ({
  ApiError: class ApiError extends Error {
    status: number
    constructor(message: string, status = 500) {
      super(message)
      this.name = 'ApiError'
      this.status = status
    }
  },
  TERMINAL_STATUSES: ['succeeded', 'failed', 'cancelled'],
  api: {
    getJob: vi.fn(() => Promise.resolve(detail.job)),
    getJobWorkflow: vi.fn(() =>
      Promise.resolve({
        definition: ran.definition,
        realized: ran.realized,
        seed_variable: null,
      }),
    ),
    galleryMetadata: vi.fn((name: string) =>
      Promise.resolve({
        name,
        metadata: metadata.byFile[name] ?? null,
        inputs: metadata.inputs,
        job: null,
      }),
    ),
    outputDownloadUrl: (name: string) => `/api/gallery/${name}/download`,
    noteForSave: vi.fn((target: string) =>
      Promise.resolve({ target, note: detail.job?.note ?? null, etag: '"e"' }),
    ),
    eventLog: vi.fn(async (id: string, query: { after?: number } = {}) => {
      if (eventLog.status) {
        const { ApiError } = await import('../api')
        throw new ApiError('Unknown job', eventLog.status)
      }
      const after = query.after ?? -1
      const page = eventLog.events.filter((e) => e.seq > after)
      return {
        id,
        status: detail.job?.status ?? null,
        events: page,
        last_seq: page.length ? page[page.length - 1].seq : after,
        truncated: false,
        note: eventLog.note,
      }
    }),
  },
  outputUrl: (path: string) => `/outputs/${path}`,
  setNote: vi.fn((target: string, note: string | null) =>
    Promise.resolve({ target, note, note_source: note ? 'run' : null }),
  ),
  streamJobEvents: (
    _jobId: string,
    after: number,
    onEvent: (event: JobEvent) => void,
    _onEnd: () => void,
    onDrop?: () => void,
  ) => {
    stream.onEvent = onEvent
    stream.onDrop = onDrop ?? null
    stream.opened.push(after)
    return () => {
      stream.stopped++
    }
  },
}))
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
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
  workflow_name: null,
  run_id: null,
  run_version: null,
  run_dir: null,
  acknowledged: 'none',
  acknowledged_cost: null,
})

beforeEach(() => {
  // The one router instance the module imported parsed the hash at load;
  // reset it here so a case that names a workspace (the URL-correction
  // test) does not leak its hash into whatever renders next
  location.hash = '#/ws/default/jobs/j1'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
})

afterEach(() => {
  cleanup()
  stream.onEvent = null
  stream.onDrop = null
  stream.opened = []
  stream.stopped = 0
  vi.mocked(api.getJob).mockClear()
  metadata.byFile = {}
  metadata.inputs = []
  ran.definition = null
  ran.realized = true
  eventLog.events = []
  eventLog.note = null
  eventLog.status = 0
  vi.mocked(api.galleryMetadata).mockClear()
  vi.mocked(api.eventLog).mockClear()
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
  detail.job = {
    ...job([{ step: 'generate', files: ['a.png', 'clip.mp4'] }]),
    output_kinds: { 'a.png': 'image', 'clip.mp4': 'video' },
  }
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

it('sets the tab title to the workflow file the job ran', async () => {
  detail.job = {
    ...job([]),
    workflow: 'Storyboard',
    workflow_name: 'templates/minimax/storyboard',
  }
  render(JobPage, { jobId: 'j1' })
  await waitFor(() =>
    expect(document.title).toBe('storyboard · diffusers-workflow'),
  )
  expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1)
})

it("drops one job's image metadata when the page switches to another", async () => {
  // Under the flat output layout two runs write the same file name, so a
  // lookup kept across the switch would show job A's recipe under job B
  metadata.byFile['a.png'] = { model_name: 'org/first' }
  detail.job = {
    ...job([{ step: 'generate', files: ['a.png'] }]),
    output_kinds: { 'a.png': 'image' },
  }
  const { rerender } = render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(screen.getByText('org/first')).toBeTruthy())

  metadata.byFile['a.png'] = { model_name: 'org/second' }
  detail.job = { ...detail.job, id: 'j2' }
  await rerender({ jobId: 'j2' })
  await waitFor(() => expect(screen.getByText('org/second')).toBeTruthy())
  expect(screen.queryByText('org/first')).toBeNull()
  expect(vi.mocked(api.galleryMetadata).mock.calls.map((c) => c[0])).toEqual([
    'a.png',
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

it("shows a running step's output inline, by the kind its step_end carries", async () => {
  detail.job = { ...job([]), status: 'running', finished_at: null }
  const { container } = render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(stream.onEvent).not.toBeNull())
  stream.onEvent!({
    seq: 1,
    event: 'step_end',
    step: 'generate',
    files: ['a.png'],
    output_kinds: { 'a.png': 'image' },
  })
  await waitFor(() =>
    expect(container.querySelector('img')?.getAttribute('src')).toContain(
      'a.png',
    ),
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

it('offers the workflow JSON behind a toggle, labelled as the realized copy', async () => {
  ran.definition = { steps: [{ name: 'base', pipeline: {} }] }
  render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(screen.getByText('realized')).toBeTruthy())
  // The JSON is for the curious, not the default view
  expect(screen.queryByTestId('json-editor')).toBeNull()
  await fireEvent.click(screen.getByRole('button', { name: 'show JSON' }))
  const shown = await screen.findByTestId('json-editor')
  // What the viewer is handed is the definition the page fetched
  expect(shown.textContent).toContain('"name": "base"')
  expect(screen.getByRole('button', { name: 'hide JSON' })).toBeTruthy()
})

it('labels a definition with no realized copy on file as submitted', async () => {
  ran.definition = { steps: [{ name: 'base', pipeline: {} }] }
  ran.realized = false
  render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(screen.getByText('as submitted')).toBeTruthy())
})

it("corrects the URL to the job's own workspace", async () => {
  location.hash = '#/ws/default/jobs/j1'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  detail.job = { ...job([]), workspace: 'studio' }
  render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(location.hash).toBe('#/ws/studio/jobs/j1'))
})

it('names the run by the version the gallery labels its files with', async () => {
  detail.job = {
    ...job([]),
    run_id: '20260922-120000-aaaaaaaa',
    run_version: 4,
  }
  render(JobPage, { jobId: 'j1' })
  const chip = await waitFor(() => screen.getByText('v4'))
  expect(chip.getAttribute('title')).toBe(
    '4th job of this workflow in this folder',
  )
})

it('shows no version for a job that never opened a run', async () => {
  detail.job = { ...job([]), run_version: null }
  render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(screen.getByText('j1')).toBeTruthy())
  expect(screen.queryByText(/^v\d+$/)).toBeNull()
})

it('renders each output by the kind the server reports', async () => {
  detail.job = {
    ...job([{ step: 'generate', files: ['a.bmp', 'b.mov', 'c.flac'] }]),
    output_kinds: { 'a.bmp': 'image', 'b.mov': 'video', 'c.flac': 'audio' },
  }
  const { container } = render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(container.querySelector('audio')).toBeTruthy())
  expect(container.querySelector('img')?.getAttribute('src')).toContain('a.bmp')
  expect(container.querySelector('video')?.getAttribute('src')).toContain(
    'b.mov',
  )
  expect(container.querySelector('audio')?.getAttribute('src')).toContain(
    'c.flac',
  )
})

it('shows the card the job ran on', async () => {
  detail.job = { ...job([]), device: 'cuda:1 NVIDIA GeForce RTX 3090' }
  render(JobPage, { jobId: 'j1' })
  await waitFor(() =>
    expect(
      screen.getByText('Card: cuda:1 NVIDIA GeForce RTX 3090'),
    ).toBeTruthy(),
  )
})

it('shows no card for a job without a device', async () => {
  detail.job = { ...job([]), device: null }
  render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(screen.getByText('j1')).toBeTruthy())
  expect(screen.queryByText(/Card:/)).toBeNull()
})

it('names what a rerun changed beside "rerun of", value per override in its title', async () => {
  detail.job = {
    ...job([]),
    rerun_of: 'orig1',
    overrides: { size: 'large', steps: 30 },
  }
  render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(screen.getByText('orig1')).toBeTruthy())
  const chip = screen.getByText('changed: size, steps')
  expect(chip.tagName).toBe('CODE')
  expect(chip.getAttribute('title')).toBe('size = large\nsteps = 30')
})

it('shows no changed chip on a rerun without overrides', async () => {
  detail.job = { ...job([]), rerun_of: 'orig1' }
  render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(screen.getByText('orig1')).toBeTruthy())
  expect(screen.queryByText(/changed:/)).toBeNull()
})

const roleJob = (): JobDetail => ({
  ...job([
    { step: 'draft', files: ['draft.png'], role: 'intermediate' },
    { step: 'film', files: ['film.mp4'], role: 'final' },
  ]),
  output_kinds: { 'draft.png': 'image', 'film.mp4': 'video' },
})

it('groups results Final, then a collapsed Intermediate, then Made from', async () => {
  metadata.inputs = [{ reference: 'asset:face.png', kind: 'asset' }]
  detail.job = roleJob()
  const { container } = render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(screen.getByText('face.png')).toBeTruthy())

  const order = [
    ...container.querySelectorAll('.roles > h3, .roles > details > summary'),
  ].map((el) => el.textContent?.trim())
  expect(order).toEqual(['Final', 'Intermediate', 'Made from'])
  expect(container.querySelector('.madefrom .role-tag')?.textContent).toBe(
    'Input',
  )
  // Final's file is on show; the intermediate's sits in a closed <details>
  const intermediate = container.querySelector(
    '.roles > details',
  ) as HTMLDetailsElement
  expect(intermediate.open).toBe(false)
  expect(intermediate.textContent).toContain('draft.png')
  expect(intermediate.textContent).not.toContain('film.mp4')
})

it('opens Intermediate when the run has no finals', async () => {
  detail.job = {
    ...job([{ step: 'draft', files: ['draft.png'], role: 'intermediate' }]),
    output_kinds: { 'draft.png': 'image' },
  }
  const { container } = render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(container.querySelector('.roles')).toBeTruthy())
  expect(
    (container.querySelector('.roles > details') as HTMLDetailsElement).open,
  ).toBe(true)
})

it('leaves the subfolder layout alone when no step has a role', async () => {
  detail.job = job([
    { step: 'film', files: ['final/f.mp4'], subfolder: 'final' },
  ])
  const { container } = render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(screen.getByText('final/')).toBeTruthy())
  expect(container.querySelector('.roles')).toBeNull()
})

it('reads Made from off the first output of a video-only run, probing once', async () => {
  metadata.inputs = [{ reference: 'asset:face.png', kind: 'asset' }]
  detail.job = {
    ...job([{ step: 'film', files: ['a.mp4', 'b.mp4'], role: 'final' }]),
    output_kinds: { 'a.mp4': 'video', 'b.mp4': 'video' },
  }
  const { container } = render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(screen.getByText('face.png')).toBeTruthy())
  expect(container.querySelector('.madefrom .role-tag')?.textContent).toBe(
    'Input',
  )
  expect(vi.mocked(api.galleryMetadata).mock.calls.map((c) => c[0])).toEqual([
    'a.mp4',
  ])
})

it('shows your note on the job when the job has a run', async () => {
  detail.job = { ...job([]), run_dir: 'wf/run-1', note: 'second pass' }
  render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(screen.getByText('second pass')).toBeTruthy())
  expect(screen.getByText('Your note on this job')).toBeTruthy()
})

it("saves the run note in the job's own workspace", async () => {
  detail.job = {
    ...job([]),
    workspace: 'studio',
    run_dir: 'wf/run-1',
    note: 'second pass',
  }
  render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(screen.getByText('second pass')).toBeTruthy())
  await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
  await fireEvent.input(screen.getByRole('textbox'), {
    target: { value: 'third pass' },
  })
  await fireEvent.click(screen.getByRole('button', { name: /save/i }))
  await waitFor(() =>
    expect(vi.mocked(setNote)).toHaveBeenCalledWith(
      'wf/run-1',
      'third pass',
      'studio',
      '"e"',
    ),
  )
})

describe('data changed underneath the page', () => {
  const notice = () => screen.queryByText('Updated by an agent or another tab')
  /** The feed moves `kind`, and the page's quiet re-reads land */
  async function moved(kind: 'jobs' | 'gallery') {
    feed.move('default', kind)
    for (let i = 0; i < 6; i++) await Promise.resolve()
  }
  const noted = (note: string) => {
    detail.job = { ...job([]), run_dir: 'wf/run-1', note }
  }

  it('a run note written elsewhere says so', async () => {
    noted('second pass')
    render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(screen.getByText('second pass')).toBeTruthy())
    noted('from the agent')
    await moved('gallery')
    await waitFor(() => expect(notice()).toBeTruthy())
    expect(screen.getByText('from the agent')).toBeTruthy()
  })

  it('a note being edited keeps its draft when a newer one lands', async () => {
    noted('second pass')
    render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(screen.getByText('second pass')).toBeTruthy())
    await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
    await fireEvent.input(screen.getByRole('textbox'), {
      target: { value: 'my third pass' },
    })
    noted('from the agent')
    await moved('gallery')
    // The box's own notice is the one: the page's line stays down (#1597)
    await waitFor(() =>
      expect(
        screen.getByText(
          /Your note on this job changed while you were editing/,
        ),
      ).toBeTruthy(),
    )
    expect(notice()).toBeNull()
    expect(screen.getAllByRole('status')).toHaveLength(1)
    const box = screen.getByRole('textbox') as HTMLTextAreaElement
    expect(box.value).toBe('my third pass')
  })

  it("the page's own note save says nothing", async () => {
    noted('second pass')
    render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(screen.getByText('second pass')).toBeTruthy())
    await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
    await fireEvent.input(screen.getByRole('textbox'), {
      target: { value: 'third pass' },
    })
    await fireEvent.click(screen.getByRole('button', { name: /save/i }))
    await waitFor(() => expect(screen.getByText('third pass')).toBeTruthy())
    // The save's own write moves the feed; the re-read holds that note
    noted('third pass')
    const reads = vi.mocked(api.getJob).mock.calls.length
    await moved('gallery')
    await waitFor(() =>
      expect(vi.mocked(api.getJob).mock.calls.length).toBeGreaterThan(reads),
    )
    expect(notice()).toBeNull()
  })

  it('a definition changed elsewhere says so', async () => {
    detail.job = job([])
    ran.definition = { steps: [{ name: 'base', pipeline: {} }] }
    render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(screen.getByText('realized')).toBeTruthy())
    ran.definition = { steps: [{ name: 'edited', pipeline: {} }] }
    await moved('jobs')
    await waitFor(() => expect(notice()).toBeTruthy())
  })

  it("the run's own realized copy replacing the submitted one says nothing", async () => {
    detail.job = job([])
    ran.definition = { steps: [{ name: 'base', pipeline: {} }] }
    ran.realized = false
    render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(screen.getByText('as submitted')).toBeTruthy())
    ran.definition = { steps: [{ name: 'base', pipeline: { seed: 7 } }] }
    ran.realized = true
    await moved('jobs')
    await waitFor(() => expect(screen.getByText('realized')).toBeTruthy())
    expect(notice()).toBeNull()
  })

  it('a move that leaves the note and definition alone says nothing', async () => {
    noted('second pass')
    ran.definition = { steps: [{ name: 'base', pipeline: {} }] }
    render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(screen.getByText('realized')).toBeTruthy())
    const reads = vi.mocked(api.getJobWorkflow).mock.calls.length
    await moved('jobs')
    await waitFor(() =>
      expect(vi.mocked(api.getJobWorkflow).mock.calls.length).toBeGreaterThan(
        reads,
      ),
    )
    await moved('gallery')
    expect(notice()).toBeNull()
  })
})

it('refetches the detail on queue_wait and on a live warning, not on a stall', async () => {
  detail.job = { ...job([]), status: 'running', finished_at: null }
  render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(stream.onEvent).not.toBeNull())
  const calls = () => vi.mocked(api.getJob).mock.calls.length
  const before = calls()
  stream.onEvent!({ seq: 1, event: 'queue_wait', reason: 'gpu_busy' })
  await waitFor(() => expect(calls()).toBe(before + 1))
  stream.onEvent!({
    seq: 2,
    event: 'warning',
    kind: 'phase_stall',
    phase: 'loading',
    seconds_since_last_progress: 90,
  })
  expect(calls()).toBe(before + 1)
  detail.job = { ...detail.job!, warnings: ['gen: audio clipped'] }
  stream.onEvent!({
    seq: 3,
    event: 'warning',
    kind: 'audio_clipped',
    message: 'x',
  })
  await waitFor(() =>
    expect(screen.getByText('gen: audio clipped')).toBeTruthy(),
  )
})

it('shows a stall line while a phase_stall is the latest event', async () => {
  detail.job = { ...job([]), status: 'running', finished_at: null }
  const { container } = render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(stream.onEvent).not.toBeNull())
  stream.onEvent!({ seq: 1, event: 'workflow_start', steps: ['gen'] })
  stream.onEvent!({ seq: 2, event: 'step_start', step: 'gen' })
  stream.onEvent!({
    seq: 3,
    event: 'warning',
    kind: 'phase_stall',
    phase: 'loading',
    seconds_since_last_progress: 240,
  })
  await waitFor(() =>
    expect(container.textContent).toContain('no event for 4m'),
  )
  stream.onEvent!({ seq: 4, event: 'phase', phase: 'generating' })
  await waitFor(() =>
    expect(container.textContent).not.toContain('no event for'),
  )
})

describe('step timing bar and peak host RAM (#1453)', () => {
  const timing = (extra: Record<string, unknown> = {}) => ({
    load_s: 4,
    encode_s: null,
    denoise_s: 10,
    decode_s: null,
    save_s: 1,
    first_pass_s: 3,
    pass_s: 0.9,
    passes: 12,
    load_cached: false,
    compiled: null,
    segments: 1,
    ...extra,
  })
  const timed = (a: object, b: object) => {
    detail.job = {
      ...job([
        { step: 'a', files: ['a.png'], ...a },
        { step: 'b', files: ['b.png'], ...b },
      ]),
      output_kinds: { 'a.png': 'image', 'b.png': 'image' },
    } as JobDetail
  }
  const bars = (c: HTMLElement) => [
    ...c.querySelectorAll<HTMLElement>('[data-testid="timing-bar"]'),
  ]

  it('draws a bar from manifest timing, skipping null parts', async () => {
    timed({ timing: timing() }, {})
    const { container } = render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(bars(container).length).toBe(1))
    const bar = bars(container)[0]
    expect(
      [...bar.querySelectorAll('[data-part]')].map((e) =>
        e.getAttribute('data-part'),
      ),
    ).toEqual(['load', 'denoise', 'save'])
    // inner first/pass figures are in the title only, never stacked
    expect(bar.getAttribute('title')).toContain('first pass 3.0s')
    expect(bar.getAttribute('title')).toContain('pass 0.9s')
    expect(bar.getAttribute('title')).toContain('12 passes')
    expect(bar.getAttribute('title')).not.toContain('segments')
    expect(bar.textContent).not.toContain('cached')
  })

  it('says cached only on true, and segments when more than one', async () => {
    timed(
      { timing: timing({ load_cached: true, segments: 3 }) },
      { timing: timing({ load_cached: null, load_s: null }) },
    )
    const { container } = render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(bars(container).length).toBe(2))
    const [first, second] = bars(container)
    expect(first.textContent).toContain('cached')
    expect(first.getAttribute('title')).toContain('3 segments')
    expect(second.textContent).not.toContain('cached')
  })

  it('draws no bar for a reused entry or one without timing', async () => {
    timed({ reused: true, timing: timing() }, {})
    const { container } = render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(screen.getByText('· reused')).toBeTruthy())
    expect(bars(container).length).toBe(0)
  })

  it('shows peak host RAM when set and nothing when null', async () => {
    detail.job = { ...job([]), host_memory_job_peak_committed_mb: 20480 }
    const first = render(JobPage, { jobId: 'j1' })
    await waitFor(() =>
      expect(screen.getByText('peak host RAM 20.0 GB')).toBeTruthy(),
    )
    first.unmount()
    detail.job = { ...job([]), host_memory_job_peak_committed_mb: null }
    render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(screen.getByText('j1')).toBeTruthy())
    expect(screen.queryByText(/peak host RAM/)).toBeNull()
  })
})

describe('event log (#1478)', () => {
  const logToggle = () => screen.getByRole('button', { name: /Event log/ })
  const logRows = (c: HTMLElement) =>
    [...c.querySelectorAll('.eventlog tbody tr')].map((tr) =>
      [...tr.querySelectorAll('td')].map((td) => td.textContent?.trim()),
    )

  it('is collapsed on a historical job and asks for nothing until opened', async () => {
    detail.job = { ...job([]), historical: true } as JobDetail
    eventLog.note =
      'Only the last 200 events of this job were retained; everything before seq 41 was dropped when the job was recorded.'
    eventLog.events = [
      { seq: 41, event: 'step_start', at: 1.5, step: 'gen' },
      { seq: 42, event: 'log', at: 2, message: 'loading model' },
    ]
    const { container } = render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(logToggle()).toBeTruthy())
    expect(logToggle().getAttribute('aria-expanded')).toBe('false')
    expect(vi.mocked(api.eventLog)).not.toHaveBeenCalled()

    await fireEvent.click(logToggle())
    await waitFor(() => expect(logRows(container).length).toBe(2))
    expect(
      screen.getByText(/everything before seq 41 was dropped/),
    ).toBeTruthy()
    expect(logRows(container)).toEqual([
      ['41', '+1.5s', 'step_start', '{"step":"gen"}'],
      ['42', '+2.0s', 'log', 'loading model'],
    ])
    expect(vi.mocked(api.eventLog).mock.calls).toEqual([
      ['j1', { after: -1, limit: 500 }],
    ])
    // A finished job's log does not grow, so nothing offers to refresh it
    expect(
      screen.queryByRole('button', { name: 'Refresh event log' }),
    ).toBeNull()
  })

  it('says there is no log for an unknown or pruned job', async () => {
    detail.job = { ...job([]), historical: true } as JobDetail
    eventLog.status = 404
    render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(logToggle()).toBeTruthy())
    await fireEvent.click(logToggle())
    await waitFor(() =>
      expect(screen.getByText('no event log for this job')).toBeTruthy(),
    )
  })

  it('filters the rows to one kind', async () => {
    detail.job = job([])
    eventLog.events = [
      { seq: 0, event: 'log', message: 'one' },
      { seq: 1, event: 'warning', kind: 'audio_clipped', message: 'clip' },
      { seq: 2, event: 'log', message: 'two' },
    ]
    const { container } = render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(logToggle()).toBeTruthy())
    await fireEvent.click(logToggle())
    await waitFor(() => expect(logRows(container).length).toBe(3))
    const select = screen.getByRole('combobox', { name: 'filter by kind' })
    await fireEvent.change(select, { target: { value: 'log' } })
    expect(logRows(container).map((r) => r[0])).toEqual(['0', '2'])
    expect(container.querySelector('.eventlog .count')?.textContent).toMatch(
      /^2 of \d+ events, seq 0–\d+$/,
    )
    await fireEvent.change(select, {
      target: { value: 'warning · audio_clipped' },
    })
    expect(logRows(container).map((r) => r[3])).toEqual(['clip'])
  })

  it("leaves a live job's flow view alone, and grows on Refresh", async () => {
    ran.definition = {
      steps: [
        {
          name: 'gen',
          pipeline: { configuration: { component_type: 'Fake' } },
        },
        { name: 'mux', task: { command: 'mux' } },
      ],
    }
    detail.job = { ...job([]), status: 'running', finished_at: null }
    const { container } = render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(stream.onEvent).not.toBeNull())
    stream.onEvent!({ seq: 0, event: 'workflow_start', steps: ['gen', 'mux'] })
    stream.onEvent!({ seq: 1, event: 'step_start', step: 'gen' })
    await waitFor(() =>
      expect(nodeFor(container, 'gen').classList.contains('active')).toBe(true),
    )

    // The log names a later step than the stream has reached; none of it
    // may reach the page's own events
    eventLog.events = [
      { seq: 0, event: 'workflow_start', steps: ['gen', 'mux'] },
      { seq: 1, event: 'step_start', step: 'gen' },
      { seq: 2, event: 'step_end', step: 'gen', files: [] },
      { seq: 3, event: 'step_start', step: 'mux' },
    ]
    await fireEvent.click(logToggle())
    await waitFor(() => expect(logRows(container).length).toBe(4))
    expect(nodeFor(container, 'gen').classList.contains('active')).toBe(true)
    expect(screen.getByText('4 events, seq 0–3')).toBeTruthy()
    expect(nodeFor(container, 'gen').classList.contains('done')).toBe(false)
    expect(nodeFor(container, 'mux').classList.contains('active')).toBe(false)

    eventLog.events.push({ seq: 4, event: 'log', message: 'muxing' })
    await fireEvent.click(
      screen.getByRole('button', { name: 'Refresh event log' }),
    )
    await waitFor(() => expect(logRows(container).length).toBe(5))
    expect(screen.getByText('5 events, seq 0–4')).toBeTruthy()
    expect(vi.mocked(api.eventLog).mock.calls.at(-1)).toEqual([
      'j1',
      { after: 3, limit: 500 },
    ])
    expect(nodeFor(container, 'mux').classList.contains('active')).toBe(false)
  })
})

it("clears a previous job id's error when the id changes to a valid job", async () => {
  const good = job([])
  detail.job = good
  vi.mocked(api.getJob).mockRejectedValueOnce(new ApiError('Unknown job', 404))
  const { rerender } = render(JobPage, { jobId: 'dead' })
  await screen.findByText("Couldn't load job dead: job dead wasn't found")
  await rerender({ jobId: 'j1' })
  await waitFor(() => expect(screen.getByText('Event log')).toBeTruthy())
  expect(screen.queryByText(/Couldn't load job/)).toBeNull()
})

describe('after an outage', () => {
  const running = (): JobDetail => ({
    ...job([]),
    status: 'running',
    finished_at: null,
  })

  it('a dropped stream re-reads the job and stops on a terminal one', async () => {
    detail.job = running()
    render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(stream.onDrop).not.toBeNull())
    const before = vi.mocked(api.getJob).mock.calls.length
    // Still running: the stream is EventSource's to repair, so it stays
    stream.onDrop!()
    await waitFor(() =>
      expect(vi.mocked(api.getJob).mock.calls.length).toBe(before + 1),
    )
    expect(stream.stopped).toBe(0)
    detail.job = { ...running(), status: 'succeeded', finished_at: 2 }
    stream.onDrop!()
    await waitFor(() => expect(stream.stopped).toBe(1))
    // The header shows the record the reload answered
    await waitFor(() =>
      expect(
        document.querySelector(
          '[title="start to finish - excludes time queued"]',
        ),
      ).not.toBeNull(),
    )
  })

  it('a job gone after a restart says so', async () => {
    detail.job = running()
    render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(stream.onDrop).not.toBeNull())
    await screen.findByText('Event log')
    vi.mocked(api.getJob).mockRejectedValueOnce(
      new ApiError('Unknown job', 404),
    )
    stream.onDrop!()
    await screen.findByText("Couldn't load job j1: job j1 wasn't found")
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(screen.queryByText('Event log')).toBeNull()
    expect(stream.stopped).toBe(1)
  })

  it('the page recovers when the server returns', async () => {
    detail.job = running()
    vi.mocked(api.getJob).mockRejectedValueOnce(
      new NetworkError('fetch failed'),
    )
    render(JobPage, { jobId: 'j1' })
    await screen.findByText("Couldn't load job j1: Can't reach the server")
    expect(stream.opened).toEqual([])
    connection.epoch++
    await screen.findByText('Event log')
    expect(screen.queryByText(/Couldn't load job/)).toBeNull()
    await waitFor(() => expect(stream.opened).toEqual([-1]))
  })

  it('an epoch rise reopens the stream of a running job where it left off', async () => {
    detail.job = running()
    render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(stream.onEvent).not.toBeNull())
    stream.onEvent!({ seq: 0, event: 'workflow_start', steps: ['gen'] })
    stream.onEvent!({ seq: 1, event: 'step_start', step: 'gen' })
    connection.epoch++
    await waitFor(() => expect(stream.opened).toEqual([-1, 1]))
    expect(stream.stopped).toBe(1)
  })

  it('a drop while offline reads nothing until the server is back', async () => {
    detail.job = running()
    render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(stream.onDrop).not.toBeNull())
    await screen.findByText('Event log')
    const before = vi.mocked(api.getJob).mock.calls.length
    connection.state = 'offline'
    try {
      // EventSource retries every few seconds while the server is down
      stream.onDrop!()
      stream.onDrop!()
      await new Promise((r) => setTimeout(r, 20))
      expect(vi.mocked(api.getJob).mock.calls.length).toBe(before)
      expect(screen.queryByText(/Couldn't refresh job/)).toBeNull()
      connection.state = 'ok'
      connection.epoch++
      await waitFor(() => expect(stream.opened).toEqual([-1, -1]))
      expect(vi.mocked(api.getJob).mock.calls.length).toBe(before + 1)
    } finally {
      connection.state = 'unknown'
    }
  })

  it('a job found gone says nothing about the next job it moves to', async () => {
    detail.job = running()
    const { rerender } = render(JobPage, { jobId: 'j1' })
    await screen.findByText('Event log')
    vi.mocked(api.getJob).mockRejectedValueOnce(
      new ApiError('Unknown job', 404),
    )
    connection.epoch++
    await screen.findByText("Couldn't load job j1: job j1 wasn't found")
    let answer: (job: JobDetail) => void = () => {}
    vi.mocked(api.getJob).mockImplementationOnce(
      () => new Promise<JobDetail>((r) => (answer = r)),
    )
    await rerender({ jobId: 'j2' })
    await waitFor(() =>
      expect(screen.queryByText(/Couldn't load job j1/)).toBeNull(),
    )
    expect(screen.queryByText(/Couldn't load job j2/)).toBeNull()
    answer({ ...running(), id: 'j2' })
    await screen.findByText('Event log')
  })

  it('a Retry on a gone job never brings back its last record', async () => {
    detail.job = running()
    render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(stream.onDrop).not.toBeNull())
    await screen.findByText('Event log')
    vi.mocked(api.getJob).mockRejectedValueOnce(
      new ApiError('Unknown job', 404),
    )
    stream.onDrop!()
    await screen.findByText("Couldn't load job j1: job j1 wasn't found")
    let answer: (job: JobDetail) => void = () => {}
    vi.mocked(api.getJob).mockImplementationOnce(
      () => new Promise<JobDetail>((r) => (answer = r)),
    )
    await fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await new Promise((r) => setTimeout(r, 20))
    expect(screen.queryByText('Event log')).toBeNull()
    answer(running())
    await screen.findByText('Event log')
  })

  it('an epoch rise leaves a finished job alone', async () => {
    detail.job = job([])
    render(JobPage, { jobId: 'j1' })
    await waitFor(() => expect(stream.opened).toEqual([-1]))
    connection.epoch++
    await waitFor(() =>
      expect(vi.mocked(api.getJob).mock.calls.length).toBeGreaterThan(1),
    )
    expect(stream.opened).toEqual([-1])
  })
})
