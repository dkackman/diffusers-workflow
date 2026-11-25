import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import JobResults from './JobResults.svelte'
import { api } from '../test/apiMock'
import type { JobDetail } from '../types'
import { workspace } from '../workspaceState.svelte'

const timeline = {
  kind: 'timeline',
  name: 'cut',
  otio_version: '0.17',
  has_audio_track: false,
  tracks: [
    {
      kind: 'Video',
      name: 'V',
      clip_count: 1,
      seconds: 2,
      items: [
        {
          type: 'Clip',
          name: 'opening',
          start_frame: 0,
          frames: 48,
          rate: 24,
          seconds: 2,
          media: 'media/a.mp4',
          source: 'output:run1/a.mp4',
        },
        {
          type: 'Clip',
          name: 'asset clip',
          start_frame: 0,
          frames: 48,
          rate: 24,
          seconds: 2,
          media: 'media/b.mp4',
          source: null,
        },
      ],
    },
  ],
  members: ['content.otio', 'media/a.mp4', 'media/b.mp4'],
}

// What the metadata route embeds for a file, by name (a timeline otherwise)
const embedded = vi.hoisted(() => ({}) as Record<string, unknown>)
// Whether the metadata route says a file is starred, by name
const starredOn = vi.hoisted(() => new Set<string>())
const notifyUndo = vi.hoisted(() => vi.fn())
// Reads that never land, while set
const held = vi.hoisted(() => ({ reads: false, fail: false }))
vi.mock('../api', async () =>
  (await import('../test/apiMock')).apiModule({
    outputUrl: (path: string) => `/outputs/${path}`,
  }),
)
api.galleryMetadata.mockImplementation((name: string) =>
  held.reads
    ? new Promise(() => {})
    : held.fail
      ? Promise.reject(new Error('read failed'))
      : Promise.resolve({
          name,
          metadata: embedded[name] ?? null,
          media: name.endsWith('.otioz') ? timeline : null,
          inputs: [],
          job: null,
          findings: [],
          linked_jobs: [],
          ...(starredOn.has(name) && { starred: true }),
        }),
)
api.getJob.mockImplementation(() => new Promise(() => {}))
api.archiveOutputs.mockImplementation(() => Promise.resolve())
api.setStarred.mockImplementation(() => Promise.resolve())
api.outputDownloadUrl.mockImplementation(
  (name: string) => `/api/gallery/${name}/download`,
)
vi.mock('../toast', () => ({
  notify: {
    error: vi.fn(),
    success: vi.fn(),
    failure: vi.fn(),
    undo: notifyUndo,
  },
}))

beforeEach(() => {
  workspace.current = 'w'
  api.galleryMetadata.mockClear()
  api.archiveOutputs.mockClear()
  api.setStarred.mockClear()
  notifyUndo.mockClear()
  history.replaceState(null, '', `${location.pathname}#/ws/w/jobs/j1`)
})
afterEach(() => {
  cleanup()
  for (const key of Object.keys(embedded)) delete embedded[key]
  starredOn.clear()
  held.reads = false
  held.fail = false
})

const hashEvent = (change: () => void) =>
  new Promise<void>((resolve) => {
    window.addEventListener('hashchange', () => resolve(), { once: true })
    change()
  })

/** Opens an output in the viewer by its tile, and waits for the sheet */
async function openOutput(name: string) {
  const leaf = name.split('/').pop()!
  await hashEvent(
    () => void fireEvent.click(screen.getByLabelText(`Open ${leaf}`)),
  )
  return screen.findByRole('dialog', { name: leaf })
}

const group = (
  step: string,
  files: string[],
  over: Partial<{ subfolder: string; role: 'final' | 'intermediate' }> = {},
) => ({
  step,
  files,
  reused: false,
  subfolder: '',
  role: null,
  ...over,
})

it('fetches an .otioz once and renders its timeline strip', async () => {
  const { container } = render(JobResults, {
    job: { id: 'j1', workspace: 'w' } as JobDetail,
    fileGroups: [
      {
        step: 'join',
        files: ['run1/cut.otioz', 'run1/a.mp4', 'run1/b.mp4'],
        reused: false,
        subfolder: '',
        role: null,
      },
    ],
    unsaved: [],
    kinds: {
      'run1/cut.otioz': null,
      'run1/a.mp4': 'video',
      'run1/b.mp4': 'video',
    },
    seedVariable: null,
  })
  await waitFor(() => expect(screen.getByText('opening · 2 s')).toBeTruthy())
  expect(container.querySelector('[role="group"]')).toBeTruthy()
  const calls = api.galleryMetadata.mock.calls.map((c) => c[0])
  // Linked by recorded source; b.mp4 shares a job file's name but has none
  expect(
    screen.getByRole('link', { name: 'Download media/a.mp4' }),
  ).toBeTruthy()
  expect(
    screen.queryByRole('link', { name: 'Download media/b.mp4' }),
  ).toBeNull()
  expect(calls).toEqual(['run1/cut.otioz'])
})

it('offers "Note this file" on each output only when the page gives onnote', async () => {
  const onnote = vi.fn()
  const groups = [
    {
      step: 'render',
      files: ['run1/a.mp4', 'run1/b.wav'],
      reused: false,
      subfolder: '',
      role: null,
    },
  ]
  const kinds = { 'run1/a.mp4': 'video', 'run1/b.wav': 'audio' }
  const job = { id: 'j1', workspace: 'w' } as JobDetail
  const args = {
    job,
    fileGroups: groups,
    unsaved: [],
    kinds,
    seedVariable: null,
  }
  render(JobResults, args)
  await openOutput('run1/b.wav')
  expect(screen.queryByRole('button', { name: /Note this file/ })).toBeNull()
  await hashEvent(
    () => void fireEvent.click(screen.getByRole('button', { name: 'Close' })),
  )
  cleanup()
  render(JobResults, { ...args, onnote })
  await openOutput('run1/b.wav')
  await fireEvent.click(
    screen.getByRole('button', {
      name: 'Note this file (b.wav) for the agent',
    }),
  )
  // the viewer closes, so the note box it opens is not under the sheet
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  await waitFor(() => expect(onnote).toHaveBeenCalledWith('run1/b.wav'))
})

const job = { id: 'j1', workspace: 'w' } as JobDetail

it('an output opens in the viewer with its recipe', async () => {
  embedded['run1/a.png'] = {
    model_name: 'org/model',
    seed: 7,
    arguments: { prompt: 'a cat' },
  }
  render(JobResults, {
    job,
    fileGroups: [group('generate', ['run1/a.png', 'run1/b.png'])],
    unsaved: [],
    kinds: { 'run1/a.png': 'image', 'run1/b.png': 'image' },
    seedVariable: null,
  })
  // one line of what made it under each tile: model and seed
  const tile = screen.getByLabelText('Open a.png')
  await waitFor(() => expect(tile).toHaveTextContent('org/model · seed 7'))
  expect(tile.querySelector('img')?.getAttribute('src')).toContain(
    'run1/a.png/thumbnail',
  )
  const sheet = await openOutput('run1/a.png')
  expect(sheet).toHaveTextContent('1 of 2')
  expect(
    within(sheet).getByRole('heading', { name: 'What made this' }),
  ).toBeTruthy()
  await waitFor(() => expect(sheet).toHaveTextContent('a cat'))
  expect(within(sheet).getByText('output:run1/a.png')).toBeTruthy()
  expect(
    within(sheet).getByRole('button', { name: /Save as asset/ }),
  ).toBeTruthy()
  expect(
    within(sheet).getByRole('button', { name: /Use as input/ }),
  ).toBeTruthy()
  expect(
    within(sheet).getByRole('button', { name: 'Star', pressed: false }),
  ).toBeTruthy()
  expect(
    within(sheet).getByRole('link', { name: 'Download' }).getAttribute('href'),
  ).toBe('/api/gallery/run1/a.png/download')
  // Back closes the viewer
  await hashEvent(() => history.back())
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
})

it('Compare takes the first four pictures, finals first', async () => {
  render(JobResults, {
    job,
    fileGroups: [
      group('boards', ['b1.png', 'b2.png', 'b3.png'], {
        role: 'intermediate',
      }),
      group('film', ['f1.mp4', 'f2.mp4', 'theme.wav'], { role: 'final' }),
    ],
    unsaved: [],
    kinds: {
      'b1.png': 'image',
      'b2.png': 'image',
      'b3.png': 'image',
      'f1.mp4': 'video',
      'f2.mp4': 'video',
      'theme.wav': 'audio',
    },
    seedVariable: null,
  })
  const compare = screen.getByRole('button', { name: 'Compare' })
  expect(compare.getAttribute('title')).toBe('Compares the first 4')
  await hashEvent(() => void fireEvent.click(compare))
  expect(location.hash).toBe('#/ws/w/compare/f1.mp4/f2.mp4/b1.png/b2.png')
})

it('offers no Compare for fewer than two pictures or clips', () => {
  render(JobResults, {
    job,
    fileGroups: [group('gen', ['a.png', 'theme.wav'])],
    unsaved: [],
    kinds: { 'a.png': 'image', 'theme.wav': 'audio' },
    seedVariable: null,
  })
  expect(screen.queryByRole('button', { name: 'Compare' })).toBeNull()
})

it('Download all archives every file', async () => {
  render(JobResults, {
    job,
    fileGroups: [group('gen', ['a.png', 'b.mp4']), group('cut', ['cut.otioz'])],
    unsaved: [],
    kinds: { 'a.png': 'image', 'b.mp4': 'video', 'cut.otioz': null },
    seedVariable: null,
  })
  await fireEvent.click(screen.getByRole('button', { name: 'Download all' }))
  await waitFor(() =>
    expect(api.archiveOutputs).toHaveBeenCalledWith([
      'a.png',
      'b.mp4',
      'cut.otioz',
    ]),
  )
})

it("a folder at the job's top reads (top folder)", () => {
  render(JobResults, {
    job,
    fileGroups: [
      group('gen', ['a.png']),
      group('film', ['final/f.mp4'], { subfolder: 'final' }),
    ],
    unsaved: [],
    kinds: { 'a.png': 'image', 'final/f.mp4': 'video' },
    seedVariable: null,
  })
  expect(
    screen
      .getAllByRole('heading', { level: 3 })
      .map((h) => h.textContent?.trim()),
  ).toEqual(['final/', '(top folder)'])
  expect(screen.queryByText('(run root)')).toBeNull()
})

const clips = {
  job,
  fileGroups: [group('film', ['run1/s3.mp4'])],
  unsaved: [],
  kinds: { 'run1/s3.mp4': 'video' },
  seedVariable: null,
}

it('a starred clip shows as starred once its recipe is read', async () => {
  starredOn.add('run1/s3.mp4')
  render(JobResults, clips)
  const sheet = await openOutput('run1/s3.mp4')
  const star = await within(sheet).findByRole('button', {
    name: 'Star',
    pressed: true,
  })
  expect(star.getAttribute('aria-pressed')).toBe('true')
  expect((star as HTMLButtonElement).disabled).toBe(false)
})

it('the star waits for the read that says whether the file is starred', async () => {
  held.reads = true
  render(JobResults, clips)
  const sheet = await openOutput('run1/s3.mp4')
  const star = within(sheet).getByRole('button', { name: 'Star' })
  expect((star as HTMLButtonElement).disabled).toBe(true)
  expect(star.title).toBe('Reading whether this file is starred')
  expect(star.hasAttribute('aria-pressed')).toBe(false)
  await fireEvent.click(star)
  expect(api.setStarred).not.toHaveBeenCalled()
})

it('the star is enabled from the listing when the recipe read fails', async () => {
  starredOn.add('run1/a.png')
  render(JobResults, {
    ...clips,
    fileGroups: [group('gen', ['run1/a.png'])],
    kinds: { 'run1/a.png': 'image' },
  })
  // let the listing's own read land, then fail the viewer's
  await waitFor(() => expect(api.galleryMetadata).toHaveBeenCalled())
  await new Promise((r) => setTimeout(r, 50))
  held.fail = true
  const sheet = await openOutput('run1/a.png')
  const star = await within(sheet).findByRole('button', {
    name: 'Star',
    pressed: true,
  })
  expect((star as HTMLButtonElement).disabled).toBe(false)
})

it("the star's Undo puts back a star the user did not just add", async () => {
  starredOn.add('run1/s3.mp4')
  render(JobResults, clips)
  const sheet = await openOutput('run1/s3.mp4')
  await fireEvent.click(
    await within(sheet).findByRole('button', { name: 'Star', pressed: true }),
  )
  await waitFor(() =>
    expect(api.setStarred).toHaveBeenCalledWith('run1/s3.mp4', false),
  )
  await waitFor(() => expect(notifyUndo).toHaveBeenCalled())
  expect(notifyUndo.mock.calls[0][0]).toBe('Unstarred s3.mp4')
  notifyUndo.mock.calls[0][1]()
  await waitFor(() => expect(api.setStarred).toHaveBeenCalledTimes(2))
  expect(api.setStarred.mock.calls[1]).toEqual(['run1/s3.mp4', true])
})

it('an open= naming no output is said once the job has finished, and cleared', async () => {
  history.replaceState(
    null,
    '',
    `${location.pathname}#/ws/w/jobs/j1?open=run1/gone.png`,
  )
  const view = render(JobResults, { ...clips, running: true })
  // while the job runs the file may still arrive: the link is kept
  expect(screen.queryByText(/is not in this view any more/)).toBeNull()
  expect(location.hash).toContain('open=')
  await view.rerender({ ...clips, running: false })
  await waitFor(() =>
    expect(
      screen.getByText(/gone\.png is not in this view any more\./),
    ).toBeTruthy(),
  )
  await waitFor(() => expect(location.hash).not.toContain('open='))
  expect(screen.queryByRole('dialog')).toBeNull()
})

it('an open= on a job with no files at all is said, not silent', async () => {
  history.replaceState(
    null,
    '',
    `${location.pathname}#/ws/w/jobs/j1?open=run1/gone.png`,
  )
  render(JobResults, { ...clips, fileGroups: [], running: false })
  await waitFor(() =>
    expect(
      screen.getByText(/gone\.png is not in this view any more\./),
    ).toBeTruthy(),
  )
})

it('a running job opens the file open= names once it arrives', async () => {
  history.replaceState(
    null,
    '',
    `${location.pathname}#/ws/w/jobs/j1?open=run1/s3.mp4`,
  )
  const view = render(JobResults, {
    ...clips,
    fileGroups: [],
    kinds: {},
    running: true,
  })
  expect(screen.queryByRole('dialog')).toBeNull()
  await view.rerender({ ...clips, running: true })
  expect(await screen.findByRole('dialog', { name: 's3.mp4' })).toBeTruthy()
  expect(screen.queryByText(/is not in this view any more/)).toBeNull()
})
