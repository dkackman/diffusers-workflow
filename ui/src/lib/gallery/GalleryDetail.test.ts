import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/svelte'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import GalleryDetail from './GalleryDetail.svelte'
import ConfirmDialog from '../ui/ConfirmDialog.svelte'
import { attempt, item, request } from '../request/fixtures'
import type { GalleryFile, GalleryLinkedJob } from '../types'

const setNote = vi.hoisted(() => vi.fn())
const keepOutput = vi.hoisted(() => vi.fn())
const notifyMock = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  failure: vi.fn(),
}))
const assess = vi.hoisted(() => vi.fn<(name: string) => Promise<unknown>>())
const galleryMetadata = vi.hoisted(() => vi.fn())
const fileText = vi.hoisted(() => vi.fn<(url: string) => Promise<string>>())
// The request-notes calls the file's note for the agent makes
const ledger = vi.hoisted(() => ({
  getRequest: vi.fn(),
  addNote: vi.fn(),
  notesChannel: vi.fn(),
}))
vi.mock('../api', async () => ({
  ApiError: (await import('../apiErrors')).ApiError,
  setNote: (t: string, n: string | null) => setNote(t, n),
  api: {
    outputDownloadUrl: (name: string) => `/download/${name}`,
    assess: (name: string) => assess(name),
    galleryMetadata: (name: string) => galleryMetadata(name),
    keepOutput: (...a: unknown[]) => keepOutput(...a),
    fileText: (url: string) => fileText(url),
    ...ledger,
  },
}))

vi.mock('../toast', () => ({ notify: notifyMock }))
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))

afterEach(() => {
  keepOutput.mockReset()
  notifyMock.success.mockReset()
  notifyMock.error.mockReset()
  notifyMock.failure.mockReset()
  vi.restoreAllMocks()
  cleanup()
  assess.mockReset()
})

const file: GalleryFile = {
  name: 'wf/run-1/final/a.png',
  folder: 'wf',
  subfolder: 'final',
  run_id: 'run-1',
  version: 1,
  url: '/outputs/wf/run-1/final/a.png',
  kind: 'image',
  size: 10,
  mtime: 1,
  label: 'a.png',
  role: 'final',
}

const props = { file, onremove: () => {}, onclose: () => {} }

/** What the metadata route answers, the run having read two inputs */
const metaReply = (over: Record<string, unknown> = {}) => ({
  metadata: { model_name: 'org/model', seed: 7 },
  job: null,
  version: 1,
  role: 'final',
  inputs: [
    { reference: 'asset:face.png', kind: 'asset' },
    { reference: 'output:wf/run-0/b.png', kind: 'output' },
  ],
  findings: [],
  linked_jobs: [],
  note: null,
  run_note: null,
  note_source: null,
  ...over,
})

beforeEach(() => {
  galleryMetadata.mockReset()
  galleryMetadata.mockResolvedValue(metaReply())
})

/** Render the panel and wait for its metadata to land */
async function view(over: Record<string, unknown> = {}, extra = {}) {
  galleryMetadata.mockResolvedValue(metaReply(over))
  const shown = render(GalleryDetail, { ...props, ...extra })
  await screen.findByRole('region', { name: 'Findings' })
  return shown
}

it('orders the panel role, Settings, Made from, Location', async () => {
  const { container } = await view()
  const meta = container.querySelector('.meta') as HTMLElement
  const order = [...meta.children].map(
    (el) =>
      (el.classList.contains('madefrom') && 'Made from') ||
      (el.querySelector('.role-tag') && 'role') ||
      (el.matches('details') && el.querySelector('summary')?.textContent) ||
      el.querySelector('.muted')?.textContent,
  )
  expect(order).toEqual([
    'role',
    'Note',
    'Findings',
    'Settings',
    'Made from',
    'Location',
  ])
  // Settings starts collapsed
  expect((meta.querySelector('details') as HTMLDetailsElement).open).toBe(false)
  expect(meta.querySelector('.location code')?.textContent).toBe(
    'wf/run-1/final/',
  )
})

it('lists each run input as a link', async () => {
  const { container } = await view()
  const links = [...container.querySelectorAll('.madefrom a')]
  expect(links.map((a) => a.textContent)).toEqual([
    'face.png',
    'wf/run-0/b.png',
  ])
  expect(
    [...container.querySelectorAll('.madefrom .role-tag')].map(
      (t) => (t as HTMLElement).dataset.role,
    ),
  ).toEqual(['input', 'input'])
  expect(links[1].getAttribute('href')).toContain('wf/run-0/b.png')
})

it('omits Made from when the run read nothing', async () => {
  const { container } = await view({ inputs: [] })
  expect(container.querySelector('.madefrom')).toBeNull()
})

it('shows an inherited run note as the run note and offers to note the file', async () => {
  await view({
    note: 'for the client',
    run_note: 'for the client',
    note_source: 'run',
  })
  expect(screen.getByText('Run note')).toBeTruthy()
  expect(screen.getByText('for the client')).toBeTruthy()
  await fireEvent.click(screen.getByRole('button', { name: 'Note this file' }))
  expect(screen.getByText('File note')).toBeTruthy()
  expect(screen.queryByText('for the client')).toBeNull()
})

it("shows the file's own note", async () => {
  await view({ note: 'keeper', note_source: 'file' })
  expect(screen.getByText('keeper')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Note this file' })).toBeNull()
})

it("labels a saved file note as the file's own, and a cleared one returns the run note", async () => {
  const run = {
    note: 'for the client',
    run_note: 'for the client',
    note_source: 'run',
  }
  const onnotesaved = vi.fn(async () => {})
  await view(run, { onnotesaved })
  await fireEvent.click(screen.getByRole('button', { name: 'Note this file' }))
  setNote.mockResolvedValue({
    target: file.name,
    note: 'mine',
    note_source: 'file',
  })
  galleryMetadata.mockResolvedValue(
    metaReply({ ...run, note: 'mine', note_source: 'file' }),
  )
  await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
  await fireEvent.input(screen.getByRole('textbox'), {
    target: { value: 'mine' },
  })
  await fireEvent.click(screen.getByRole('button', { name: /save/i }))
  await waitFor(() => expect(screen.getByText('File note')).toBeTruthy())
  expect(screen.getByText('mine')).toBeTruthy()
  expect(onnotesaved).toHaveBeenCalledWith('file', 'mine')

  setNote.mockResolvedValue({
    target: file.name,
    note: 'for the client',
    note_source: 'run',
  })
  galleryMetadata.mockResolvedValue(metaReply(run))
  await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
  await fireEvent.input(screen.getByRole('textbox'), { target: { value: '' } })
  await fireEvent.click(screen.getByRole('button', { name: /save/i }))
  await waitFor(() => expect(screen.getByText('Run note')).toBeTruthy())
  expect(screen.getByText('for the client')).toBeTruthy()
})

it('a note saved while the metadata will not read again tells the page nothing', async () => {
  const onnotesaved = vi.fn(async () => {})
  await view({ note: 'old', note_source: 'file' }, { onnotesaved })
  setNote.mockResolvedValue({
    target: file.name,
    note: 'new',
    note_source: 'file',
  })
  galleryMetadata.mockRejectedValue(new Error('boom'))
  await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
  await fireEvent.input(screen.getByRole('textbox'), {
    target: { value: 'new' },
  })
  await fireEvent.click(screen.getByRole('button', { name: /save/i }))
  await waitFor(() => expect(setNote).toHaveBeenCalledWith(file.name, 'new'))
  await waitFor(() => expect(galleryMetadata).toHaveBeenCalledTimes(2))
  for (let i = 0; i < 3; i++) await new Promise((r) => setTimeout(r, 10))
  // The pre-save note would be all the panel holds to hand on
  expect(onnotesaved).not.toHaveBeenCalled()
})

it('a note save whose reread is overtaken hands on nothing from before it', async () => {
  const { feed } = await import('../test/feedMock')
  const onnotesaved = vi.fn(async () => {})
  await view({ note: 'old', note_source: 'file' }, { onnotesaved })
  setNote.mockResolvedValue({
    target: file.name,
    note: 'new',
    note_source: 'file',
  })
  // The save's own reread, then a feed reload that overtakes it
  let loud: (v: unknown) => void = () => {}
  let quiet: (v: unknown) => void = () => {}
  galleryMetadata
    .mockImplementationOnce(() => new Promise((r) => (loud = r)))
    .mockImplementationOnce(() => new Promise((r) => (quiet = r)))
  await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
  await fireEvent.input(screen.getByRole('textbox'), {
    target: { value: 'new' },
  })
  await fireEvent.click(screen.getByRole('button', { name: /save/i }))
  await waitFor(() => expect(galleryMetadata).toHaveBeenCalledTimes(2))
  for (const w of [...feed.watches])
    if (w.kinds.includes('gallery')) void w.fn()
  await waitFor(() => expect(galleryMetadata).toHaveBeenCalledTimes(3))
  loud(metaReply({ note: 'old', note_source: 'file' }))
  for (let i = 0; i < 3; i++) await new Promise((r) => setTimeout(r, 10))
  expect(onnotesaved).not.toHaveBeenCalled()
  quiet(metaReply({ note: 'new', note_source: 'file' }))
  await waitFor(() => expect(screen.getByText('new')).toBeTruthy())
})

it('a failed metadata read is not "no embedded metadata"', async () => {
  galleryMetadata.mockRejectedValue(new Error('boom'))
  render(GalleryDetail, props)
  await screen.findByText(/Couldn't load what made this file/)
  expect(screen.queryByText(/no embedded metadata/)).toBeNull()
  galleryMetadata.mockResolvedValue(metaReply())
  await fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await screen.findByRole('region', { name: 'Findings' })
})

it('reads the metadata as a skeleton, then says when an image has none', async () => {
  galleryMetadata.mockResolvedValue(
    metaReply({ metadata: null, role: null, inputs: [] }),
  )
  render(GalleryDetail, { ...props, file: { ...file, role: undefined } })
  expect(
    screen.getByRole('status', { name: 'loading what made this file' }),
  ).toBeTruthy()
  await screen.findByText(/no embedded metadata/)
})

const finding = (
  severity: 'warn' | 'info',
  says: string,
  at: number | null = null,
) => ({
  rule: `${severity}_rule`,
  severity,
  at,
  value: 1,
  threshold: 0,
  says,
})

const video: GalleryFile = {
  ...file,
  name: 'wf/run-1/final/clip.mp4',
  url: '/outputs/wf/run-1/final/clip.mp4',
  kind: 'video',
  label: 'clip.mp4',
}

const reply = (findings: ReturnType<typeof finding>[]) => ({
  name: video.name,
  source: 'output',
  kind: 'video',
  findings,
  rules_applied: ['near_silent', 'shot_dead_air'],
  rules_skipped: [],
  not_applicable: { analyze_seams: 'no shot boundaries recorded' },
})

const findingsRegion = () => screen.getByRole('region', { name: 'Findings' })

it("lists the metadata's findings warn first", async () => {
  await view({
    findings: [finding('info', 'quiet note'), finding('warn', 'too loud')],
  })
  const rows = [...findingsRegion().querySelectorAll('li')]
  expect(rows.map((r) => r.querySelector('.severity')?.textContent)).toEqual([
    'warn',
    'info',
  ])
  expect(rows[0].textContent).toContain('too loud')
})

it('says No findings when there are none', async () => {
  await view()
  expect(findingsRegion().textContent).toContain('No findings')
})

it("disables an image's Assess and says why", async () => {
  await view()
  const button = screen.getByRole('button', { name: 'Assess' })
  expect((button as HTMLButtonElement).disabled).toBe(true)
  expect(button.getAttribute('title')).toBe(
    'a still has no shots, seams or soundtrack to measure',
  )
  expect(assess).not.toHaveBeenCalled()
})

it("Assess shows the reply's findings, their time and what did not apply", async () => {
  assess.mockResolvedValue(
    reply([finding('info', 'picture jumps', 64.25), finding('warn', 'silent')]),
  )
  await view({}, { file: video })
  await fireEvent.click(screen.getByRole('button', { name: 'Assess' }))
  expect(assess).toHaveBeenCalledWith(video.name)
  await waitFor(() =>
    expect(findingsRegion().querySelectorAll('li')).toHaveLength(2),
  )
  const rows = [...findingsRegion().querySelectorAll('li')]
  expect(rows[0].textContent).toContain('silent')
  expect(rows[1].querySelector('.at')?.textContent).toBe('1:04.3')
  expect(findingsRegion().textContent).toContain(
    'analyze_seams not applicable (no shot boundaries recorded)',
  )
})

it('shows Assessing… while the request runs', async () => {
  assess.mockImplementation(() => new Promise(() => {}))
  await view({}, { file: video })
  await fireEvent.click(screen.getByRole('button', { name: 'Assess' }))
  const busy = await screen.findByRole('button', { name: /Assessing/ })
  expect((busy as HTMLButtonElement).disabled).toBe(true)
})

it('a failed Assess keeps the earlier list and shows the error', async () => {
  assess.mockResolvedValueOnce(reply([finding('warn', 'silent')]))
  await view({}, { file: video })
  await fireEvent.click(screen.getByRole('button', { name: 'Assess' }))
  await waitFor(() => expect(findingsRegion().textContent).toContain('silent'))
  assess.mockRejectedValueOnce(new Error('cannot read this file'))
  await fireEvent.click(screen.getByRole('button', { name: 'Assess' }))
  expect((await screen.findByRole('alert')).textContent).toContain(
    'cannot read this file',
  )
  expect(findingsRegion().textContent).toContain('silent')
})

describe('the note for the agent', () => {
  const take = (id: string, n: number): GalleryLinkedJob => ({
    id,
    request: 'r1',
    item: 's3',
    attempt: n,
  })
  beforeEach(() => {
    sessionStorage.clear()
    for (const call of Object.values(ledger)) call.mockReset()
    ledger.notesChannel.mockResolvedValue({ listening: false, last_seen: null })
    ledger.addNote.mockResolvedValue({ revision: 4 })
    ledger.getRequest.mockResolvedValue(
      request([
        item('s3', {
          title: 'Shot 3',
          status: 'done',
          attempts: [attempt('j1'), attempt('j2')],
          notes: [
            {
              id: 'n1',
              text: 'about this file',
              created: 'x',
              at: 'x',
              refs: [{ item: 's3', job: 'j1', output: file.name }],
            },
            {
              id: 'n2',
              text: 'about another file',
              created: 'x',
              at: 'x',
              refs: [{ item: 's3', job: 'j1', output: 'wf/run-1/b.png' }],
            },
          ],
        }),
      ]),
    )
  })
  const box = () => screen.getByLabelText('Note for the agent on Shot 3')
  async function send(text: string) {
    await fireEvent.click(
      screen.getByRole('button', { name: 'Add a note for the agent' }),
    )
    await fireEvent.input(box(), { target: { value: text } })
    await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  }

  it('is not there for a file no take of a request item published', async () => {
    await view({ linked_jobs: [] })
    expect(screen.queryByText('Note for the agent')).toBeNull()
    expect(ledger.getRequest).not.toHaveBeenCalled()
  })

  it("with one take, shows the file's notes and a new one is about it", async () => {
    await view({ linked_jobs: [take('j1', 1)] })
    await screen.findByText('about this file')
    expect(screen.queryByText('about another file')).toBeNull()
    expect(screen.queryByRole('combobox')).toBeNull()
    await send('too much grain')
    await waitFor(() => expect(ledger.addNote).toHaveBeenCalled())
    expect(ledger.addNote).toHaveBeenCalledWith('default', {
      request: 'r1',
      item: 's3',
      text: 'too much grain',
      refs: [{ item: 's3', job: 'j1', output: file.name }],
    })
  })

  it('with two takes, asks which one first and notes the one chosen', async () => {
    await view({ linked_jobs: [take('j2', 2), take('j1', 1)] })
    const picker = screen.getByRole('combobox') as HTMLSelectElement
    expect(
      [...picker.options].slice(1).map((option) => option.textContent),
    ).toEqual(['take 2 of item s3', 'take 1 of item s3'])
    expect(screen.queryByText('Note for the agent')).toBeNull()
    await fireEvent.change(picker, { target: { value: 'j2' } })
    await screen.findByText('about this file')
    await send('the cached one is fine')
    await waitFor(() => expect(ledger.addNote).toHaveBeenCalled())
    expect(ledger.addNote.mock.calls[0][1].refs).toEqual([
      { item: 's3', job: 'j2', output: file.name },
    ])
  })
})

// keep_output's exists refusal carries no code, only this wording
// (dw/server/routes/assets.py)
const TAKEN =
  'asset:face.png already exists - pass overwrite=true to replace it'

it('cancelling Replace on keep-as-asset shows no error', async () => {
  const { ApiError } = await import('../apiErrors')
  vi.spyOn(window, 'prompt').mockReturnValue('face.png')
  keepOutput.mockRejectedValueOnce(new ApiError(TAKEN, 409))
  render(ConfirmDialog)
  await view()
  await fireEvent.click(screen.getByRole('button', { name: /keep as asset/i }))
  const dialog = await waitFor(() => screen.getByRole('alertdialog'))
  expect(dialog.textContent).toContain(
    'asset:face.png already exists. Replace it?',
  )
  await fireEvent.click(
    within(dialog).getByRole('button', { name: /^cancel$/i }),
  )
  await new Promise((r) => setTimeout(r, 10))
  expect(keepOutput).toHaveBeenCalledTimes(1)
  expect(notifyMock.error).not.toHaveBeenCalled()
  expect(notifyMock.success).not.toHaveBeenCalled()
})

it('accepting Replace on keep-as-asset retries with overwrite', async () => {
  const { ApiError } = await import('../apiErrors')
  vi.spyOn(window, 'prompt').mockReturnValue('face.png')
  keepOutput
    .mockRejectedValueOnce(new ApiError(TAKEN, 409))
    .mockResolvedValueOnce({ reference: 'asset:face.png' })
  render(ConfirmDialog)
  await view()
  await fireEvent.click(screen.getByRole('button', { name: /keep as asset/i }))
  const dialog = await waitFor(() => screen.getByRole('alertdialog'))
  await fireEvent.click(
    within(dialog).getByRole('button', { name: /^replace$/i }),
  )
  await waitFor(() => expect(keepOutput).toHaveBeenCalledTimes(2))
  expect(keepOutput).toHaveBeenLastCalledWith(file.name, 'face.png', true)
})

it('a keep refused for another reason shows the reason, not Replace', async () => {
  const { ApiError } = await import('../apiErrors')
  vi.spyOn(window, 'prompt').mockReturnValue('face.png')
  keepOutput.mockRejectedValueOnce(
    new ApiError('This workspace has no asset library', 409),
  )
  render(ConfirmDialog)
  await view()
  await fireEvent.click(screen.getByRole('button', { name: /keep as asset/i }))
  await waitFor(() =>
    expect(notifyMock.failure).toHaveBeenCalledWith(
      expect.stringContaining('keep'),
      expect.objectContaining({
        message: 'This workspace has no asset library',
      }),
    ),
  )
  expect(screen.queryByRole('alertdialog')).toBeNull()
  expect(keepOutput).toHaveBeenCalledTimes(1)
})

describe('a text file', () => {
  const text: GalleryFile = {
    ...file,
    name: 'wf/run-1/final/prompt.txt',
    url: '/outputs/wf/run-1/final/prompt.txt',
    kind: 'text',
    label: 'prompt.txt',
  }

  it('shows its contents', async () => {
    fileText.mockResolvedValue('a red fox')
    render(GalleryDetail, { ...props, file: text })
    await screen.findByText('a red fox')
    expect(fileText).toHaveBeenCalledWith(text.url)
  })

  it('reads a network failure as the offline copy', async () => {
    const { NetworkError } = await import('../apiErrors')
    fileText.mockRejectedValue(new NetworkError('Failed to fetch'))
    render(GalleryDetail, { ...props, file: text })
    await screen.findByText("Can't reach the server")
    expect(screen.queryByText(/Something went wrong/)).toBeNull()
    expect(screen.queryByText('Failed to fetch')).toBeNull()
  })
})
