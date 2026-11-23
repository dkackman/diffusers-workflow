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
    keepOutput: (...a: unknown[]) => keepOutput(...a),
    ...ledger,
  },
}))

vi.mock('../toast', () => ({ notify: notifyMock }))

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

const props = {
  file,
  metadata: { model_name: 'org/model', seed: 7 },
  metadataLoading: false,
  sourceJob: null,
  role: 'final' as const,
  inputs: [
    { reference: 'asset:face.png', kind: 'asset' as const },
    { reference: 'output:wf/run-0/b.png', kind: 'output' as const },
  ],
  onremove: () => {},
  onclose: () => {},
}

it('orders the panel role, Settings, Made from, Location', () => {
  const { container } = render(GalleryDetail, props)
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

it('lists each run input as a link', () => {
  const { container } = render(GalleryDetail, props)
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

it('omits Made from when the run read nothing', () => {
  const { container } = render(GalleryDetail, { ...props, inputs: [] })
  expect(container.querySelector('.madefrom')).toBeNull()
})

it('shows an inherited run note as the run note and offers to note the file', async () => {
  render(GalleryDetail, {
    ...props,
    note: 'for the client',
    runNote: 'for the client',
    noteSource: 'run' as const,
  })
  expect(screen.getByText('Run note')).toBeTruthy()
  expect(screen.getByText('for the client')).toBeTruthy()
  await fireEvent.click(screen.getByRole('button', { name: 'Note this file' }))
  expect(screen.getByText('File note')).toBeTruthy()
  expect(screen.queryByText('for the client')).toBeNull()
})

it("shows the file's own note", () => {
  render(GalleryDetail, {
    ...props,
    note: 'keeper',
    noteSource: 'file' as const,
  })
  expect(screen.getByText('keeper')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Note this file' })).toBeNull()
})

it("labels a saved file note as the file's own, and a cleared one returns the run note", async () => {
  const run = {
    ...props,
    note: 'for the client',
    runNote: 'for the client',
    noteSource: 'run' as const,
  }
  const spy = setNote
  const onnotesaved = vi.fn(async () => {})
  const { rerender } = render(GalleryDetail, { ...run, onnotesaved })
  await fireEvent.click(screen.getByRole('button', { name: 'Note this file' }))
  spy.mockResolvedValue({
    target: file.name,
    note: 'mine',
    note_source: 'file',
  })
  onnotesaved.mockImplementation(async () => {
    await rerender({ ...run, note: 'mine', noteSource: 'file', onnotesaved })
  })
  await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
  await fireEvent.input(screen.getByRole('textbox'), {
    target: { value: 'mine' },
  })
  await fireEvent.click(screen.getByRole('button', { name: /save/i }))
  await waitFor(() => expect(screen.getByText('File note')).toBeTruthy())
  expect(screen.getByText('mine')).toBeTruthy()

  spy.mockResolvedValue({
    target: file.name,
    note: 'for the client',
    note_source: 'run',
  })
  onnotesaved.mockImplementation(async () => {
    await rerender({ ...run, onnotesaved })
  })
  await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
  await fireEvent.input(screen.getByRole('textbox'), { target: { value: '' } })
  await fireEvent.click(screen.getByRole('button', { name: /save/i }))
  await waitFor(() => expect(screen.getByText('Run note')).toBeTruthy())
  expect(screen.getByText('for the client')).toBeTruthy()
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

it("lists the metadata's findings warn first", () => {
  render(GalleryDetail, {
    ...props,
    findings: [finding('info', 'quiet note'), finding('warn', 'too loud')],
  })
  const rows = [...findingsRegion().querySelectorAll('li')]
  expect(rows.map((r) => r.querySelector('.severity')?.textContent)).toEqual([
    'warn',
    'info',
  ])
  expect(rows[0].textContent).toContain('too loud')
})

it('says No findings when there are none', () => {
  render(GalleryDetail, props)
  expect(findingsRegion().textContent).toContain('No findings')
})

it("disables an image's Assess and says why", () => {
  render(GalleryDetail, props)
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
  render(GalleryDetail, { ...props, file: video })
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
  render(GalleryDetail, { ...props, file: video })
  await fireEvent.click(screen.getByRole('button', { name: 'Assess' }))
  const busy = await screen.findByRole('button', { name: /Assessing/ })
  expect((busy as HTMLButtonElement).disabled).toBe(true)
})

it('a failed Assess keeps the earlier list and shows the error', async () => {
  assess.mockResolvedValueOnce(reply([finding('warn', 'silent')]))
  render(GalleryDetail, { ...props, file: video })
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

  it('is not there for a file no take of a request item published', () => {
    render(GalleryDetail, { ...props, linkedJobs: [] })
    expect(screen.queryByText('Note for the agent')).toBeNull()
    expect(ledger.getRequest).not.toHaveBeenCalled()
  })

  it("with one take, shows the file's notes and a new one is about it", async () => {
    render(GalleryDetail, { ...props, linkedJobs: [take('j1', 1)] })
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
    render(GalleryDetail, {
      ...props,
      linkedJobs: [take('j2', 2), take('j1', 1)],
    })
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
  render(GalleryDetail, props)
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
  render(GalleryDetail, props)
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
  render(GalleryDetail, props)
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
