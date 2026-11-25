import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import GalleryAside from './GalleryAside.svelte'
import { api } from '../test/apiMock'
import { formatExact, formatMtime } from '../format'
import { attempt, item, request } from '../request/fixtures'
import type { GalleryFile, GalleryLinkedJob, GalleryMetadata } from '../types'

const setNote = vi.hoisted(() => vi.fn())
// What the server holds as the note a save reads first (its etag read)
const noteHeld = vi.hoisted(() => ({ note: null as string | null }))
vi.mock('../api', async () =>
  (await import('../test/apiMock')).apiModule({
    setNote: (t: string, n: string | null) => setNote(t, n),
  }),
)
api.noteForSave.mockImplementation((target: string) =>
  Promise.resolve({ target, note: noteHeld.note, etag: '"e"' }),
)
vi.mock('../toast', () => ({
  notify: { success: vi.fn(), error: vi.fn(), failure: vi.fn() },
}))
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))

const file: GalleryFile = {
  name: 'wf/run-1/final/a.png',
  folder: 'wf',
  subfolder: 'final',
  run_id: 'run-1',
  version: 4,
  url: '/outputs/wf/run-1/final/a.png',
  kind: 'image',
  size: 10,
  mtime: 1,
  label: 'a.png',
  role: 'final',
}

const meta = (over: Partial<GalleryMetadata> = {}): GalleryMetadata =>
  ({
    name: file.name,
    source: 'output',
    metadata: null,
    job: null,
    run_id: 'run-1',
    version: 4,
    media: null,
    role: 'final',
    inputs: [],
    findings: [],
    linked_jobs: [],
    note: null,
    run_note: null,
    note_source: null,
    ...over,
  }) as GalleryMetadata

const RUN = {
  note: 'for the client',
  run_note: 'for the client',
  note_source: 'run',
} as const

/** The aside under a panel whose read is `first`; `reread` is what the
 * panel's read again answers, and it hands the aside the fresh read as
 * the panel does */
function aside(first: GalleryMetadata, extra: Record<string, unknown> = {}) {
  const reread = vi.fn<() => Promise<GalleryMetadata | null>>()
  const shown = render(GalleryAside, {
    props: {
      file,
      meta: first,
      workspace: 'default',
      reload: async () => {
        const fresh = await reread()
        if (fresh) await shown.rerender({ meta: fresh })
        return fresh
      },
      ...extra,
    },
  })
  return { ...shown, reread }
}

beforeEach(() => {
  noteHeld.note = null
  sessionStorage.clear()
  setNote.mockReset()
})
afterEach(() => {
  cleanup()
  sessionStorage.clear()
})

it("a file inheriting its job's note shows that one, and Note this file swaps in its own", async () => {
  aside(meta(RUN))
  expect(screen.getByText('Your note on this job')).toBeTruthy()
  expect(screen.getByText('for the client')).toBeTruthy()
  await fireEvent.click(screen.getByRole('button', { name: 'Note this file' }))
  expect(screen.getByText('Your note on this file')).toBeTruthy()
  expect(screen.queryByText('for the client')).toBeNull()
})

it('says the role, the run and its version', () => {
  aside(meta())
  expect(document.querySelector('.role-tag')?.textContent).toBe('Final')
  const version = screen.getByTitle('4th job of this workflow in this folder')
  expect(version.textContent).toContain('v4')
  expect(version.querySelector('code')?.textContent).toBe('run-1')
})

it('says the size and when the file was written, exact, to the second on hover', () => {
  aside(meta())
  const time = document.querySelector('time') as HTMLElement
  expect(time.closest('span')?.textContent?.trim()).toBe(
    `1 KB · ${formatMtime(file.mtime)}`,
  )
  expect(time.getAttribute('title')).toBe(formatExact(file.mtime))
})

it('a half-written run note survives "Note this file" and coming back', async () => {
  aside(meta(RUN))
  await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
  await fireEvent.input(screen.getByRole('textbox'), {
    target: { value: 'half a thought' },
  })
  await fireEvent.click(screen.getByRole('button', { name: 'Note this file' }))
  expect(screen.getByText('Your note on this file')).toBeTruthy()
  // Coming back to the file: the run's box holds the draft
  cleanup()
  aside(meta(RUN))
  expect(screen.getByText('You have an unsaved draft.')).toBeTruthy()
  await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
  expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(
    'half a thought',
  )
})

it("shows the file's own note", () => {
  aside(meta({ note: 'keeper', note_source: 'file' }))
  expect(screen.getByText('keeper')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Note this file' })).toBeNull()
})

it("labels a saved file note as the file's own, and a cleared one returns the run note", async () => {
  const onnotesaved = vi.fn(async () => {})
  const { reread } = aside(meta(RUN), { onnotesaved })
  await fireEvent.click(screen.getByRole('button', { name: 'Note this file' }))
  setNote.mockResolvedValue({
    target: file.name,
    note: 'mine',
    note_source: 'file',
  })
  reread.mockResolvedValue(meta({ ...RUN, note: 'mine', note_source: 'file' }))
  await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
  await fireEvent.input(screen.getByRole('textbox'), {
    target: { value: 'mine' },
  })
  await fireEvent.click(screen.getByRole('button', { name: /^save$/i }))
  // The save reads the note's etag first, so it lands a tick later
  await waitFor(() => expect(onnotesaved).toHaveBeenCalledWith('file', 'mine'))
  expect(screen.getByText('Your note on this file')).toBeTruthy()
  expect(screen.getByText('mine')).toBeTruthy()

  noteHeld.note = 'mine'
  setNote.mockResolvedValue({
    target: file.name,
    note: 'for the client',
    note_source: 'run',
  })
  reread.mockResolvedValue(meta(RUN))
  await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
  await fireEvent.input(screen.getByRole('textbox'), { target: { value: '' } })
  await fireEvent.click(screen.getByRole('button', { name: /^save$/i }))
  await waitFor(() =>
    expect(screen.getByText('Your note on this job')).toBeTruthy(),
  )
  expect(screen.getByText('for the client')).toBeTruthy()
})

it('a note saved while the file will not read again tells the page nothing', async () => {
  const onnotesaved = vi.fn(async () => {})
  const { reread } = aside(meta({ note: 'old', note_source: 'file' }), {
    onnotesaved,
  })
  noteHeld.note = 'old'
  setNote.mockResolvedValue({
    target: file.name,
    note: 'new',
    note_source: 'file',
  })
  // failed, or overtaken by a later read: the panel hands back null
  reread.mockResolvedValue(null)
  await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
  await fireEvent.input(screen.getByRole('textbox'), {
    target: { value: 'new' },
  })
  await fireEvent.click(screen.getByRole('button', { name: /^save$/i }))
  await waitFor(() => expect(reread).toHaveBeenCalledTimes(1))
  for (let i = 0; i < 3; i++) await new Promise((r) => setTimeout(r, 10))
  expect(onnotesaved).not.toHaveBeenCalled()
})

const finding = (severity: 'warn' | 'info', says: string) => ({
  rule: `${severity}_rule`,
  severity,
  at: null,
  value: 1,
  threshold: 0,
  says,
})
const findingsRegion = () => screen.getByRole('region', { name: 'Findings' })

it("lists the read's findings warn first, and says No findings without", async () => {
  aside(
    meta({
      findings: [finding('info', 'quiet note'), finding('warn', 'too loud')],
    }),
  )
  const rows = [...findingsRegion().querySelectorAll('li')]
  expect(rows.map((r) => r.querySelector('.severity')?.textContent)).toEqual([
    'warn',
    'info',
  ])
  expect(rows[0].textContent).toContain('too loud')
  cleanup()
  aside(meta())
  expect(findingsRegion().textContent).toContain('No findings')
})

describe('the note for the agent', () => {
  const take = (id: string, n: number): GalleryLinkedJob => ({
    id,
    request: 'r1',
    item: 's3',
    attempt: n,
  })
  beforeEach(() => {
    api.getRequest.mockReset()
    api.addNote.mockReset()
    api.notesChannel.mockReset()
    api.notesChannel.mockResolvedValue({ listening: false, last_seen: null })
    api.addNote.mockResolvedValue({ revision: 4 })
    api.getRequest.mockResolvedValue(
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
    aside(meta({ linked_jobs: [] }))
    expect(screen.queryByText('Note for the agent')).toBeNull()
    expect(api.getRequest).not.toHaveBeenCalled()
  })

  it("with one take, shows the file's notes and a new one is about it", async () => {
    aside(meta({ linked_jobs: [take('j1', 1)] }))
    await screen.findByText('about this file')
    expect(screen.queryByText('about another file')).toBeNull()
    expect(screen.queryByRole('combobox')).toBeNull()
    await send('too much grain')
    await waitFor(() => expect(api.addNote).toHaveBeenCalled())
    expect(api.addNote).toHaveBeenCalledWith('default', {
      request: 'r1',
      item: 's3',
      text: 'too much grain',
      refs: [{ item: 's3', job: 'j1', output: file.name }],
    })
  })

  it('with two takes, asks which one first and notes the one chosen', async () => {
    aside(meta({ linked_jobs: [take('j2', 2), take('j1', 1)] }))
    const picker = screen.getByRole('combobox') as HTMLSelectElement
    expect(
      [...picker.options].slice(1).map((option) => option.textContent),
    ).toEqual(['take 2 of item s3', 'take 1 of item s3'])
    expect(screen.queryByText('Note for the agent')).toBeNull()
    await fireEvent.change(picker, { target: { value: 'j2' } })
    await screen.findByText('about this file')
    await send('the cached one is fine')
    await waitFor(() => expect(api.addNote).toHaveBeenCalled())
    expect(api.addNote.mock.calls[0][1].refs).toEqual([
      { item: 's3', job: 'j2', output: file.name },
    ])
  })
})
