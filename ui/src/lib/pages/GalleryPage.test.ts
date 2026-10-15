import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
// Hoisted above the imports so the static import of the component below -
// itself hoisted - sees an initialized mock. Importing the component inside
// the test instead would charge its (multi-second) compile to the test timeout
import GalleryPage from './GalleryPage.svelte'
import ConfirmDialog from '../ui/ConfirmDialog.svelte'
import type { GalleryFile } from '../types'
import { DEFAULT_WORKSPACE, workspace } from '../workspace.svelte'

const file = (
  name: string,
  subfolder = '',
  role?: GalleryFile['role'],
): GalleryFile => ({
  name,
  role,
  folder: name.includes('/') ? name.split('/')[0] : '',
  subfolder,
  run_id: '',
  version: null,
  url: `/outputs/${name}`,
  kind: 'image',
  size: 1024,
  mtime: 1,
  label: name.split('/').pop()!.split('.')[0],
})

const listing = vi.hoisted(() => ({ files: [] as GalleryFile[] }))

// Resolves on a macrotask so that a re-triggering effect cannot starve the
// timers this test waits on - the failure then shows as a call count
const gallery = vi.hoisted(() =>
  vi.fn(
    () =>
      new Promise<{ files: GalleryFile[] }>((resolve) =>
        setTimeout(() => resolve({ files: listing.files }), 0),
      ),
  ),
)
const deleteOutput = vi.hoisted(() =>
  vi.fn<(name: string) => Promise<void>>(() => Promise.resolve()),
)
const archiveOutputs = vi.hoisted(() =>
  vi.fn<(names: string[]) => Promise<void>>(() => Promise.resolve()),
)
const setNote = vi.hoisted(() => vi.fn())
const setStarred = vi.hoisted(() =>
  vi.fn((name: string, starred: boolean) => Promise.resolve({ name, starred })),
)
const galleryMetadata = vi.hoisted(() =>
  vi.fn<(name: string) => Promise<unknown>>(() => new Promise(() => {})),
)
vi.mock('../api', () => ({
  setNote: (t: string, n: string | null) => setNote(t, n),
  api: {
    gallery: () => gallery(),
    galleryThumbnailUrl: (name: string) => `/thumb/${name}`,
    galleryUrl: (name: string) => `/outputs/${name}`,
    outputDownloadUrl: (name: string) => `/download/${name}`,
    deleteOutput: (name: string) => deleteOutput(name),
    setStarred: (name: string, starred: boolean) => setStarred(name, starred),
    archiveOutputs: (names: string[]) => archiveOutputs(names),
    galleryMetadata: (name: string) => galleryMetadata(name),
  },
}))

const notifyError = vi.hoisted(() => vi.fn())
vi.mock('../toast', () => ({ notify: { error: notifyError } }))

beforeEach(() => {
  listing.files = ['a.png', 'b.png', 'demo/c.png'].map((name) => file(name))
})
afterEach(() => {
  // Without this each render's DOM stays behind and the next test's
  // queries find two of everything
  cleanup()
  gallery.mockClear()
  deleteOutput.mockClear()
  setStarred.mockClear()
  archiveOutputs.mockClear()
  notifyError.mockClear()
  galleryMetadata.mockReset()
  galleryMetadata.mockImplementation(() => new Promise(() => {}))
  vi.unstubAllGlobals()
  workspace.current = DEFAULT_WORKSPACE
})

/** The grid's per-file selection checkbox. */
const checkbox = (name: string) =>
  screen.getByRole('checkbox', { name: `select ${name}` })

async function renderGallery(first = 'a.png') {
  // ConfirmDialog is normally mounted once in App.svelte and driven through
  // the shared confirm.svelte.ts state - render it alongside so a test can
  // answer the dialogs GalleryPage's delete/replace flows open.
  render(ConfirmDialog)
  render(GalleryPage)
  await waitFor(() =>
    expect(screen.getByLabelText(`select ${first}`)).toBeTruthy(),
  )
}

/** Answers the confirm dialog opened by a delete/replace action - scoped to
 * the dialog itself, since its "Delete" button shares a name with whatever
 * trigger button opened it. */
// A key press starts at the focused element and bubbles to the document
// and the window - where an overlay and the page each listen
function pressEscape() {
  ;(document.activeElement ?? document.body).dispatchEvent(
    new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }),
  )
}

/** Opens a file's detail panel by clicking its tile. */
const openFile = (name: string) =>
  fireEvent.click(
    screen.getByRole('button', {
      name: new RegExp(`^${name.replace('.', '\\.')}`),
    }),
  )

async function answerConfirm(accept: boolean) {
  const dialog = await waitFor(() => screen.getByRole('alertdialog'))
  within(dialog)
    .getByRole('button', { name: accept ? /^delete$/i : /^cancel$/i })
    .click()
}

it('fetches the gallery listing exactly once on mount', async () => {
  render(GalleryPage)
  // Let the request settle and any (wrongly) re-triggered effects run
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 10))
  expect(gallery).toHaveBeenCalledTimes(1)
})

it('counts the files ticked in the grid', async () => {
  await renderGallery()

  checkbox('a.png').click()
  checkbox('demo/c.png').click()

  await waitFor(() => expect(screen.getByText('2 selected')).toBeTruthy())
})

it('shift-clicking extends the selection across the range', async () => {
  await renderGallery()

  checkbox('a.png').click()
  checkbox('demo/c.png').dispatchEvent(
    new MouseEvent('click', { bubbles: true, shiftKey: true }),
  )

  await waitFor(() => expect(screen.getByText('3 selected')).toBeTruthy())
})

it('selects only the files the filter leaves visible', async () => {
  listing.files = ['keep-a.png', 'keep-b.png', 'other.png'].map((name) =>
    file(name),
  )
  await renderGallery('keep-a.png')

  const filter = screen.getByPlaceholderText('filter…') as HTMLInputElement
  filter.value = 'keep'
  filter.dispatchEvent(new Event('input', { bubbles: true }))
  await waitFor(() =>
    expect(screen.queryByLabelText('select other.png')).toBeNull(),
  )
  screen.getByRole('button', { name: /select all/i }).click()

  await waitFor(() => expect(screen.getByText('2 selected')).toBeTruthy())
})

it('archives every selected file in one request', async () => {
  await renderGallery()

  checkbox('a.png').click()
  checkbox('b.png').click()
  await waitFor(() => expect(screen.getByText('2 selected')).toBeTruthy())
  screen.getByRole('button', { name: /download/i }).click()

  await waitFor(() =>
    expect(archiveOutputs).toHaveBeenCalledWith(['a.png', 'b.png']),
  )
})

it('drops deleted files from the grid and the selection', async () => {
  await renderGallery()

  checkbox('a.png').click()
  checkbox('b.png').click()
  await waitFor(() => expect(screen.getByText('2 selected')).toBeTruthy())
  screen.getByRole('button', { name: /^delete/i }).click()
  await answerConfirm(true)

  await waitFor(() =>
    expect(screen.queryByLabelText('select a.png')).toBeNull(),
  )
  expect(screen.queryByLabelText('select b.png')).toBeNull()
  expect(screen.queryByText(/selected/)).toBeNull()
  expect(deleteOutput).toHaveBeenCalledTimes(2)
})

it('keeps a file that failed to delete selected and reports it', async () => {
  deleteOutput.mockImplementation((name: string) =>
    name === 'b.png' ? Promise.reject(new Error('busy')) : Promise.resolve(),
  )
  await renderGallery()

  checkbox('a.png').click()
  checkbox('b.png').click()
  await waitFor(() => expect(screen.getByText('2 selected')).toBeTruthy())
  screen.getByRole('button', { name: /^delete/i }).click()
  await answerConfirm(true)

  await waitFor(() => expect(screen.getByText('1 selected')).toBeTruthy())
  expect(screen.queryByLabelText('select a.png')).toBeNull()
  expect(screen.getByLabelText('select b.png')).toBeTruthy()
  expect(notifyError).toHaveBeenCalled()
})

it('does not delete anything when the confirmation is declined', async () => {
  await renderGallery()

  checkbox('a.png').click()
  await waitFor(() => expect(screen.getByText('1 selected')).toBeTruthy())
  screen.getByRole('button', { name: /^delete/i }).click()
  await answerConfirm(false)

  await new Promise((r) => setTimeout(r, 10))
  expect(deleteOutput).not.toHaveBeenCalled()
})

it('drops a file deleted from the detail panel out of the selection', async () => {
  await renderGallery()

  checkbox('a.png').click()
  checkbox('b.png').click()
  await waitFor(() => expect(screen.getByText('2 selected')).toBeTruthy())
  // open a.png's details and delete it from there, not from the bulk bar
  // (a tile's accessible name comes from its thumbnail's alt text)
  screen.getByRole('button', { name: /^a\.png/ }).click()
  await waitFor(() =>
    expect(
      screen.getByRole('button', {
        name: 'delete this file from the output directory',
      }),
    ).toBeTruthy(),
  )
  screen
    .getByRole('button', { name: 'delete this file from the output directory' })
    .click()
  await answerConfirm(true)

  await waitFor(() => expect(screen.getByText('1 selected')).toBeTruthy())
  expect(screen.getByLabelText('select b.png')).toBeTruthy()
})

const roleFiles = () => [
  file('wf/run-1/deliverable.png', 'final', 'final'),
  file('wf/run-1/scratch.png', 'intermediate', 'intermediate'),
]

it('opens on finals only and shows each tile its role', async () => {
  listing.files = roleFiles()
  await renderGallery('wf/run-1/deliverable.png')

  expect(
    screen.getByRole('button', { name: 'Final' }).getAttribute('aria-pressed'),
  ).toBe('true')
  expect(
    screen
      .getByRole('button', { name: 'Intermediate' })
      .getAttribute('aria-pressed'),
  ).toBe('false')
  expect(screen.queryByLabelText('select wf/run-1/scratch.png')).toBeNull()
  const tag = document.querySelector('.cell .role-tag') as HTMLElement
  expect(tag.textContent).toBe('Final')
  expect(tag.dataset.role).toBe('final')
})

it('counts a file with no role as final', async () => {
  listing.files = [file('flat.png')]
  await renderGallery('flat.png')
  expect(screen.getByLabelText('select flat.png')).toBeTruthy()
})

it('toggling Intermediate shows intermediates too, finals first', async () => {
  listing.files = [...roleFiles()].reverse()
  await renderGallery('wf/run-1/deliverable.png')

  screen.getByRole('button', { name: 'Intermediate' }).click()

  await waitFor(() =>
    expect(screen.getByLabelText('select wf/run-1/scratch.png')).toBeTruthy(),
  )
  const order = screen
    .getAllByRole('checkbox')
    .map((c) => c.getAttribute('aria-label'))
  expect(order).toEqual([
    'select wf/run-1/deliverable.png',
    'select wf/run-1/scratch.png',
  ])
})

it('keeps newest-first between runs and puts finals first only inside a run', async () => {
  const inRun = (
    run: string,
    name: string,
    role: 'final' | 'intermediate',
  ) => ({
    ...file(`wf/${run}/${name}.png`, role, role),
    folder: 'wf',
    run_id: run,
  })
  // The server lists newest first: the newer run's intermediate leads
  listing.files = [
    inRun('new', 'scratch', 'intermediate'),
    inRun('old', 'scratch', 'intermediate'),
    inRun('new', 'deliverable', 'final'),
    inRun('old', 'deliverable', 'final'),
  ]
  await renderGallery('wf/new/deliverable.png')
  screen.getByRole('button', { name: 'Intermediate' }).click()
  await waitFor(() =>
    expect(screen.getByLabelText('select wf/new/scratch.png')).toBeTruthy(),
  )
  expect(
    screen.getAllByRole('checkbox').map((c) => c.getAttribute('aria-label')),
  ).toEqual([
    'select wf/new/deliverable.png',
    'select wf/new/scratch.png',
    'select wf/old/deliverable.png',
    'select wf/old/scratch.png',
  ])
})

it('never lets both chips be off', async () => {
  listing.files = roleFiles()
  await renderGallery('wf/run-1/deliverable.png')

  screen.getByRole('button', { name: 'Final' }).click()

  await waitFor(() =>
    expect(
      screen
        .getByRole('button', { name: 'Intermediate' })
        .getAttribute('aria-pressed'),
    ).toBe('true'),
  )
  expect(
    screen.getByRole('button', { name: 'Final' }).getAttribute('aria-pressed'),
  ).toBe('false')
  expect(screen.getByLabelText('select wf/run-1/scratch.png')).toBeTruthy()
})

it('has no subfolder filter in the top bar', async () => {
  listing.files = roleFiles()
  await renderGallery('wf/run-1/deliverable.png')
  expect(screen.queryByRole('combobox', { name: 'subfolder' })).toBeNull()
})

it('intersects the role chips with the text filter', async () => {
  listing.files = [
    file('wf/run-1/a.png', 'final', 'final'),
    file('wf/run-1/b.png', 'final', 'final'),
  ]
  await renderGallery('wf/run-1/a.png')
  const filter = screen.getByPlaceholderText('filter…') as HTMLInputElement
  filter.value = '/a.'
  filter.dispatchEvent(new Event('input', { bubbles: true }))
  await waitFor(() =>
    expect(screen.queryByLabelText('select wf/run-1/b.png')).toBeNull(),
  )
  expect(screen.getByLabelText('select wf/run-1/a.png')).toBeTruthy()
})

// The confirm dialog answers Escape itself, so the page must not also take
// the detail away underneath it. The guard used to name only [role=dialog],
// which is not what ConfirmDialog renders
it('leaves the detail open when Escape answers a confirm dialog', async () => {
  await renderGallery()
  screen.getAllByTitle(/show details/)[0].click()
  await waitFor(() =>
    expect(
      screen.getByLabelText('delete this file from the output directory'),
    ).toBeTruthy(),
  )

  screen.getByLabelText('delete this file from the output directory').click()
  await waitFor(() => expect(screen.getByRole('alertdialog')).toBeTruthy())

  pressEscape()

  await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
  expect(
    screen.getByLabelText('delete this file from the output directory'),
  ).toBeTruthy()
})

it('an Escape that closes a confirm leaves the selection; the next clears it', async () => {
  await renderGallery()
  checkbox('a.png').click()
  await waitFor(() => expect(screen.getByText('1 selected')).toBeTruthy())
  screen.getByRole('button', { name: /^delete the selected/i }).click()
  await waitFor(() => expect(screen.getByRole('alertdialog')).toBeTruthy())

  pressEscape()
  await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
  expect(screen.getByText('1 selected')).toBeTruthy()

  pressEscape()
  await waitFor(() => expect(screen.queryByText('1 selected')).toBeNull())
})

it('marks each file with the version of the run that wrote it', async () => {
  // Two runs of one workflow write the same basename - the case where the
  // label alone tells a person nothing about which is which
  listing.files = [
    { ...file('acorn/r1/film.mp4'), label: 'film.mp4', version: 1 },
    { ...file('acorn/r2/film.mp4'), label: 'film.mp4', version: 4 },
  ]
  render(GalleryPage)
  await waitFor(() => expect(screen.getAllByText('film.mp4')).toHaveLength(2))
  expect(screen.getByText('v1')).toBeTruthy()
  expect(screen.getByText('v4')).toBeTruthy()
  // One space between chip and label, so a screen reader does not run
  // 'v4' into the file name
  const caption = screen.getByText('v4').closest('.caption') as HTMLElement
  expect(caption.textContent?.replace(/\s+/g, ' ').trim()).toBe('v4 film.mp4')
})

it('shows no version for a flat-layout file, which belongs to no run', async () => {
  // The flat layout has no runs to number, and a card must not read
  // 'vnull' or 'vundefined' because of it
  listing.files = [{ ...file('ltx/flat.png'), label: 'flat.png' }]
  render(GalleryPage)
  await waitFor(() => expect(screen.getByText('flat.png')).toBeTruthy())
  expect(screen.queryByText(/^v\S+$/)).toBeNull()
})

it('names the run and its version in the details of the selected file', async () => {
  // "Look at version 4" ends here: the pane says which run it reached, so
  // the number in the grid can be checked against the one quoted
  listing.files = [
    {
      ...file('acorn/r2/film.mp4'),
      label: 'film.mp4',
      run_id: 'r2',
      version: 4,
    },
  ]
  render(GalleryPage)
  await waitFor(() => expect(screen.getByText('film.mp4')).toBeTruthy())
  await fireEvent.click(screen.getByText('film.mp4'))
  const detail = document.querySelector('.detail') as HTMLElement
  // One span holding both forms, so the query is over its whole text
  expect(detail.textContent).toContain('version 4')
  expect(detail.querySelector('code')?.textContent).toBe('r2')
})

// A <video> per tile held a connection each and hung a gallery of a few
// hundred clips; the grid shows a video as its server-made first frame
it('shows a video as a lazy thumbnail, never a <video>', async () => {
  listing.files = [{ ...file('demo/clip.mp4'), kind: 'video' }]
  const { container } = render(GalleryPage)

  await waitFor(() =>
    expect(
      container.querySelector('img[src="/thumb/demo/clip.mp4"]'),
    ).not.toBeNull(),
  )
  const img = container.querySelector('img[src="/thumb/demo/clip.mp4"]')!
  expect(img.getAttribute('loading')).toBe('lazy')
  expect(container.querySelector('video')).toBeNull()
  expect(container.querySelector('.videobadge')).not.toBeNull()
})

it('marks a tile that carries a note, titled with its first line', async () => {
  listing.files = [
    { ...file('a.png'), note: 'first line\nsecond' },
    file('b.png'),
  ]
  await renderGallery()
  const marks = document.querySelectorAll('.note-mark')
  expect(marks).toHaveLength(1)
  expect(marks[0].getAttribute('title')).toBe('first line')
})

it('marks every tile of a run after its run note is saved', async () => {
  const inRun = (name: string, note?: string): GalleryFile => ({
    ...file(`wf/r1/${name}`),
    folder: 'wf',
    run_id: 'r1',
    note,
  })
  listing.files = [inRun('a.png'), inRun('b.png')]
  galleryMetadata.mockResolvedValue({
    metadata: {},
    job: null,
    inputs: [],
    role: null,
    note: 'old',
    run_note: 'old',
    note_source: 'run',
  })
  await renderGallery('wf/r1/a.png')
  screen
    .getByLabelText('select wf/r1/a.png')
    .closest('.cellwrap')!
    .querySelector<HTMLButtonElement>('button.cell')!
    .click()
  await fireEvent.click(
    await screen.findByRole('button', { name: /edit note/i }),
  )
  setNote.mockImplementation(async () => {
    listing.files = [inRun('a.png', 'fresh'), inRun('b.png', 'fresh')]
    return { target: 'wf/r1', note: 'fresh', note_source: 'run' }
  })
  await fireEvent.input(screen.getByLabelText('Run note'), {
    target: { value: 'fresh' },
  })
  await fireEvent.click(screen.getByRole('button', { name: /save/i }))
  await waitFor(() =>
    expect(document.querySelectorAll('.note-mark')).toHaveLength(2),
  )
  expect(setNote).toHaveBeenCalledWith('wf/r1', 'fresh')
})

it('the arrow keys step through the files the grid shows', async () => {
  await renderGallery()
  await openFile('a.png')
  await waitFor(() => expect(galleryMetadata).toHaveBeenLastCalledWith('a.png'))
  await fireEvent.keyDown(window, { key: 'ArrowRight' })
  expect(galleryMetadata).toHaveBeenLastCalledWith('b.png')
  await fireEvent.keyDown(window, { key: 'ArrowLeft' })
  expect(galleryMetadata).toHaveBeenLastCalledWith('a.png')
})

it('an arrow key typed in the note box does not move the selection', async () => {
  await renderGallery()
  await openFile('a.png')
  await waitFor(() => expect(galleryMetadata).toHaveBeenLastCalledWith('a.png'))
  const before = galleryMetadata.mock.calls.length
  const box = document.createElement('textarea')
  document.body.append(box)
  await fireEvent.keyDown(box, { key: 'ArrowRight' })
  expect(galleryMetadata.mock.calls.length).toBe(before)
  box.remove()
})

it('Delete removes the open file and opens the next', async () => {
  await renderGallery()
  await openFile('a.png')
  await fireEvent.keyDown(window, { key: 'Delete' })
  await answerConfirm(true)
  await waitFor(() => expect(deleteOutput).toHaveBeenCalledWith('a.png'))
  await waitFor(() => expect(galleryMetadata).toHaveBeenLastCalledWith('b.png'))
})

it('stars the open file and can filter to starred files', async () => {
  await renderGallery()
  await openFile('a.png')
  await fireEvent.click(screen.getByRole('button', { name: 'star this file' }))
  expect(setStarred).toHaveBeenCalledWith('a.png', true)
  await fireEvent.click(screen.getByRole('button', { name: 'Starred' }))
  expect(screen.queryByTitle('b.png')).toBeNull()
})

it('the S key stars the open file', async () => {
  await renderGallery()
  await openFile('a.png')
  await fireEvent.keyDown(window, { key: 's' })
  await waitFor(() => expect(setStarred).toHaveBeenCalledWith('a.png', true))
})
