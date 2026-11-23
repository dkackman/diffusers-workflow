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
import { feed } from '../test/feedMock'

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
const listRequests = vi.hoisted(() =>
  vi.fn(() =>
    Promise.resolve({
      workspace: 'default',
      requests: [{ id: 'r1', title: 'Spring campaign' }] as unknown[],
    }),
  ),
)
const gallery = vi.hoisted(() =>
  vi.fn<
    (
      roles?: string[],
      request?: string,
      item?: string,
    ) => Promise<{ files: GalleryFile[] }>
  >(
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
    gallery: (roles?: string[], request?: string, item?: string) =>
      gallery(roles, request, item),
    listRequests: () => listRequests(),
    galleryThumbnailUrl: (name: string) => `/thumb/${name}`,
    galleryUrl: (name: string) => `/outputs/${name}`,
    outputDownloadUrl: (name: string) => `/download/${name}`,
    deleteOutput: (name: string) => deleteOutput(name),
    setStarred: (name: string, starred: boolean) => setStarred(name, starred),
    archiveOutputs: (names: string[]) => archiveOutputs(names),
    galleryMetadata: (name: string) => galleryMetadata(name),
    assess: () => new Promise(() => {}),
  },
}))

vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))

const notifyError = vi.hoisted(() => vi.fn())
const notifyFailure = vi.hoisted(() => vi.fn())
const notifyUndo = vi.hoisted(() => vi.fn())
vi.mock('../toast', () => ({
  notify: {
    error: notifyError,
    failure: notifyFailure,
    undo: notifyUndo,
    success: vi.fn(),
  },
}))

beforeEach(() => {
  listing.files = ['a.png', 'b.png', 'demo/c.png'].map((name) => file(name))
})
afterEach(() => {
  // Without this each render's DOM stays behind and the next test's
  // queries find two of everything
  cleanup()
  gallery.mockClear()
  location.hash = '#/ws/default/gallery'
  deleteOutput.mockClear()
  setStarred.mockClear()
  archiveOutputs.mockClear()
  notifyError.mockClear()
  notifyFailure.mockClear()
  notifyUndo.mockClear()
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

const settle = async () => {
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 10))
}

it('lists once on mount, and again when the gallery moves, with no blanking', async () => {
  await renderGallery()
  await settle()
  expect(gallery).toHaveBeenCalledTimes(1)
  // the page's gallery, and the request picker's requests
  expect(feed.watched(DEFAULT_WORKSPACE).sort()).toEqual([
    'gallery',
    'requests',
  ])

  listing.files = [...listing.files, file('new.png')]
  feed.move(DEFAULT_WORKSPACE, 'gallery')
  // the old grid stays while the answer is awaited
  expect(screen.getByLabelText('select a.png')).toBeTruthy()
  await waitFor(() => expect(screen.getByLabelText('select new.png')))
  expect(gallery).toHaveBeenCalledTimes(2)
  await settle()
  expect(gallery).toHaveBeenCalledTimes(2)
})

it('reloads nothing for another kind, or the gallery of another workspace', async () => {
  await renderGallery()
  await settle()
  feed.move(DEFAULT_WORKSPACE, 'jobs')
  feed.move('elsewhere', 'gallery')
  await settle()
  expect(gallery).toHaveBeenCalledTimes(1)
})

it('a quiet reload keeps the ticked files and the files when it fails', async () => {
  await renderGallery()
  checkbox('a.png').click()
  await waitFor(() => expect(screen.getByText('1 selected')).toBeTruthy())
  // a.png is gone from the answer: a quiet reload does not drop the tick
  listing.files = [file('b.png'), file('demo/c.png')]
  feed.move(DEFAULT_WORKSPACE, 'gallery')
  await waitFor(() =>
    expect(screen.queryByLabelText('select a.png')).toBeNull(),
  )
  expect(gallery).toHaveBeenCalledTimes(2)
  gallery.mockImplementationOnce(() => Promise.reject(new Error('down')))
  feed.move(DEFAULT_WORKSPACE, 'gallery')
  await settle()
  expect(screen.getByLabelText('select b.png')).toBeTruthy()
  expect(screen.queryByText(/Could not read the gallery/)).toBeNull()
})

it('a new output landing leaves the open file and the note being typed', async () => {
  galleryMetadata.mockResolvedValue({
    metadata: {},
    job: null,
    inputs: [],
    role: null,
    note: null,
    run_note: null,
    note_source: null,
  })
  await renderGallery()
  await openFile('a.png')
  await fireEvent.click(await screen.findByRole('button', { name: /note/i }))
  const box = (await screen.findByRole('textbox', {
    name: /note/i,
  })) as HTMLTextAreaElement
  await fireEvent.input(box, { target: { value: 'half a thought' } })

  listing.files = [...listing.files, file('new.png')]
  feed.move(DEFAULT_WORKSPACE, 'gallery')
  await waitFor(() => expect(screen.getByLabelText('select new.png')))
  const still = screen.getByRole('textbox', {
    name: /note/i,
  }) as HTMLTextAreaElement
  expect(still).toBe(box)
  expect(still.value).toBe('half a thought')
})

it('a reload begun before a delete does not bring the file back', async () => {
  await renderGallery()
  await openFile('a.png')
  let land: (reply: { files: GalleryFile[] }) => void = () => {}
  const stale = listing.files
  gallery.mockImplementationOnce(
    () => new Promise<{ files: GalleryFile[] }>((resolve) => (land = resolve)),
  )
  feed.move(DEFAULT_WORKSPACE, 'gallery')
  await fireEvent.keyDown(window, { key: 'Delete' })
  await answerConfirm(true)
  await waitFor(() =>
    expect(screen.queryByLabelText('select a.png')).toBeNull(),
  )
  land({ files: stale })
  await settle()
  expect(screen.queryByLabelText('select a.png')).toBeNull()
})

/** Whether a tile shows the star mark */
const starMark = (name: string) =>
  within(screen.getByTitle(name)).queryByLabelText('Starred')

it('a listing that began after a star shows an unstar made elsewhere', async () => {
  await renderGallery()
  await openFile('a.png')
  await fireEvent.click(screen.getByRole('button', { name: 'star this file' }))
  await waitFor(() => expect(starMark('a.png')).toBeTruthy())
  // The listing still says unstarred: someone unstarred it since
  feed.move(DEFAULT_WORKSPACE, 'gallery')
  await settle()
  expect(gallery).toHaveBeenCalledTimes(2)
  expect(starMark('a.png')).toBeNull()
})

it('a listing that began before a star cannot undo it', async () => {
  await renderGallery()
  await openFile('a.png')
  let land: (reply: { files: GalleryFile[] }) => void = () => {}
  gallery.mockImplementationOnce(
    () => new Promise<{ files: GalleryFile[] }>((resolve) => (land = resolve)),
  )
  feed.move(DEFAULT_WORKSPACE, 'gallery')
  await fireEvent.click(screen.getByRole('button', { name: 'star this file' }))
  await waitFor(() => expect(starMark('a.png')).toBeTruthy())
  land({ files: listing.files })
  await settle()
  expect(starMark('a.png')).toBeTruthy()
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

it('a double click on Download sends one request', async () => {
  await renderGallery()

  checkbox('a.png').click()
  await waitFor(() => expect(screen.getByText('1 selected')).toBeTruthy())
  const download = screen.getByRole('button', { name: /download/i })
  // Both clicks land before the page renders the first one's busy state
  download.click()
  download.click()

  await waitFor(() => expect(archiveOutputs).toHaveBeenCalled())
  await new Promise((r) => setTimeout(r, 20))
  expect(archiveOutputs).toHaveBeenCalledTimes(1)
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
  // Why, through describeError's copy, with the names in Copy details
  const busy = deleteOutput.mock.results[1].value
  expect(notifyFailure).toHaveBeenCalledWith(
    'delete 1 of 2 files',
    await busy.catch((e: unknown) => e),
    undefined,
    'b.png: Something went wrong',
  )
  expect(notifyError).not.toHaveBeenCalled()
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
  // "Look at v4" ends here: the pane says which job it reached, so
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
  expect(detail.textContent).toContain('v4')
  expect(
    detail.querySelector('[title^="4th job"]')?.getAttribute('title'),
  ).toBe('4th job of this workflow in this folder')
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
  await fireEvent.click(screen.getByRole('button', { name: /^save$/i }))
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

/** Whether a tile carries the changed mark */
const marked = (name: string) =>
  !!screen
    .getByTitle(name)
    .closest('.cellwrap')
    ?.classList.contains('row-changed')

it("the user's own star is not marked as another writer's change", async () => {
  await renderGallery()
  await openFile('a.png')
  // a listing begun before the star, landing after it
  let land: (reply: { files: GalleryFile[] }) => void = () => {}
  gallery.mockImplementationOnce(
    () => new Promise<{ files: GalleryFile[] }>((resolve) => (land = resolve)),
  )
  feed.move(DEFAULT_WORKSPACE, 'gallery')
  await fireEvent.click(screen.getByRole('button', { name: 'star this file' }))
  await waitFor(() => expect(starMark('a.png')).toBeTruthy())
  land({ files: listing.files })
  await settle()
  expect(marked('a.png')).toBe(false)
  // the star's own move of the feed: the server now says starred, and
  // someone else retitled b
  listing.files = listing.files.map((f) =>
    f.name === 'a.png'
      ? { ...f, starred: true }
      : f.name === 'b.png'
        ? { ...f, label: 'b, retitled' }
        : f,
  )
  feed.move(DEFAULT_WORKSPACE, 'gallery')
  await waitFor(() => expect(marked('b.png')).toBe(true))
  expect(marked('a.png')).toBe(false)
  expect(starMark('a.png')).toBeTruthy()
})

it('stars the open file and can filter to starred files', async () => {
  await renderGallery()
  await openFile('a.png')
  await fireEvent.click(screen.getByRole('button', { name: 'star this file' }))
  expect(setStarred).toHaveBeenCalledWith('a.png', true)
  await fireEvent.click(screen.getByRole('button', { name: 'Starred' }))
  expect(screen.queryByTitle('b.png')).toBeNull()
  expect(screen.getByTitle('a.png')).toBeTruthy()
})

it('starring offers Undo, which unstars', async () => {
  listing.files = [file('demo/c.png'), file('a.png')]
  await renderGallery('demo/c.png')
  await openFile('demo/c.png')
  await fireEvent.click(screen.getByRole('button', { name: 'star this file' }))
  await waitFor(() => expect(notifyUndo).toHaveBeenCalledTimes(1))
  expect(notifyUndo.mock.calls[0][0]).toBe('Starred c.png')
  await waitFor(() => expect(starMark('demo/c.png')).toBeTruthy())
  notifyUndo.mock.calls[0][1]()
  await waitFor(() =>
    expect(setStarred).toHaveBeenLastCalledWith('demo/c.png', false),
  )
  // Undone here at once, as the star was: no listing has to land first
  await waitFor(() => expect(starMark('demo/c.png')).toBeNull())
  expect(screen.getByRole('button', { name: 'star this file' })).toBeTruthy()
})

it('an undo that fails is reported, not thrown', async () => {
  listing.files = [{ ...file('a.png'), starred: true }, file('b.png')]
  await renderGallery()
  await openFile('a.png')
  await fireEvent.click(
    screen.getByRole('button', { name: 'unstar this file' }),
  )
  await waitFor(() => expect(notifyUndo).toHaveBeenCalledTimes(1))
  expect(notifyUndo.mock.calls[0][0]).toBe('Unstarred a.png')
  // The file was deleted elsewhere before Undo was pressed
  const gone = new Error('no output a.png')
  setStarred.mockImplementationOnce(() => Promise.reject(gone))
  // An unhandled rejection fails the vitest run, so reaching the end of
  // this test with the failure reported is the whole claim
  notifyUndo.mock.calls[0][1]()
  await waitFor(() => expect(notifyFailure).toHaveBeenCalled())
  expect(notifyFailure.mock.calls[0][0]).toBe('undo that')
  expect(notifyFailure.mock.calls[0][1]).toBe(gone)
  await settle()
  expect(starMark('a.png')).toBeNull()
})

it('the S key stars the open file', async () => {
  await renderGallery()
  await openFile('a.png')
  await fireEvent.keyDown(window, { key: 's' })
  await waitFor(() => expect(setStarred).toHaveBeenCalledWith('a.png', true))
})

it('the S key leaves a bare s to the page: Shift-S does nothing', async () => {
  await renderGallery()
  await openFile('a.png')
  await fireEvent.keyDown(window, { key: 'S', shiftKey: true })
  expect(setStarred).not.toHaveBeenCalled()
})

it('unstarring the open file under Starred steps on from its place', async () => {
  listing.files = [
    { ...file('a.png'), starred: true },
    { ...file('b.png'), starred: true },
    file('c.png'),
  ]
  await renderGallery()
  await fireEvent.click(screen.getByRole('button', { name: 'Starred' }))
  await openFile('a.png')
  await waitFor(() => expect(galleryMetadata).toHaveBeenLastCalledWith('a.png'))
  await fireEvent.keyDown(window, { key: 's' })
  await waitFor(() => expect(setStarred).toHaveBeenCalledWith('a.png', false))
  await waitFor(() => expect(screen.queryByTitle('a.png')).toBeNull())
  await fireEvent.keyDown(window, { key: 'ArrowRight' })
  expect(galleryMetadata).toHaveBeenLastCalledWith('b.png')
})

it("shows the open file's level findings from its metadata", async () => {
  galleryMetadata.mockResolvedValue({
    metadata: {},
    job: null,
    inputs: [],
    role: null,
    findings: [
      {
        rule: 'near_silent',
        severity: 'warn',
        at: null,
        value: -62,
        threshold: -50,
        says: 'near-silent for a track meant to be heard',
      },
    ],
  })
  await renderGallery()
  await openFile('a.png')
  const region = await screen.findByRole('region', { name: 'Findings' })
  await waitFor(() => expect(region.textContent).toContain('near-silent'))
})

it('with req in the hash, asks for that request and shows a removable chip', async () => {
  location.hash = '#/ws/default/gallery?req=r1'
  await renderGallery()
  expect(gallery).toHaveBeenLastCalledWith(undefined, 'r1', undefined)
  const chip = screen.getByTestId('req-filter')
  expect(chip.querySelector('a')?.getAttribute('href')).toBe(
    '#/ws/default/requests/r1',
  )
  await fireEvent.click(screen.getByLabelText('remove the request filter'))
  await waitFor(() =>
    expect(gallery).toHaveBeenLastCalledWith(undefined, undefined, undefined),
  )
  expect(screen.queryByTestId('req-filter')).toBeNull()
})

it('with req=<id>/<item> in the hash, keeps the item and asks for it', async () => {
  location.hash = '#/ws/default/gallery?req=r1/alpha'
  await renderGallery()
  expect(gallery).toHaveBeenLastCalledWith(undefined, 'r1', 'alpha')
  const chip = screen.getByTestId('req-filter')
  expect(chip.textContent).toContain('r1 › alpha')
  await Promise.resolve()
  expect(location.hash).toBe('#/ws/default/gallery?req=r1%2Falpha')
  await fireEvent.click(screen.getByLabelText('remove the request filter'))
  await waitFor(() =>
    expect(gallery).toHaveBeenLastCalledWith(undefined, undefined, undefined),
  )
  expect(screen.queryByTestId('req-filter')).toBeNull()
})

it('says the filter matched nothing, not that nothing was generated', async () => {
  location.hash = '#/ws/default/gallery?req=r1/alpha'
  listing.files = []
  render(GalleryPage)
  expect(
    await screen.findByText(/No outputs for this request item yet/),
  ).toBeTruthy()
  expect(screen.queryByText(/Nothing generated yet/)).toBeNull()
})

it('offers a request picker, and picking one reloads narrowed', async () => {
  await renderGallery()
  const picker = await screen.findByLabelText('filter by request')
  await fireEvent.change(picker, { target: { value: 'r1' } })
  await waitFor(() =>
    expect(gallery).toHaveBeenLastCalledWith(undefined, 'r1', undefined),
  )
  expect(screen.getByTestId('req-filter')).toBeTruthy()
})

it('hides the picker when the workspace has no requests', async () => {
  listRequests.mockResolvedValueOnce({ workspace: 'default', requests: [] })
  await renderGallery()
  await Promise.resolve()
  expect(screen.queryByLabelText('filter by request')).toBeNull()
})

it('drops an unfiltered listing that lands after the filtered one', async () => {
  let land: (reply: { files: GalleryFile[] }) => void = () => {}
  gallery.mockImplementationOnce(
    () => new Promise<{ files: GalleryFile[] }>((resolve) => (land = resolve)),
  )
  render(GalleryPage)
  const everything = listing.files
  listing.files = [file('only.png')]
  const picker = await screen.findByLabelText('filter by request')
  await fireEvent.change(picker, { target: { value: 'r1' } })
  await waitFor(() => expect(screen.getByLabelText('select only.png')))
  land({ files: everything })
  await new Promise((r) => setTimeout(r, 10))
  expect(screen.getByTestId('req-filter')).toBeTruthy()
  expect(screen.queryByLabelText('select a.png')).toBeNull()
  expect(screen.getByLabelText('select only.png')).toBeTruthy()
})

it('deleting a selection with starred files says how many and can spare them', async () => {
  listing.files = [file('a.png'), { ...file('b.png'), starred: true }]
  await renderGallery()
  checkbox('a.png').click()
  checkbox('b.png').click()
  await waitFor(() => expect(screen.getByText('2 selected')).toBeTruthy())
  screen.getByRole('button', { name: /^delete/i }).click()
  const dialog = await waitFor(() => screen.getByRole('alertdialog'))
  expect(dialog.textContent).toContain('Delete 2 files?')
  expect(dialog.textContent).toContain(
    'This removes them on disk. 1 of them is starred.',
  )
  within(dialog).getByRole('button', { name: 'Delete the 1 unstarred' }).click()
  await waitFor(() => expect(deleteOutput).toHaveBeenCalledTimes(1))
  expect(deleteOutput).toHaveBeenCalledWith('a.png')
})

it('shows a skeleton while loading, not the empty state', async () => {
  gallery.mockImplementationOnce(() => new Promise(() => {}))
  render(GalleryPage)
  expect(
    screen.getByRole('status', { name: 'loading the gallery' }),
  ).toBeTruthy()
  expect(screen.queryByText(/Nothing generated yet/)).toBeNull()
})

it('a failed first load offers Retry and Retry loads', async () => {
  gallery.mockImplementationOnce(() => Promise.reject(new Error('boom')))
  render(GalleryPage)
  await screen.findByText(/Couldn't load the gallery/)
  await fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await waitFor(() =>
    expect(screen.getByLabelText('select a.png')).toBeTruthy(),
  )
})

it('filtered to nothing offers Clear filters', async () => {
  await renderGallery()
  await fireEvent.input(screen.getByPlaceholderText('filter…'), {
    target: { value: 'zzz' },
  })
  expect(screen.getByText('Nothing matches those filters.')).toBeTruthy()
  await fireEvent.click(screen.getByRole('button', { name: 'Clear filters' }))
  await waitFor(() =>
    expect(screen.getByLabelText('select a.png')).toBeTruthy(),
  )
})

const DETAIL_DELETE = 'delete this file from the output directory'

it("a starred file's confirm says so", async () => {
  listing.files = [{ ...file('a.png'), starred: true }, file('b.png')]
  await renderGallery()
  await openFile('a.png')
  ;(await screen.findByRole('button', { name: DETAIL_DELETE })).click()
  const dialog = await waitFor(() => screen.getByRole('alertdialog'))
  expect(dialog.querySelector('.ui-title')?.textContent).toContain(
    'Delete a.png?',
  )
  expect(dialog.textContent).toContain(
    'This removes it on disk. It is starred.',
  )
  const yes = within(dialog).getByRole('button', { name: /^delete$/i })
  expect(yes.classList.contains('danger')).toBe(true)
  await answerConfirm(false)
  expect(deleteOutput).not.toHaveBeenCalled()
})

it('a double click deletes once', async () => {
  await renderGallery()
  await openFile('a.png')
  const button = await screen.findByRole('button', { name: DETAIL_DELETE })
  button.click()
  button.click()
  await answerConfirm(true)
  await waitFor(() => expect(deleteOutput).toHaveBeenCalledWith('a.png'))
  await settle()
  expect(deleteOutput).toHaveBeenCalledTimes(1)
  expect(screen.queryByRole('alertdialog')).toBeNull()
})
