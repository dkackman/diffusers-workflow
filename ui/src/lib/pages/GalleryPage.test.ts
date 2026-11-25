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
import { confirmState, resolveChoice } from '../confirm.svelte'
import { ApiError } from '../apiErrors'
import type { GalleryFile } from '../types'
import { DEFAULT_WORKSPACE, workspace } from '../workspace.svelte'
import { api } from '../test/apiMock'
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

const listing = vi.hoisted(() => ({
  files: [] as GalleryFile[],
  prompts: {} as Record<string, string>,
}))

const gallery = vi.hoisted(() =>
  vi.fn<
    (
      roles?: string[],
      request?: string,
      item?: string,
      opts?: { prompts?: boolean },
    ) => Promise<{ files: GalleryFile[]; prompts?: Record<string, string> }>
  >(
    (_roles, _request, _item, opts) =>
      new Promise((resolve) =>
        setTimeout(
          () =>
            resolve({
              files: listing.files,
              ...(opts?.prompts ? { prompts: listing.prompts } : {}),
            }),
          0,
        ),
      ),
  ),
)
// The background read for durations (`media: true`), kept apart so the
// listing's own call counts stay the listing's
const galleryMedia = vi.hoisted(() =>
  vi.fn<() => Promise<{ files: GalleryFile[] }>>(() =>
    Promise.resolve({ files: [] }),
  ),
)
const setNote = vi.hoisted(() => vi.fn())
// The page also passes the workspace; the assertions here are on the name
// alone, so the stub forwards only that
const galleryMetadata = vi.hoisted(() =>
  vi.fn<(name: string) => Promise<unknown>>(() => new Promise(() => {})),
)
// What the server holds as the note a save reads first (its etag read)
const noteHeld = vi.hoisted(() => ({ note: null as string | null }))
// Use as input's workflows: two of the same leaf in different folders
vi.mock('../api', async () =>
  (await import('../test/apiMock')).apiModule({
    setNote: (t: string, n: string | null) => setNote(t, n),
  }),
)
api.gallery.mockImplementation(
  (
    roles?: string[],
    request?: string,
    item?: string,
    opts?: { media?: boolean; prompts?: boolean },
  ) => (opts?.media ? galleryMedia() : gallery(roles, request, item, opts)),
)
api.listRequests.mockImplementation(() =>
  Promise.resolve({
    workspace: 'default',
    requests: [{ id: 'r1', title: 'Spring campaign' }] as unknown[],
  }),
)
api.outputDownloadUrl.mockImplementation((name: string) => `/download/${name}`)
api.deleteOutput.mockImplementation(() => Promise.resolve())
api.setStarred.mockImplementation((name: string, starred: boolean) =>
  Promise.resolve({ name, starred }),
)
api.archiveOutputs.mockImplementation(() => Promise.resolve())
api.galleryMetadata.mockImplementation((name: string) => galleryMetadata(name))
api.noteForSave.mockImplementation((target: string) =>
  Promise.resolve({ target, note: noteHeld.note, etag: '"e"' }),
)
api.assess.mockImplementation(() => new Promise(() => {}))
api.listAssets.mockImplementation(() =>
  Promise.resolve({ assets: [{ name: 'a.png' }] }),
)
// Use as input's workflows: two of the same leaf in different folders
api.listWorkflows.mockImplementation(() =>
  Promise.resolve({
    workflows: ['people/edit', 'pets/edit'],
    details: {
      'people/edit': { variable_names: ['image'], writable: true },
      'pets/edit': { variable_names: ['image'], writable: true },
    },
  }),
)
api.getJob.mockImplementation(() => new Promise(() => {}))

vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))

const notifySuccess = vi.hoisted(() => vi.fn())
const notifyError = vi.hoisted(() => vi.fn())
const notifyFailure = vi.hoisted(() => vi.fn())
const notifyUndo = vi.hoisted(() => vi.fn())
vi.mock('../toast', () => ({
  notify: {
    error: notifyError,
    failure: notifyFailure,
    undo: notifyUndo,
    success: notifySuccess,
  },
}))

beforeEach(() => {
  noteHeld.note = null
  listing.files = ['a.png', 'b.png', 'demo/c.png'].map((name) => file(name))
  listing.prompts = {}
})
afterEach(() => {
  // Without this each render's DOM stays behind and the next test's
  // queries find two of everything
  cleanup()
  gallery.mockClear()
  galleryMedia.mockReset()
  galleryMedia.mockImplementation(() => Promise.resolve({ files: [] }))
  location.hash = '#/ws/default/gallery'
  api.deleteOutput.mockReset()
  api.deleteOutput.mockImplementation(() => Promise.resolve())
  api.setStarred.mockClear()
  api.archiveOutputs.mockClear()
  notifyError.mockClear()
  notifyFailure.mockClear()
  notifyUndo.mockClear()
  notifySuccess.mockClear()
  api.keepOutput.mockReset()
  api.listAssets.mockClear()
  api.assess.mockReset()
  api.assess.mockImplementation(() => new Promise(() => {}))
  while (confirmState.open) resolveChoice(null)
  galleryMetadata.mockReset()
  galleryMetadata.mockImplementation(() => new Promise(() => {}))
  vi.unstubAllGlobals()
  workspace.current = DEFAULT_WORKSPACE
  localStorage.removeItem('dw-collapsed-gallery-folders')
})

const SEARCH = 'search names, prompts and notes…'
// What the listing asks for besides the filter: the runs' prompts
const PROMPTS = { prompts: true }

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

/** Opens a file in the viewer by clicking its tile, and waits for it */
async function openFile(name: string) {
  await fireEvent.click(
    screen.getByRole('button', {
      name: new RegExp(`^${name.replace('.', '\\.')}`),
    }),
  )
  return viewer(name)
}

/** The viewer, open on `name` */
const viewer = (name: string) =>
  screen.findByRole('dialog', { name: name.split('/').pop() })

/** A key pressed in the viewer: where focus is, once the sheet has it */
function press(init: KeyboardEventInit) {
  const sheet = screen.getByRole('dialog')
  const at = document.activeElement
  return fireEvent.keyDown(at && sheet.contains(at) ? at : sheet, init)
}

/** A tile's caption, which carries the file's name as its title */
const tile = (name: string) =>
  document.querySelector(`.tile-caption[title="${name}"]`)

async function answerConfirm(accept: boolean) {
  const dialog = await waitFor(() => screen.getByRole('alertdialog'))
  within(dialog)
    .getByRole('button', { name: accept ? /^delete$/i : /^cancel$/i })
    .click()
}

/** What the metadata route answers for a file with no recipe */
const metaReply = (over: Record<string, unknown> = {}) => ({
  name: 'a.png',
  source: 'output',
  metadata: null,
  job: null,
  run_id: '',
  version: null,
  media: null,
  role: null,
  inputs: [],
  findings: [],
  linked_jobs: [],
  note: null,
  run_note: null,
  note_source: null,
  ...over,
})

const settle = async () => {
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 10))
}

it('lists once on mount, and again when the gallery moves, with no blanking', async () => {
  await renderGallery()
  await settle()
  expect(gallery).toHaveBeenCalledTimes(1)
  // the page's gallery and its durations, the request picker's requests,
  // and Use as input's workflows
  expect(feed.watched(DEFAULT_WORKSPACE).sort()).toEqual([
    'gallery',
    'gallery',
    'requests',
    'workflows',
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
  await press({ key: 'Delete' })
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
  within(tile(name)!.closest('.cellwrap') as HTMLElement).queryByLabelText(
    'Starred',
  )

it('a listing that began after a star shows an unstar made elsewhere', async () => {
  await renderGallery()
  await openFile('a.png')
  await fireEvent.click(
    screen.getByRole('button', { name: 'Star', pressed: false }),
  )
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
  const before = listing.files
  await fireEvent.click(
    screen.getByRole('button', { name: 'Star', pressed: false }),
  )
  await waitFor(() => expect(starMark('a.png')).toBeTruthy())
  // The server carries the star from here on
  listing.files = listing.files.map((f) =>
    f.name === 'a.png' ? { ...f, starred: true } : f,
  )
  land({ files: before })
  await waitFor(() => expect(gallery).toHaveBeenCalledTimes(3))
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

  const filter = screen.getByPlaceholderText(SEARCH) as HTMLInputElement
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
    expect(api.archiveOutputs).toHaveBeenCalledWith(['a.png', 'b.png']),
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

  await waitFor(() => expect(api.archiveOutputs).toHaveBeenCalled())
  await new Promise((r) => setTimeout(r, 20))
  expect(api.archiveOutputs).toHaveBeenCalledTimes(1)
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
  expect(api.deleteOutput).toHaveBeenCalledTimes(2)
})

it('keeps a file that failed to delete selected and reports it', async () => {
  api.deleteOutput.mockImplementation((name: string) =>
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
  const busy = api.deleteOutput.mock.results[1].value
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
  expect(api.deleteOutput).not.toHaveBeenCalled()
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
        name: 'Delete',
      }),
    ).toBeTruthy(),
  )
  screen.getByRole('button', { name: 'Delete' }).click()
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
  const tag = document.querySelector('.filetile .role-tag') as HTMLElement
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
  const filter = screen.getByPlaceholderText(SEARCH) as HTMLInputElement
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
it('leaves the viewer open when Escape answers a confirm dialog', async () => {
  await renderGallery()
  await openFile('a.png')

  expect(screen.getByLabelText('Delete').classList.contains('danger')).toBe(
    true,
  )
  screen.getByLabelText('Delete').click()
  await waitFor(() => expect(screen.getByRole('alertdialog')).toBeTruthy())

  pressEscape()

  await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
  expect(await viewer('a.png')).toBeTruthy()
  expect(api.deleteOutput).not.toHaveBeenCalled()
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
    {
      ...file('acorn/r1/film.mp4'),
      run_id: 'r1',
      label: 'film.mp4',
      version: 1,
    },
    {
      ...file('acorn/r2/film.mp4'),
      run_id: 'r2',
      label: 'film.mp4',
      version: 4,
    },
  ]
  render(GalleryPage)
  await waitFor(() => expect(screen.getAllByText('film.mp4')).toHaveLength(2))
  expect(screen.getByText('v1')).toBeTruthy()
  expect(screen.getByText('v4')).toBeTruthy()
  // The version sits in the tile's corner row, its own element, so a
  // screen reader never runs 'v4' into the file name
  const v4 = screen.getByText('v4')
  expect(v4.closest('.tile-marks')).not.toBeNull()
  const caption = v4.closest('.filetile')!.querySelector('.tile-caption')!
  expect(caption.textContent?.trim()).toBe('film.mp4')
})

it('shows no version for a flat-layout file, which belongs to no run', async () => {
  // The flat layout has no runs to number, and a card must not read
  // 'vnull' or 'vundefined' because of it
  listing.files = [{ ...file('ltx/flat.png'), label: 'flat.png' }]
  render(GalleryPage)
  await waitFor(() => expect(screen.getByText('flat.png')).toBeTruthy())
  expect(screen.queryByText(/^v\S+$/)).toBeNull()
})

it('names the run and its version beside the open file', async () => {
  // "Look at v4" ends here: the aside says which job it reached, so
  // the number in the grid can be checked against the one quoted
  listing.files = [
    {
      ...file('acorn/r2/film.mp4'),
      label: 'film.mp4',
      run_id: 'r2',
      version: 4,
    },
  ]
  galleryMetadata.mockResolvedValue(metaReply())
  render(GalleryPage)
  await waitFor(() => expect(screen.getByText('film.mp4')).toBeTruthy())
  await fireEvent.click(screen.getByText('film.mp4'))
  const sheet = await viewer('film.mp4')
  const version = await within(sheet).findByTitle(
    '4th job of this workflow in this folder',
  )
  expect(version.textContent).toContain('v4')
  expect(version.querySelector('code')?.textContent).toBe('r2')
})

// A <video> per tile held a connection each and hung a gallery of a few
// hundred clips; the grid shows a video as its server-made first frame
it('shows a video as a lazy thumbnail, never a <video>', async () => {
  listing.files = [{ ...file('demo/clip.mp4'), kind: 'video' }]
  const { container } = render(GalleryPage)

  const thumb = 'img[src="/api/gallery/demo/clip.mp4/thumbnail"]'
  await waitFor(() => expect(container.querySelector(thumb)).not.toBeNull())
  const img = container.querySelector(thumb)!
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
  noteHeld.note = 'old'
  await renderGallery('wf/r1/a.png')
  screen
    .getByLabelText('select wf/r1/a.png')
    .closest('.cellwrap')!
    .querySelector<HTMLButtonElement>('button.filetile')!
    .click()
  await fireEvent.click(
    await screen.findByRole('button', { name: /edit note/i }),
  )
  setNote.mockImplementation(async () => {
    listing.files = [inRun('a.png', 'fresh'), inRun('b.png', 'fresh')]
    return { target: 'wf/r1', note: 'fresh', note_source: 'run' }
  })
  await fireEvent.input(screen.getByLabelText('Your note on this job'), {
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
  await press({ key: 'ArrowRight' })
  expect(await viewer('b.png')).toHaveTextContent('2 of 3')
  await waitFor(() => expect(galleryMetadata).toHaveBeenLastCalledWith('b.png'))
  await press({ key: 'ArrowLeft' })
  expect(await viewer('a.png')).toBeTruthy()
})

it('an arrow key typed in the note box does not move the viewer', async () => {
  galleryMetadata.mockResolvedValue(metaReply())
  await renderGallery()
  await openFile('a.png')
  await fireEvent.click(
    await screen.findByRole('button', { name: /edit note/i }),
  )
  const box = await screen.findByRole('textbox', { name: /note/i })
  await fireEvent.keyDown(box, { key: 'ArrowRight' })
  await fireEvent.keyDown(box, { key: 's' })
  expect(await viewer('a.png')).toBeTruthy()
  expect(api.setStarred).not.toHaveBeenCalled()
})

it('S stars the open file and Delete asks, then steps on', async () => {
  await renderGallery()
  await openFile('a.png')
  await press({ key: 's' })
  await waitFor(() =>
    expect(api.setStarred).toHaveBeenCalledWith('a.png', true),
  )
  await waitFor(() =>
    expect(
      screen.getByRole('button', { name: 'Star', pressed: true }),
    ).toHaveAttribute('aria-pressed', 'true'),
  )
  await press({ key: 'Delete' })
  await answerConfirm(true)
  await waitFor(() => expect(api.deleteOutput).toHaveBeenCalledWith('a.png'))
  expect(await viewer('b.png')).toHaveTextContent('1 of 2')
  await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
  // Cmd+Backspace deletes too
  await press({ key: 'Backspace', metaKey: true })
  await answerConfirm(true)
  await waitFor(() => expect(api.deleteOutput).toHaveBeenCalledWith('b.png'))
  expect(await viewer('demo/c.png')).toHaveTextContent('1 of 1')
})

it('a delete that fails leaves its file open in the viewer', async () => {
  api.deleteOutput.mockImplementation((n) =>
    n === 'b.png' ? Promise.reject(new Error('busy')) : Promise.resolve(),
  )
  await renderGallery()
  await openFile('a.png')
  await press({ key: 'Delete' })
  await answerConfirm(true)
  expect(await viewer('b.png')).toBeTruthy()
  await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
  await press({ key: 'Delete' })
  await answerConfirm(true)
  await waitFor(() => expect(notifyFailure).toHaveBeenCalled())
  await settle()
  expect(await viewer('b.png')).toBeTruthy()
})

/** Whether a tile carries the changed mark */
const marked = (name: string) =>
  !!tile(name)?.closest('.cellwrap')?.classList.contains('row-changed')

it("the user's own star is not marked as another writer's change", async () => {
  await renderGallery()
  await openFile('a.png')
  // a listing begun before the star, landing after it
  let land: (reply: { files: GalleryFile[] }) => void = () => {}
  gallery.mockImplementationOnce(
    () => new Promise<{ files: GalleryFile[] }>((resolve) => (land = resolve)),
  )
  feed.move(DEFAULT_WORKSPACE, 'gallery')
  const before = listing.files
  await fireEvent.click(
    screen.getByRole('button', { name: 'Star', pressed: false }),
  )
  await waitFor(() => expect(starMark('a.png')).toBeTruthy())
  // the server carries the star from here on
  listing.files = listing.files.map((f) =>
    f.name === 'a.png' ? { ...f, starred: true } : f,
  )
  land({ files: before })
  await waitFor(() => expect(gallery).toHaveBeenCalledTimes(3))
  await settle()
  expect(marked('a.png')).toBe(false)
  // the star's own move of the feed, and someone else retitled b
  listing.files = listing.files.map((f) =>
    f.name === 'b.png' ? { ...f, label: 'b, retitled' } : f,
  )
  feed.move(DEFAULT_WORKSPACE, 'gallery')
  await waitFor(() => expect(marked('b.png')).toBe(true))
  expect(marked('a.png')).toBe(false)
  expect(starMark('a.png')).toBeTruthy()
})

it('stars the open file and can filter to starred files', async () => {
  await renderGallery()
  await openFile('a.png')
  await fireEvent.click(
    screen.getByRole('button', { name: 'Star', pressed: false }),
  )
  expect(api.setStarred).toHaveBeenCalledWith('a.png', true)
  await waitFor(() => expect(starMark('a.png')).toBeTruthy())
  await fireEvent.click(screen.getByRole('button', { name: 'Close' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  await fireEvent.click(screen.getByRole('button', { name: 'Starred' }))
  expect(tile('b.png')).toBeNull()
  expect(tile('a.png')).toBeTruthy()
})

it('starring offers Undo, which unstars', async () => {
  listing.files = [file('demo/c.png'), file('a.png')]
  await renderGallery('demo/c.png')
  await openFile('demo/c.png')
  await fireEvent.click(
    screen.getByRole('button', { name: 'Star', pressed: false }),
  )
  await waitFor(() => expect(notifyUndo).toHaveBeenCalledTimes(1))
  expect(notifyUndo.mock.calls[0][0]).toBe('Starred c.png')
  await waitFor(() => expect(starMark('demo/c.png')).toBeTruthy())
  notifyUndo.mock.calls[0][1]()
  await waitFor(() =>
    expect(api.setStarred).toHaveBeenLastCalledWith('demo/c.png', false),
  )
  // Undone here at once, as the star was: no listing has to land first
  await waitFor(() => expect(starMark('demo/c.png')).toBeNull())
  expect(
    screen.getByRole('button', { name: 'Star', pressed: false }),
  ).toBeTruthy()
})

it("a star Undo leaves another writer's star", async () => {
  await renderGallery()
  await openFile('a.png')
  await fireEvent.click(
    screen.getByRole('button', { name: 'Star', pressed: false }),
  )
  await waitFor(() => expect(notifyUndo).toHaveBeenCalledTimes(1))
  await waitFor(() => expect(starMark('a.png')).toBeTruthy())
  // Unstarred elsewhere since: the listing that lands says so
  feed.move(DEFAULT_WORKSPACE, 'gallery')
  await waitFor(() => expect(starMark('a.png')).toBeNull())
  notifyUndo.mock.calls[0][1]()
  await waitFor(() =>
    expect(notifyError).toHaveBeenCalledWith(
      'Nothing undone: the star was changed since.',
    ),
  )
  expect(api.setStarred).toHaveBeenCalledTimes(1)
})

it('an undo that fails is reported, not thrown', async () => {
  listing.files = [{ ...file('a.png'), starred: true }, file('b.png')]
  await renderGallery()
  await openFile('a.png')
  await fireEvent.click(
    screen.getByRole('button', { name: 'Star', pressed: true }),
  )
  await waitFor(() => expect(notifyUndo).toHaveBeenCalledTimes(1))
  expect(notifyUndo.mock.calls[0][0]).toBe('Unstarred a.png')
  // The file was deleted elsewhere before Undo was pressed
  const gone = new Error('no output a.png')
  api.setStarred.mockImplementationOnce(() => Promise.reject(gone))
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
  await press({ key: 's' })
  await waitFor(() =>
    expect(api.setStarred).toHaveBeenCalledWith('a.png', true),
  )
})

it('the S key leaves a bare s to the page: Shift-S does nothing', async () => {
  await renderGallery()
  await openFile('a.png')
  await press({ key: 'S', shiftKey: true })
  expect(api.setStarred).not.toHaveBeenCalled()
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
  await press({ key: 's' })
  await waitFor(() =>
    expect(api.setStarred).toHaveBeenCalledWith('a.png', false),
  )
  await waitFor(() => expect(tile('a.png')).toBeNull())
  // the viewer goes on to what now fills its place
  expect(await viewer('b.png')).toHaveTextContent('1 of 1')
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
  expect(gallery).toHaveBeenLastCalledWith(undefined, 'r1', undefined, PROMPTS)
  const chip = screen.getByTestId('req-filter')
  expect(chip.querySelector('a')?.getAttribute('href')).toBe(
    '#/ws/default/requests/r1',
  )
  await fireEvent.click(screen.getByLabelText('remove the request filter'))
  await waitFor(() =>
    expect(gallery).toHaveBeenLastCalledWith(
      undefined,
      undefined,
      undefined,
      PROMPTS,
    ),
  )
  expect(screen.queryByTestId('req-filter')).toBeNull()
})

it('with req=<id>/<item> in the hash, keeps the item and asks for it', async () => {
  location.hash = '#/ws/default/gallery?req=r1/alpha'
  await renderGallery()
  expect(gallery).toHaveBeenLastCalledWith(undefined, 'r1', 'alpha', PROMPTS)
  const chip = screen.getByTestId('req-filter')
  expect(chip.textContent).toMatch(/request:\s*Spring campaign\s*› alpha/)
  await Promise.resolve()
  expect(location.hash).toBe('#/ws/default/gallery?req=r1%2Falpha')
  await fireEvent.click(screen.getByLabelText('remove the request filter'))
  await waitFor(() =>
    expect(gallery).toHaveBeenLastCalledWith(
      undefined,
      undefined,
      undefined,
      PROMPTS,
    ),
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
    expect(gallery).toHaveBeenLastCalledWith(
      undefined,
      'r1',
      undefined,
      PROMPTS,
    ),
  )
  expect(screen.getByTestId('req-filter')).toBeTruthy()
})

it('hides the picker when the workspace has no requests', async () => {
  api.listRequests.mockResolvedValueOnce({ workspace: 'default', requests: [] })
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
  await waitFor(() => expect(api.deleteOutput).toHaveBeenCalledTimes(1))
  expect(api.deleteOutput).toHaveBeenCalledWith('a.png')
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
  await fireEvent.input(screen.getByPlaceholderText(SEARCH), {
    target: { value: 'zzz' },
  })
  expect(screen.getByText('Nothing matches those filters.')).toBeTruthy()
  await fireEvent.click(screen.getByRole('button', { name: 'Clear filters' }))
  await waitFor(() =>
    expect(screen.getByLabelText('select a.png')).toBeTruthy(),
  )
})

const DETAIL_DELETE = 'Delete'

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
  expect(api.deleteOutput).not.toHaveBeenCalled()
})

it('a double click deletes once', async () => {
  await renderGallery()
  await openFile('a.png')
  const button = await screen.findByRole('button', { name: DETAIL_DELETE })
  button.click()
  button.click()
  await answerConfirm(true)
  await waitFor(() => expect(api.deleteOutput).toHaveBeenCalledWith('a.png'))
  await settle()
  expect(api.deleteOutput).toHaveBeenCalledTimes(1)
  expect(screen.queryByRole('alertdialog')).toBeNull()
})

it('a double press deletes the picked once', async () => {
  await renderGallery()
  checkbox('a.png').click()
  checkbox('b.png').click()
  await waitFor(() => expect(screen.getByText('2 selected')).toBeTruthy())
  const button = screen.getByRole('button', { name: /^delete the selected/i })
  button.click()
  button.click()
  await answerConfirm(true)
  await waitFor(() => expect(api.deleteOutput).toHaveBeenCalled())
  await settle()
  expect(api.deleteOutput.mock.calls.map((c) => c[0]).sort()).toEqual([
    'a.png',
    'b.png',
  ])
  expect(screen.queryByRole('alertdialog')).toBeNull()
})

const hashEvent = (change: () => void) =>
  new Promise<void>((resolve) => {
    window.addEventListener('hashchange', () => resolve(), { once: true })
    change()
  })

it('the text, role and Starred filters survive leaving and coming back', async () => {
  await hashEvent(() => (location.hash = '#/ws/default/gallery?x=1'))
  await renderGallery()
  const filter = screen.getByPlaceholderText(SEARCH) as HTMLInputElement
  await fireEvent.input(filter, { target: { value: 'c' } })
  await fireEvent.click(screen.getByRole('button', { name: 'Intermediate' }))
  await fireEvent.click(screen.getByRole('button', { name: 'Starred' }))
  expect(location.hash).toBe(
    '#/ws/default/gallery?x=1&q=c&roles=final,intermediate&starred=1',
  )
  // open something else, then press Back
  cleanup()
  await hashEvent(() => (location.hash = '#/ws/default/jobs'))
  await hashEvent(() => history.back())
  render(GalleryPage)
  const again = (await screen.findByPlaceholderText(SEARCH)) as HTMLInputElement
  expect(again.value).toBe('c')
  const pressed = (name: string) =>
    screen.getByRole('button', { name }).getAttribute('aria-pressed')
  expect(pressed('Intermediate')).toBe('true')
  expect(pressed('Starred')).toBe('true')
})

it('the grid renders no <video>, and fills durations from one background read', async () => {
  listing.files = [
    { ...file('a.png') },
    { ...file('demo/clip.mp4'), kind: 'video', label: 'clip.mp4' },
    { ...file('demo/tone.wav'), kind: 'audio', label: 'tone.wav' },
  ]
  galleryMedia.mockImplementation(() =>
    Promise.resolve({
      files: [
        { ...file('demo/clip.mp4'), kind: 'video', duration_seconds: 75 },
        { ...file('demo/tone.wav'), kind: 'audio', duration_seconds: 2.5 },
      ],
    }),
  )
  const { container } = render(GalleryPage)
  await waitFor(() =>
    expect(
      screen
        .getByTitle('demo/clip.mp4')
        .closest('.filetile')!
        .querySelector('.tile-time')?.textContent,
    ).toBe('1m 15s'),
  )
  await waitFor(() =>
    expect(
      screen.getByTitle('demo/tone.wav').closest('.filetile')!.textContent,
    ).toContain('2.5s'),
  )
  await settle()
  expect(container.querySelector('video')).toBeNull()
  expect(galleryMedia).toHaveBeenCalledTimes(1)
  expect(gallery).toHaveBeenCalledTimes(1)
})

it('tiles take their workspace from the listing', async () => {
  gallery.mockImplementationOnce(
    () =>
      new Promise((resolve) =>
        setTimeout(
          () => resolve({ files: listing.files, workspace: 'studio' } as never),
          0,
        ),
      ),
  )
  const { container } = render(GalleryPage)
  await waitFor(() => expect(container.querySelectorAll('img').length).toBe(3))
  for (const img of container.querySelectorAll('img'))
    expect(img.getAttribute('src')).toContain('workspace=studio')
})

// The viewer: opened from the grid or the link, closed by Back

const hashOpen = () =>
  new URLSearchParams(location.hash.split('?')[1]).get('open')

it('clicking a tile opens the viewer and writes open= to the hash', async () => {
  await renderGallery()
  const before = history.length
  const sheet = await openFile('demo/c.png')
  expect(sheet).toHaveTextContent('3 of 3')
  expect(hashOpen()).toBe('demo/c.png')
  // a pushed entry, so Back closes it
  expect(history.length).toBe(before + 1)
  await hashEvent(() => history.back())
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  expect(hashOpen()).toBeNull()
})

it('a hash with open= opens that file', async () => {
  location.hash = '#/ws/default/gallery?open=b.png'
  await renderGallery()
  expect(await viewer('b.png')).toHaveTextContent('2 of 3')
  // a deep link pushed nothing: Close replaces open= away
  await fireEvent.click(screen.getByRole('button', { name: 'Close' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  expect(location.hash).toBe('#/ws/default/gallery')
})

it('an open= naming a missing file says so', async () => {
  location.hash = '#/ws/default/gallery?open=gone%2Fold.png'
  await renderGallery()
  const notice = await screen.findByText(
    'old.png is not in this view any more.',
  )
  expect(notice.classList.contains('hint')).toBe(true)
  expect(screen.queryByRole('dialog')).toBeNull()
  await waitFor(() => expect(hashOpen()).toBeNull())
  await fireEvent.click(within(notice).getByRole('button', { name: 'Dismiss' }))
  expect(screen.queryByText(/is not in this view any more/)).toBeNull()
})

it('an open= naming a file in a collapsed folder opens it and the folder', async () => {
  await renderShutDemo()
  cleanup()
  location.hash = '#/ws/default/gallery?open=demo%2Fc.png'
  await renderGallery()
  expect(await viewer('demo/c.png')).toBeTruthy()
  expect(screen.queryByText(/is not in this view any more/)).toBeNull()
  await waitFor(() => expect(checkbox('demo/c.png')).toBeTruthy())
  // the deep-linked folder is aria-expanded
  expect(demoToggle()).toHaveAttribute('aria-expanded', 'true')
})

it('a stray open= said in one workspace is not said in the next', async () => {
  location.hash = '#/ws/default/gallery?open=gone%2Fold.png'
  await renderGallery()
  await screen.findByText('old.png is not in this view any more.')
  workspace.current = 'studio'
  await waitFor(() =>
    expect(screen.queryByText(/is not in this view any more/)).toBeNull(),
  )
  workspace.current = DEFAULT_WORKSPACE
})

it('a stray open= is unsaid when the file arrives', async () => {
  location.hash = '#/ws/default/gallery?open=late.png'
  await renderGallery()
  await screen.findByText('late.png is not in this view any more.')
  listing.files = [...listing.files, file('late.png')]
  feed.move(DEFAULT_WORKSPACE, 'gallery')
  await waitFor(() =>
    expect(screen.queryByText(/is not in this view any more/)).toBeNull(),
  )
  // the link was cleared when it was said: the viewer stays shut
  expect(screen.queryByRole('dialog')).toBeNull()
})

it('same-leaf tiles are told apart by their caption and name', async () => {
  listing.files = [file('a.png'), file('one/x.png'), file('two/x.png')]
  await renderGallery()
  expect(tile('one/x.png')).toHaveTextContent('one/x.png')
  expect(tile('two/x.png')).toHaveTextContent('two/x.png')
  // a leaf no other file shares keeps the listing's own label
  expect(tile('a.png')).toHaveTextContent(/^a$/)
})

const useAsInputFrom = async (more: boolean) => {
  await renderGallery()
  const sheet = await openFile('a.png')
  if (more) {
    await fireEvent.click(within(sheet).getByRole('button', { name: 'More' }))
    const menu = await screen.findByRole('dialog', { name: 'More' })
    await fireEvent.click(
      within(menu).getByRole('button', { name: /Use as input/ }),
    )
  } else
    await fireEvent.click(
      within(sheet).getByRole('button', { name: /Use as input/ }),
    )
  return screen.findByRole('dialog', { name: 'Use a.png as input' })
}

it("Use as input's choose path opens the workflow with the file filled in", async () => {
  const dialog = await useAsInputFrom(false)
  // the two same-leaf workflows are told apart
  await within(dialog).findByRole('button', { name: 'people/edit' })
  await fireEvent.click(
    within(dialog).getByRole('button', { name: 'pets/edit' }),
  )
  await waitFor(() =>
    expect(location.hash).toBe('#/ws/default/workflows/pets/edit'),
  )
  expect(
    JSON.parse(sessionStorage.getItem('dw-workflow-prefill') ?? 'null'),
  ).toEqual({ workflow: 'pets/edit', arguments: { image: 'output:a.png' } })
  sessionStorage.clear()
})

it("Use as input's dialog opens from More at a phone's width", async () => {
  const widthOf = (px: number) => {
    Object.defineProperty(window, 'innerWidth', {
      configurable: true,
      value: px,
    })
    window.dispatchEvent(new Event('resize'))
  }
  widthOf(390)
  try {
    const dialog = await useAsInputFrom(true)
    expect(
      await within(dialog).findByRole('button', { name: 'people/edit' }),
    ).toBeTruthy()
  } finally {
    widthOf(1024)
  }
})

it('the viewer steps on when its file is deleted elsewhere', async () => {
  await renderGallery()
  await openFile('a.png')
  listing.files = [file('b.png'), file('demo/c.png')]
  feed.move(DEFAULT_WORKSPACE, 'gallery')
  expect(await viewer('b.png')).toHaveTextContent('1 of 2')
  expect(hashOpen()).toBe('b.png')
  expect(screen.queryByText(/is not in this view any more/)).toBeNull()
  // the last one goes: the viewer closes
  listing.files = []
  feed.move(DEFAULT_WORKSPACE, 'gallery')
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
})

it('switching workspace closes the viewer', async () => {
  await renderGallery()
  await openFile('a.png')
  // the picker alone: the viewer does not wait for the hash to follow
  workspace.current = 'studio'
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
})

it('titles the star as starring the file, never keeping it', async () => {
  await renderGallery()
  await openFile('a.png')
  const star = screen.getByRole('button', { name: 'Star', pressed: false })
  expect(star.getAttribute('title')).toBe('Star this file (S)')
  expect(star.getAttribute('aria-pressed')).toBe('false')
})

it("offers the file's reference and a download beside it", async () => {
  await renderGallery()
  const sheet = await openFile('demo/c.png')
  expect(within(sheet).getByText('output:demo/c.png')).toBeTruthy()
  expect(
    within(sheet).getByRole('link', { name: 'Download' }).getAttribute('href'),
  ).toBe('/download/demo/c.png')
})

// Save as asset: the name dialog, and Replace or Keep both when it is taken

// keep_output's exists refusal: code 'exists' (dw/server/routes/assets.py)
const TAKEN = 'asset:a.png already exists - pass overwrite=true to replace it'

/** Press Save as asset and save under the name it offers */
async function saveAsAsset() {
  const button = screen.getByRole('button', { name: 'Save as asset' })
  expect(button.getAttribute('title')).toBe(
    'Save this file to the asset library, under a name later workflows can use',
  )
  await fireEvent.click(button)
  const dialog = await screen.findByRole('alertdialog', {
    name: 'Save as asset',
  })
  expect(
    (
      within(dialog).getByRole('textbox', {
        name: 'Name in the asset library',
      }) as HTMLInputElement
    ).value,
  ).toBe('a.png')
  await fireEvent.click(within(dialog).getByRole('button', { name: 'Save' }))
}
const takenQuestion = () =>
  waitFor(() =>
    screen.getByRole('alertdialog', {
      name: '"a.png" is already in the asset library.',
    }),
  )

it('offers Save as asset, never window.prompt, and lists nothing to save', async () => {
  const prompt = vi.spyOn(window, 'prompt')
  api.keepOutput.mockResolvedValueOnce({ reference: 'asset:a.png' })
  await renderGallery()
  await openFile('a.png')
  await saveAsAsset()
  await waitFor(() =>
    expect(notifySuccess).toHaveBeenCalledWith('Saved as asset a.png'),
  )
  expect(api.keepOutput).toHaveBeenCalledWith('a.png', 'a.png')
  expect(api.listAssets).not.toHaveBeenCalled()
  expect(prompt).not.toHaveBeenCalled()
})

it('cancelling the exists question on Save as asset shows no error', async () => {
  api.keepOutput.mockRejectedValueOnce(new ApiError(TAKEN, 409, 'exists'))
  await renderGallery()
  await openFile('a.png')
  await saveAsAsset()
  const dialog = await takenQuestion()
  expect(api.listAssets).toHaveBeenCalledTimes(1)
  await fireEvent.click(
    within(dialog).getByRole('button', { name: /^cancel$/i }),
  )
  await settle()
  expect(api.keepOutput).toHaveBeenCalledTimes(1)
  expect(notifyError).not.toHaveBeenCalled()
  expect(notifyFailure).not.toHaveBeenCalled()
  expect(notifySuccess).not.toHaveBeenCalled()
  // the viewer is still on the file
  expect(await viewer('a.png')).toBeTruthy()
})

it('Replace on Save as asset retries with overwrite, Keep both under a free name', async () => {
  api.keepOutput
    .mockRejectedValueOnce(new ApiError(TAKEN, 409, 'exists'))
    .mockResolvedValueOnce({ reference: 'asset:a.png' })
  await renderGallery()
  await openFile('a.png')
  await saveAsAsset()
  await fireEvent.click(
    within(await takenQuestion()).getByRole('button', { name: /^replace$/i }),
  )
  await waitFor(() => expect(api.keepOutput).toHaveBeenCalledTimes(2))
  expect(api.keepOutput).toHaveBeenLastCalledWith('a.png', 'a.png', true)
  await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())

  api.keepOutput
    .mockRejectedValueOnce(new ApiError(TAKEN, 409, 'exists'))
    .mockResolvedValueOnce({ reference: 'asset:a-2.png' })
  await saveAsAsset()
  await fireEvent.click(
    within(await takenQuestion()).getByRole('button', { name: 'Keep both' }),
  )
  await waitFor(() => expect(api.keepOutput).toHaveBeenCalledTimes(4))
  expect(api.keepOutput).toHaveBeenLastCalledWith('a.png', 'a-2.png')
})

it('a save refused for another reason shows the reason, not Replace', async () => {
  api.keepOutput.mockRejectedValueOnce(
    new ApiError('This workspace has no asset library', 409),
  )
  await renderGallery()
  await openFile('a.png')
  await saveAsAsset()
  await waitFor(() =>
    expect(notifyFailure).toHaveBeenCalledWith(
      'save it as an asset',
      expect.objectContaining({
        message: 'This workspace has no asset library',
      }),
    ),
  )
  expect(
    screen.queryByRole('alertdialog', { name: /already in the asset/ }),
  ).toBeNull()
  expect(api.keepOutput).toHaveBeenCalledTimes(1)
})

// Assess, and the findings beside the open file

const clipFiles = () => [
  { ...file('demo/clip.mp4'), kind: 'video' as const, label: 'clip.mp4' },
  file('a.png'),
]
const finding = (
  severity: 'warn' | 'info',
  says: string,
  at: number | null = null,
) => ({ rule: `${severity}_rule`, severity, at, value: 1, threshold: 0, says })
const assessReply = (findings: ReturnType<typeof finding>[]) => ({
  name: 'demo/clip.mp4',
  source: 'output',
  kind: 'video',
  findings,
  rules_applied: ['near_silent', 'shot_dead_air'],
  rules_skipped: [],
  not_applicable: { analyze_seams: 'no shot boundaries recorded' },
})

it("disables an image's Assess and says why", async () => {
  await renderGallery()
  await openFile('a.png')
  const button = screen.getByRole('button', { name: 'Assess' })
  expect((button as HTMLButtonElement).disabled).toBe(true)
  expect(button.getAttribute('title')).toBe(
    'a still has no shots, seams or soundtrack to measure',
  )
  expect(api.assess).not.toHaveBeenCalled()
})

it("Assess shows the reply's findings, their time and what did not apply", async () => {
  listing.files = clipFiles()
  galleryMetadata.mockResolvedValue(metaReply())
  api.assess.mockResolvedValue(
    assessReply([
      finding('info', 'picture jumps', 64.25),
      finding('warn', 'silent'),
    ]),
  )
  await renderGallery('demo/clip.mp4')
  await openFile('demo/clip.mp4')
  const region = await screen.findByRole('region', { name: 'Findings' })
  await fireEvent.click(screen.getByRole('button', { name: 'Assess' }))
  expect(api.assess).toHaveBeenCalledWith('demo/clip.mp4')
  await waitFor(() => expect(region.querySelectorAll('li')).toHaveLength(2))
  const rows = [...region.querySelectorAll('li')]
  expect(rows[0].textContent).toContain('silent')
  expect(rows[1].querySelector('.at')?.textContent).toBe('1:04.3')
  expect(region.textContent).toContain(
    'analyze_seams not applicable (no shot boundaries recorded)',
  )
})

it('shows Assessing… while the request runs', async () => {
  listing.files = clipFiles()
  await renderGallery('demo/clip.mp4')
  await openFile('demo/clip.mp4')
  await fireEvent.click(screen.getByRole('button', { name: 'Assess' }))
  const busy = await screen.findByRole('button', { name: /Assessing/ })
  expect((busy as HTMLButtonElement).disabled).toBe(true)
})

it('a failed Assess keeps the earlier list and says why in plain words', async () => {
  listing.files = clipFiles()
  galleryMetadata.mockResolvedValue(metaReply())
  api.assess.mockResolvedValueOnce(assessReply([finding('warn', 'silent')]))
  await renderGallery('demo/clip.mp4')
  await openFile('demo/clip.mp4')
  const region = await screen.findByRole('region', { name: 'Findings' })
  await fireEvent.click(screen.getByRole('button', { name: 'Assess' }))
  await waitFor(() => expect(region.textContent).toContain('silent'))
  const raw = 'ffprobe exited 1: moov atom not found in /runs/a/clip.mp4'
  api.assess.mockRejectedValueOnce(new ApiError(raw, 500))
  await fireEvent.click(screen.getByRole('button', { name: 'Assess' }))
  const alert = await within(region).findByRole('alert')
  expect(alert.querySelector('.title')?.textContent).toBe(
    "Couldn't assess this file: The server hit an error",
  )
  // the engine's words sit folded behind Details, not in the line
  expect(alert.querySelector('details')?.open).toBe(false)
  expect(alert.querySelector('details pre')?.textContent).toBe(raw)
  expect(region.textContent).toContain('silent')
})

// Finding a file: search covers prompts and notes, sort, group by date, and
// folders that collapse

/** The grid's files, in the order shown */
const shownNames = () =>
  screen
    .getAllByRole('checkbox')
    .map((c) => c.getAttribute('aria-label')!.replace(/^select /, ''))

const demoToggle = () =>
  [...document.querySelectorAll('button.group')].find((b) =>
    b.textContent?.includes('demo/'),
  ) as HTMLButtonElement

it('collapsing a folder hides it with the default roles', async () => {
  await renderGallery()
  const toggle = demoToggle()
  expect(toggle).not.toBeDisabled()
  await fireEvent.click(toggle)
  await waitFor(() =>
    expect(screen.queryByLabelText('select demo/c.png')).toBeNull(),
  )
  expect(toggle).toHaveAttribute('aria-expanded', 'false')
  // a filter shows every match again, and the toggle says why it is off
  await fireEvent.input(screen.getByPlaceholderText(SEARCH), {
    target: { value: 'c' },
  })
  await waitFor(() =>
    expect(screen.getByLabelText('select demo/c.png')).toBeTruthy(),
  )
  expect(demoToggle()).toBeDisabled()
  await fireEvent.input(screen.getByPlaceholderText(SEARCH), {
    target: { value: '' },
  })
  await fireEvent.click(demoToggle())
})

it('searching a prompt finds its file', async () => {
  listing.files = [
    { ...file('a.png'), run_id: 'r1' },
    { ...file('b.mp4'), run_id: 'r2', kind: 'video' },
    { ...file('c.png'), run_id: 'gone' },
  ]
  listing.prompts = { r1: 'a red fox', r2: 'waves on a grey shore' }
  await renderGallery()
  expect(gallery).toHaveBeenCalledWith(undefined, undefined, undefined, {
    prompts: true,
  })
  const search = screen.getByPlaceholderText(SEARCH)
  await fireEvent.input(search, { target: { value: 'grey shore' } })
  await waitFor(() => expect(shownNames()).toEqual(['b.mp4']))
  await fireEvent.input(search, { target: { value: 'FOX' } })
  await waitFor(() => expect(shownNames()).toEqual(['a.png']))
  // typing never takes focus away
  search.focus()
  await fireEvent.input(search, { target: { value: 'fox red' } })
  expect(document.activeElement).toBe(search)
})

it('the count says n of m while filtered', async () => {
  await renderGallery()
  const count = () => document.querySelector('.pagehead .count')?.textContent
  expect(count()).toBe('3 files')
  await fireEvent.input(screen.getByPlaceholderText(SEARCH), {
    target: { value: 'demo' },
  })
  await waitFor(() => expect(count()).toBe('1 of 3 files'))
})

it('sort and group survive Back', async () => {
  listing.files = [
    { ...file('b.png'), run_id: 'r2', mtime: Date.now() / 1000 },
    { ...file('a.png'), run_id: 'r1', mtime: 1 },
  ]
  await hashEvent(() => (location.hash = '#/ws/default/gallery?x=1'))
  await renderGallery('b.png')
  const sort = screen.getByRole('group', { name: 'Sort' })
  const groupBy = screen.getByRole('group', { name: 'Group by' })
  await fireEvent.click(within(sort).getByRole('button', { name: 'Name' }))
  await fireEvent.click(within(groupBy).getByRole('button', { name: 'Date' }))
  await waitFor(() =>
    expect(location.hash).toBe('#/ws/default/gallery?x=1&sort=name&group=date'),
  )
  // grouped by date, newest group first; the viewer steps in that order
  expect(
    [...document.querySelectorAll('button.group')].map((b) =>
      b.textContent?.trim(),
    ),
  ).toEqual(['Today (1)', 'Earlier (1)'])
  expect(shownNames()).toEqual(['b.png', 'a.png'])
  await openFile('b.png')
  await press({ key: 'ArrowRight' })
  expect(await viewer('a.png')).toHaveTextContent('2 of 2')
  // Escape closes the viewer by going Back
  await hashEvent(() => void press({ key: 'Escape' }))
  expect(location.hash).not.toContain('open=')

  cleanup()
  await hashEvent(() => (location.hash = '#/ws/default/jobs'))
  await hashEvent(() => history.back())
  render(GalleryPage)
  await waitFor(() =>
    expect(screen.getByLabelText('select a.png')).toBeTruthy(),
  )
  const pressed = (group: string, name: string) =>
    within(screen.getByRole('group', { name: group }))
      .getByRole('button', { name })
      .getAttribute('aria-pressed')
  expect(pressed('Sort', 'Name')).toBe('true')
  expect(pressed('Group by', 'Date')).toBe('true')
})

it('after Clear filters, folder toggles work', async () => {
  await renderGallery()
  await fireEvent.input(screen.getByPlaceholderText(SEARCH), {
    target: { value: 'zzz' },
  })
  await fireEvent.click(screen.getByRole('button', { name: 'Clear filters' }))
  await waitFor(() =>
    expect(screen.getByLabelText('select demo/c.png')).toBeTruthy(),
  )
  // Clear filters shows intermediates too, which hides nothing
  expect(demoToggle()).not.toBeDisabled()
  await fireEvent.click(demoToggle())
  await waitFor(() =>
    expect(screen.queryByLabelText('select demo/c.png')).toBeNull(),
  )
})

/** a.png and b.png at the top, demo/c.png shut away, zz/d.png below it */
async function renderShutDemo() {
  listing.files = ['a.png', 'b.png', 'demo/c.png', 'zz/d.png'].map((name) =>
    file(name),
  )
  await renderGallery()
  await fireEvent.click(demoToggle())
  await waitFor(() =>
    expect(screen.queryByLabelText('select demo/c.png')).toBeNull(),
  )
}

it('select all with a collapsed folder picks only the files shown', async () => {
  await renderShutDemo()
  await fireEvent.click(screen.getByRole('button', { name: /select all/i }))
  await waitFor(() => expect(screen.getByText('3 selected')).toBeTruthy())
  await fireEvent.click(screen.getByRole('button', { name: /^download/i }))
  await waitFor(() => expect(api.archiveOutputs).toHaveBeenCalledTimes(1))
  expect(api.archiveOutputs).toHaveBeenCalledWith([
    'a.png',
    'b.png',
    'zz/d.png',
  ])
})

it('a shift-range across a collapsed folder skips its files', async () => {
  await renderShutDemo()
  checkbox('a.png').click()
  checkbox('zz/d.png').dispatchEvent(
    new MouseEvent('click', { bubbles: true, shiftKey: true }),
  )
  await waitFor(() => expect(screen.getByText('3 selected')).toBeTruthy())
  // opening the folder again shows its file was never picked
  await fireEvent.click(demoToggle())
  await waitFor(() => expect(checkbox('demo/c.png')).not.toBeChecked())
  expect(screen.getByText('3 selected')).toBeTruthy()
})

it("the viewer's count leaves out a collapsed folder's files", async () => {
  await renderShutDemo()
  await openFile('b.png')
  expect(await viewer('b.png')).toHaveTextContent('2 of 3')
  await press({ key: 'ArrowRight' })
  expect(await viewer('d.png')).toHaveTextContent('3 of 3')
})

it('a delete confirm counts only the picks that are shown', async () => {
  await renderShutDemo()
  await fireEvent.click(demoToggle())
  await waitFor(() => expect(checkbox('demo/c.png')).toBeTruthy())
  await fireEvent.click(screen.getByRole('button', { name: /select all/i }))
  await waitFor(() => expect(screen.getByText('4 selected')).toBeTruthy())
  await fireEvent.click(demoToggle())
  await waitFor(() => expect(screen.getByText('3 selected')).toBeTruthy())
  await fireEvent.click(screen.getByRole('button', { name: /^delete/i }))
  const dialog = await waitFor(() => screen.getByRole('alertdialog'))
  expect(dialog.textContent).toContain('Delete 3 files?')
  within(dialog)
    .getByRole('button', { name: /^delete$/i })
    .click()
  await waitFor(() => expect(api.deleteOutput).toHaveBeenCalledTimes(3))
  expect(api.deleteOutput).not.toHaveBeenCalledWith('demo/c.png')
})

it('stars a selection, offers Undo, and marks nothing changed', async () => {
  listing.files = [{ ...file('a.png'), starred: true }, file('b.png')]
  await renderGallery()
  checkbox('a.png').click()
  checkbox('b.png').click()
  await waitFor(() => expect(screen.getByText('2 selected')).toBeTruthy())
  await fireEvent.click(screen.getByRole('button', { name: 'Star' }))

  await waitFor(() => expect(notifyUndo).toHaveBeenCalledTimes(1))
  expect(notifyUndo.mock.calls[0][0]).toBe('Starred 2 files')
  await waitFor(() => expect(starMark('b.png')).toBeTruthy())
  // the server now says so, in a listing that lands after the star
  listing.files = listing.files.map((f) => ({ ...f, starred: true }))
  feed.move(DEFAULT_WORKSPACE, 'gallery')
  await settle()
  expect(marked('a.png')).toBe(false)
  expect(marked('b.png')).toBe(false)
  // Undo puts a back starred and b back unstarred
  notifyUndo.mock.calls[0][1]()
  await waitFor(() =>
    expect(api.setStarred).toHaveBeenCalledWith('b.png', false),
  )
  expect(api.setStarred).toHaveBeenLastCalledWith('b.png', false)
  expect(api.setStarred).toHaveBeenCalledWith('a.png', true)
  await waitFor(() => expect(starMark('b.png')).toBeNull())
  expect(starMark('a.png')).toBeTruthy()
})

it('names the files a bulk star could not star', async () => {
  await renderGallery()
  checkbox('a.png').click()
  checkbox('b.png').click()
  api.setStarred.mockImplementationOnce(() => Promise.reject(new Error('gone')))
  await fireEvent.click(await screen.findByRole('button', { name: 'Star' }))

  await waitFor(() => expect(notifyFailure).toHaveBeenCalled())
  expect(notifyFailure.mock.calls[0][3]).toContain('a.png')
  expect(notifyUndo.mock.calls[0][0]).toBe('Starred 1 file')
})

it('refuses a bulk star Undo after a workspace switch', async () => {
  await renderGallery()
  checkbox('a.png').click()
  await fireEvent.click(await screen.findByRole('button', { name: 'Star' }))
  await waitFor(() => expect(notifyUndo).toHaveBeenCalledTimes(1))
  api.setStarred.mockClear()
  workspace.current = 'elsewhere'
  notifyUndo.mock.calls[0][1]()
  await settle()
  expect(api.setStarred).not.toHaveBeenCalled()
  expect(notifyError).toHaveBeenCalledWith(
    'Nothing undone: another workspace is open now.',
  )
})

it('a double click on Star sends one round', async () => {
  await renderGallery()
  checkbox('a.png').click()
  checkbox('b.png').click()
  const star = await screen.findByRole('button', { name: 'Star' })
  star.click()
  star.click()

  await waitFor(() => expect(api.setStarred).toHaveBeenCalledTimes(2))
  await settle()
  expect(api.setStarred).toHaveBeenCalledTimes(2)
  expect(notifyUndo).toHaveBeenCalledTimes(1)
})

it('an empty gallery says where to go', async () => {
  location.hash = '#/ws/default/gallery'
  listing.files = []
  render(GalleryPage)
  const link = await screen.findByRole('link', { name: 'Run a workflow' })
  expect(link.getAttribute('href')).toMatch(/^#\/ws\/[^/]+\/workflows$/)
})

it('GalleryPage says when each output changed', async () => {
  const now = Math.floor(Date.now() / 1000)
  listing.files = [{ ...file('a.png'), mtime: now - 10 * 60 }]
  await renderGallery()
  const stamp = tile('a.png')!
    .closest('.filetile')!
    .querySelector('.tile-meta time')!
  expect(stamp.textContent?.trim()).toBe('changed 10 min ago')
  expect(stamp.getAttribute('title')).toBeTruthy()
})
