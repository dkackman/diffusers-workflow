import {
  cleanup,
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
import ConfirmDialog from '../ConfirmDialog.svelte'
import type { GalleryFile } from '../types'
import { DEFAULT_WORKSPACE, workspace } from '../workspace.svelte'

const file = (name: string, subfolder = ''): GalleryFile => ({
  name,
  folder: name.includes('/') ? name.split('/')[0] : '',
  subfolder,
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
vi.mock('../api', () => ({
  api: {
    gallery: () => gallery(),
    galleryThumbnailUrl: (name: string) => `/thumb/${name}`,
    galleryUrl: (name: string) => `/outputs/${name}`,
    outputDownloadUrl: (name: string) => `/download/${name}`,
    deleteOutput: (name: string) => deleteOutput(name),
    archiveOutputs: (names: string[]) => archiveOutputs(names),
    galleryMetadata: vi.fn(() => new Promise(() => {})),
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
  archiveOutputs.mockClear()
  notifyError.mockClear()
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

it('offers no subfolder control when nothing was written to one', async () => {
  await renderGallery()
  expect(screen.queryByRole('combobox', { name: 'subfolder' })).toBeNull()
})

it('lists the subfolders the outputs landed in and filters the grid by one', async () => {
  listing.files = [
    file('wf/run-1/final/deliverable.png', 'final'),
    file('wf/run-1/intermediate/scratch.png', 'intermediate'),
    file('wf/run-1/root.png', ''),
  ]
  await renderGallery('wf/run-1/final/deliverable.png')

  const pick = screen.getByRole('combobox', { name: 'subfolder' }) as HTMLSelectElement
  expect([...pick.options].map((o) => o.textContent?.trim())).toEqual([
    'all subfolders',
    '(run root)',
    'final/',
    'intermediate/',
  ])

  pick.value = 'final'
  pick.dispatchEvent(new Event('change', { bubbles: true }))
  await waitFor(() =>
    expect(screen.queryByLabelText('select wf/run-1/intermediate/scratch.png')).toBeNull(),
  )
  expect(screen.getByLabelText('select wf/run-1/final/deliverable.png')).toBeTruthy()
  expect(screen.queryByLabelText('select wf/run-1/root.png')).toBeNull()
  // Select all takes what the subfolder filter leaves showing
  screen.getByRole('button', { name: /select all matching \(1\)/i }).click()
  await waitFor(() => expect(screen.getByText('1 selected')).toBeTruthy())
})

it('intersects the subfolder pick with the text filter', async () => {
  listing.files = [
    file('wf/run-1/final/a.png', 'final'),
    file('wf/run-1/final/b.png', 'final'),
    file('wf/run-1/intermediate/a.png', 'intermediate'),
  ]
  await renderGallery('wf/run-1/final/a.png')

  const pick = screen.getByRole('combobox', { name: 'subfolder' }) as HTMLSelectElement
  pick.value = 'final'
  pick.dispatchEvent(new Event('change', { bubbles: true }))
  const filter = screen.getByPlaceholderText('filter…') as HTMLInputElement
  filter.value = '/a.'
  filter.dispatchEvent(new Event('input', { bubbles: true }))

  await waitFor(() =>
    expect(screen.queryByLabelText('select wf/run-1/final/b.png')).toBeNull(),
  )
  expect(screen.getByLabelText('select wf/run-1/final/a.png')).toBeTruthy()
  expect(screen.queryByLabelText('select wf/run-1/intermediate/a.png')).toBeNull()
})

it('falls back to every subfolder when the picked one empties out', async () => {
  listing.files = [
    file('wf/run-1/final/only.png', 'final'),
    file('wf/run-1/root.png', ''),
  ]
  await renderGallery('wf/run-1/final/only.png')
  const pick = screen.getByRole('combobox', { name: 'subfolder' }) as HTMLSelectElement
  pick.value = 'final'
  pick.dispatchEvent(new Event('change', { bubbles: true }))
  await waitFor(() =>
    expect(screen.queryByLabelText('select wf/run-1/root.png')).toBeNull(),
  )

  checkbox('wf/run-1/final/only.png').click()
  await waitFor(() => expect(screen.getByText('1 selected')).toBeTruthy())
  screen.getByRole('button', { name: /^delete/i }).click()
  await answerConfirm(true)

  // The last final/ file is gone: the control goes with it and the grid
  // shows everything again rather than an empty page pinned to a value
  // that no longer exists
  await waitFor(() =>
    expect(screen.getByLabelText('select wf/run-1/root.png')).toBeTruthy(),
  )
  expect(screen.queryByRole('combobox', { name: 'subfolder' })).toBeNull()
})

it('resets a run-root pick when the control it depends on disappears', async () => {
  listing.files = [
    file('wf/run-1/final/only.png', 'final'),
    file('wf/run-1/root.png', ''),
  ]
  await renderGallery('wf/run-1/final/only.png')
  const pick = screen.getByRole('combobox', { name: 'subfolder' }) as HTMLSelectElement
  // "all subfolders" and "(run root)" both carry the native value "" -
  // Svelte tells them apart by each option's bound __value - so picking by
  // index is what actually lands on "(run root)" rather than falling back
  // to "all subfolders"
  pick.selectedIndex = 1
  pick.dispatchEvent(new Event('change', { bubbles: true }))
  await waitFor(() =>
    expect(screen.queryByLabelText('select wf/run-1/final/only.png')).toBeNull(),
  )
  expect(screen.getByLabelText('select wf/run-1/root.png')).toBeTruthy()

  // The run-root pick has just hidden the only other file, so nothing in
  // this grid's own checkboxes can reach it to delete it - a workspace
  // switch stands in for that (or the same file being deleted from
  // elsewhere): the listing comes back with no subfolder at all, the same
  // shape the last foldered file's own deletion would leave behind
  listing.files = [file('wf/run-1/root.png', '')]
  workspace.current = 'other'
  await waitFor(() => expect(gallery).toHaveBeenCalledTimes(2))

  // The control goes with the pick it can no longer offer, and "Select
  // all" reads as unfiltered again rather than sticking on a filter
  // nothing can now clear
  expect(screen.queryByRole('combobox', { name: 'subfolder' })).toBeNull()
  expect(
    screen.getByRole('button', { name: /^select all \(1\)$/i }),
  ).toBeTruthy()
})
