import {
  cleanup,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
// Hoisted above the imports for the reason GalleryPage.test.ts states: the
// static import below is hoisted too, and must see an initialized mock
import AssetsPage from './AssetsPage.svelte'
import ConfirmDialog from '../ConfirmDialog.svelte'
import type { AssetFile } from '../types'
import { DEFAULT_WORKSPACE, workspace } from '../workspace.svelte'

const asset = (
  name: string,
  origin: AssetFile['origin'] = 'workspace',
  kind: AssetFile['kind'] = 'image',
): AssetFile => ({
  name,
  reference: `asset:${name}`,
  folder: name.includes('/') ? name.split('/')[0] : '',
  kind,
  size: 2048,
  mtime: 1,
  origin,
  url: `/inputs/${name}`,
})

const listing = vi.hoisted(() => ({
  assets: [] as AssetFile[],
  asset_dir: '/ws/assets' as string | null,
  asset_dirs: ['/ws/assets'] as string[],
  libraries: [{ origin: 'workspace', dir: '/ws/assets', writable: true }],
  shadowed: [] as unknown[],
}))

const listAssets = vi.hoisted(() =>
  vi.fn(
    () =>
      new Promise<typeof listing>((resolve) =>
        setTimeout(() => resolve({ ...listing, folders: [] } as never), 0),
      ),
  ),
)
const deleteAsset = vi.hoisted(() =>
  vi.fn<(name: string) => Promise<void>>(() => Promise.resolve()),
)
const archiveAssets = vi.hoisted(() =>
  vi.fn<(names: string[]) => Promise<void>>(() => Promise.resolve()),
)
const uploadMedia = vi.hoisted(() =>
  vi.fn<(file: File, assetName?: string, shared?: boolean) => Promise<unknown>>(
    () => Promise.resolve({ reference: 'asset:uploads/x.png' }),
  ),
)
vi.mock('../api', () => ({
  api: {
    listAssets: () => listAssets(),
    deleteAsset: (name: string) => deleteAsset(name),
    archiveAssets: (names: string[]) => archiveAssets(names),
    uploadMedia: (file: File, assetName?: string, shared?: boolean) =>
      uploadMedia(file, assetName, shared),
  },
}))

const notifyError = vi.hoisted(() => vi.fn())
const notifySuccess = vi.hoisted(() => vi.fn())
vi.mock('../toast', () => ({
  notify: { error: notifyError, success: notifySuccess },
}))

beforeEach(() => {
  listing.assets = [asset('iris.png'), asset('cast/priya.jpg')]
  listing.asset_dir = '/ws/assets'
  listing.asset_dirs = ['/ws/assets']
  deleteAsset.mockResolvedValue(undefined)
  archiveAssets.mockResolvedValue(undefined)
})
afterEach(() => {
  cleanup()
  listAssets.mockClear()
  // Reset, not clear: one test sets an implementation, and mockClear
  // leaves it in place for whatever runs next
  deleteAsset.mockReset()
  archiveAssets.mockReset()
  uploadMedia.mockClear()
  notifyError.mockClear()
  notifySuccess.mockClear()
  vi.unstubAllGlobals()
  workspace.current = DEFAULT_WORKSPACE
})

async function renderAssets(first = 'iris.png') {
  render(ConfirmDialog)
  render(AssetsPage)
  await waitFor(() => expect(screen.getByText(first)).toBeTruthy())
}

async function answerConfirm(accept: boolean) {
  const dialog = await waitFor(() => screen.getByRole('alertdialog'))
  within(dialog)
    .getByRole('button', { name: accept ? /^delete$/i : /^cancel$/i })
    .click()
}

it('fetches the asset library exactly once on mount', async () => {
  render(AssetsPage)
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 10))
  expect(listAssets).toHaveBeenCalledTimes(1)
})

it('shows every asset, including one in a folder', async () => {
  await renderAssets()

  expect(screen.getByText('iris.png')).toBeTruthy()
  expect(screen.getByText('priya.jpg')).toBeTruthy()
})

it('shows the reference, not the path, when an asset is selected', async () => {
  await renderAssets()

  screen.getByLabelText('show details for iris.png').click()

  await waitFor(() => expect(screen.getByText('asset:iris.png')).toBeTruthy())
})

it('marks an asset that came from another library', async () => {
  listing.assets = [asset('iris.png'), asset('shared/logo.png', 'common')]
  await renderAssets()

  // Specific to the badge on the tile: 'common' is also an option in the
  // library pick, which two origins bring out
  expect(
    screen.getByTitle(
      "from the common library - this workspace's own names shadow it",
    ).textContent,
  ).toContain('common')
})

it('offers no library pick when everything is the workspace own', async () => {
  await renderAssets()

  expect(screen.queryByLabelText('library')).toBeNull()
})

it('offers a library pick once more than one is on the path', async () => {
  listing.assets = [asset('iris.png'), asset('demo/x.png', 'examples')]
  await renderAssets()

  expect(screen.getByLabelText('library')).toBeTruthy()
})

it('deletes an asset once the confirmation is answered', async () => {
  await renderAssets()
  screen.getByLabelText('show details for iris.png').click()
  await waitFor(() =>
    screen.getByLabelText('delete this asset from the library').click(),
  )

  await answerConfirm(true)

  await waitFor(() => expect(deleteAsset).toHaveBeenCalledWith('iris.png'))
})

it('deletes nothing when the confirmation is declined', async () => {
  await renderAssets()
  screen.getByLabelText('show details for iris.png').click()
  await waitFor(() =>
    screen.getByLabelText('delete this asset from the library').click(),
  )

  await answerConfirm(false)

  await new Promise((r) => setTimeout(r, 10))
  expect(deleteAsset).not.toHaveBeenCalled()
})

it('offers no delete for a read-only examples asset', async () => {
  listing.assets = [asset('demo/x.png', 'examples')]
  await renderAssets('x.png')

  screen.getByLabelText('show details for demo/x.png').click()

  await waitFor(() => expect(screen.getByText('read-only')).toBeTruthy())
  expect(
    screen.queryByLabelText('delete this asset from the library'),
  ).toBeNull()
})

it('filters by name', async () => {
  await renderAssets()
  const filter = screen.getByPlaceholderText('filter…') as HTMLInputElement
  filter.value = 'priya'
  filter.dispatchEvent(new Event('input', { bubbles: true }))

  await waitFor(() => expect(screen.queryByText('iris.png')).toBeNull())
  expect(screen.getByText('priya.jpg')).toBeTruthy()
})

it('refetches when the workspace changes', async () => {
  await renderAssets()
  expect(listAssets).toHaveBeenCalledTimes(1)

  workspace.current = 'other'

  await waitFor(() => expect(listAssets).toHaveBeenCalledTimes(2))
})

// Escape means "close the thing on top". A confirm dialog answers it
// itself, so the page must not also take the detail away underneath it
it('leaves the detail open when Escape answers a confirm dialog', async () => {
  await renderAssets()
  screen.getByLabelText('show details for iris.png').click()
  await waitFor(() => expect(screen.getByText('asset:iris.png')).toBeTruthy())

  screen.getByLabelText('delete this asset from the library').click()
  await waitFor(() => expect(screen.getByRole('alertdialog')).toBeTruthy())

  window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))

  await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
  expect(screen.getByText('asset:iris.png')).toBeTruthy()
})

// The gallery's bulk actions, for the input side: tick files in the grid,
// then download or delete the lot in one go

it('downloads the ticked assets as one zip', async () => {
  await renderAssets()

  screen.getByLabelText('select iris.png').click()
  screen.getByLabelText('select cast/priya.jpg').click()
  await waitFor(() => screen.getByRole('button', { name: /download \.zip/i }))
  screen.getByRole('button', { name: /download \.zip/i }).click()

  await waitFor(() => expect(archiveAssets).toHaveBeenCalledTimes(1))
  // The order the grid shows them in, not the order they were ticked
  expect(archiveAssets).toHaveBeenCalledWith(['iris.png', 'cast/priya.jpg'])
})

it('deletes every ticked asset and refetches once', async () => {
  await renderAssets()

  screen.getByLabelText('select iris.png').click()
  screen.getByLabelText('select cast/priya.jpg').click()
  await waitFor(() => screen.getByRole('button', { name: /^delete$/i }))
  screen.getByRole('button', { name: /^delete$/i }).click()
  await answerConfirm(true)

  await waitFor(() => expect(deleteAsset).toHaveBeenCalledTimes(2))
  expect(deleteAsset.mock.calls.map((c) => c[0])).toEqual([
    'iris.png',
    'cast/priya.jpg',
  ])
})

// A read-only asset answers 403. Whatever could not go stays ticked, so a
// retry needs no re-ticking and the failure is visible rather than dropped
it('keeps an asset that would not delete in the selection', async () => {
  deleteAsset.mockImplementation((name: string) =>
    name === 'iris.png'
      ? Promise.reject(new Error('read-only'))
      : Promise.resolve(),
  )
  await renderAssets()

  screen.getByLabelText('select iris.png').click()
  screen.getByLabelText('select cast/priya.jpg').click()
  await waitFor(() => screen.getByRole('button', { name: /^delete$/i }))
  screen.getByRole('button', { name: /^delete$/i }).click()
  await answerConfirm(true)

  await waitFor(() => expect(notifyError).toHaveBeenCalled())
  expect(screen.getByText('1 selected')).toBeTruthy()
  expect(
    (screen.getByLabelText('select iris.png') as HTMLInputElement).checked,
  ).toBe(true)
})

it('clears the selection on Escape before it closes the detail', async () => {
  await renderAssets()
  screen.getByLabelText('show details for iris.png').click()
  await waitFor(() => expect(screen.getByText('asset:iris.png')).toBeTruthy())
  screen.getByLabelText('select iris.png').click()
  await waitFor(() => expect(screen.getByText('1 selected')).toBeTruthy())

  window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
  await waitFor(() => expect(screen.queryByText('1 selected')).toBeNull())
  // The detail is the older state, so it survives the first press
  expect(screen.getByText('asset:iris.png')).toBeTruthy()

  window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
  await waitFor(() => expect(screen.queryByText('asset:iris.png')).toBeNull())
})

it('selects all matching when a filter is on', async () => {
  await renderAssets()

  const filter = screen.getByPlaceholderText('filter…') as HTMLInputElement
  filter.value = 'priya'
  filter.dispatchEvent(new Event('input', { bubbles: true }))
  await waitFor(() => expect(screen.queryByText('iris.png')).toBeNull())

  screen.getByRole('button', { name: /select all matching \(1\)/i }).click()

  await waitFor(() => expect(screen.getByText('1 selected')).toBeTruthy())
})

// A shared or examples library is on every workspace's search path, so
// most of the grid can be identical in two different workspaces. The page
// has to say so, or switching workspace reads as a page that did nothing

it('keeps the library pick when every asset comes from another library', async () => {
  listing.assets = [
    asset('qa-cast/a.png', 'common'),
    asset('uploads/b.png', 'common'),
  ]
  await renderAssets('a.png')

  // One origin, but not this workspace's own: the pick is the control that
  // explains why these files follow you from workspace to workspace
  expect(screen.getByLabelText('library')).toBeTruthy()
})

it('says how many assets came from another library', async () => {
  listing.assets = [
    asset('iris.png'),
    asset('qa-cast/a.png', 'common'),
    asset('ex.png', 'examples'),
  ]
  await renderAssets()

  // One node, and the separator keeps its spaces - Svelte trims the leading
  // whitespace of a block, which has eaten this space twice now
  expect(screen.getByText('3 files · 2 from other libraries')).toBeTruthy()
})

it('says nothing about other libraries when every asset is this workspace own', async () => {
  await renderAssets()

  expect(screen.queryByText(/from other libraries/)).toBeNull()
})

it('names where an upload will land, and uploads there', async () => {
  await renderAssets()

  const destination = screen.getByLabelText(
    'where an upload lands',
  ) as HTMLSelectElement
  expect(destination.value).toBe('workspace')
  destination.value = 'shared'
  destination.dispatchEvent(new Event('change', { bubbles: true }))

  vi.stubGlobal(
    'prompt',
    vi.fn(() => 'kept.png'),
  )
  const input = document.querySelector('input[type="file"]') as HTMLInputElement
  const file = new File(['x'], 'kept.png', { type: 'image/png' })
  Object.defineProperty(input, 'files', { value: [file], configurable: true })
  input.dispatchEvent(new Event('change', { bubbles: true }))

  await waitFor(() => expect(uploadMedia).toHaveBeenCalledTimes(1))
  expect(uploadMedia).toHaveBeenCalledWith(file, 'kept.png', true)
})
