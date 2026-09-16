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
import type { AssetFile, AssetLibrary, ShadowedAsset } from '../types'
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

// An entry a nearer library hides: an asset's shape minus the `url` the
// server will not hand out, plus which origin won
const shadowedAsset = (
  name: string,
  origin: AssetFile['origin'],
  shadowedBy: AssetFile['origin'],
): ShadowedAsset => {
  const { url, ...rest } = asset(name, origin)
  // Dropped deliberately: that URL would serve the file that won
  void url
  return { ...rest, shadowed_by: shadowedBy }
}

const WORKSPACE_LIBRARY: AssetLibrary = {
  origin: 'workspace',
  dir: '/ws/assets',
  writable: true,
}
const SHARED_LIBRARY: AssetLibrary = {
  origin: 'common',
  dir: '/root/common/assets',
  writable: true,
}
const EXAMPLES_LIBRARY: AssetLibrary = {
  origin: 'examples',
  dir: '/examples/assets',
  writable: false,
}

const listing = vi.hoisted(() => ({
  assets: [] as AssetFile[],
  asset_dir: '/ws/assets' as string | null,
  asset_dirs: ['/ws/assets'] as string[],
  libraries: [] as AssetLibrary[],
  shadowed: [] as ShadowedAsset[],
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
  listing.libraries = [WORKSPACE_LIBRARY]
  listing.shadowed = []
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
  // The collapse state is persisted, so a test that collapses a section
  // would otherwise hand it to whatever renders next
  localStorage.clear()
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

// The search path is the page's top level: a library is a section, and
// folders sit inside it. Which library a name lives in is what decides
// whether it can be deleted, what a delete costs, and what it hides

it('sections the grid by library, in search-path order', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, SHARED_LIBRARY, EXAMPLES_LIBRARY]
  listing.assets = [
    asset('iris.png'),
    asset('shared/logo.png', 'common'),
    asset('demo/x.png', 'examples'),
  ]
  await renderAssets()

  const headers = [...document.querySelectorAll('.library')].map(
    (node) => node.textContent ?? '',
  )
  expect(headers.length).toBe(3)
  expect(headers[0]).toContain('This workspace')
  expect(headers[1]).toContain('Shared library')
  expect(headers[2]).toContain('Examples')
  expect(screen.getByText('/root/common/assets')).toBeTruthy()
})

it('keeps an empty library section so an upload has somewhere to land', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, SHARED_LIBRARY]
  listing.assets = [asset('shared/logo.png', 'common')]
  await renderAssets('logo.png')

  expect(screen.getByRole('button', { name: /^Upload$/ })).toBeTruthy()
  expect(
    [...document.querySelectorAll('.library')].some((node) =>
      node.textContent?.includes('This workspace'),
    ),
  ).toBe(true)
})

// Nothing bulk can do to a read-only asset - the server answers 403 - so
// the tile offers no tick to begin with
it('offers no checkbox on a tile from a read-only library', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, EXAMPLES_LIBRARY]
  listing.assets = [asset('iris.png'), asset('demo/x.png', 'examples')]
  await renderAssets()

  expect(screen.getByLabelText('select iris.png')).toBeTruthy()
  expect(screen.queryByLabelText('select demo/x.png')).toBeNull()
})

// A bulk action must never touch what the user cannot see or untick, so
// select-all takes what the grid is actually showing with a checkbox on it

it('selects only what a bulk action could act on', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, EXAMPLES_LIBRARY]
  listing.assets = [
    asset('iris.png'),
    asset('cast/priya.jpg'),
    asset('demo/x.png', 'examples'),
  ]
  await renderAssets()

  // Two, not three: an examples tile has no checkbox, and a delete would
  // 403 on it and leave it ticked with nothing on the tile to untick
  screen.getByRole('button', { name: /select all \(2\)/i }).click()

  await waitFor(() => expect(screen.getByText('2 selected')).toBeTruthy())
})

it('leaves a collapsed library out of what select-all takes', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, SHARED_LIBRARY]
  listing.assets = [asset('iris.png'), asset('shared/logo.png', 'common')]
  await renderAssets()
  expect(screen.getByRole('button', { name: /select all \(2\)/i })).toBeTruthy()

  screen.getByRole('button', { name: /This workspace/ }).click()

  await waitFor(() =>
    expect(
      screen.getByRole('button', { name: /select all \(1\)/i }),
    ).toBeTruthy(),
  )
  screen.getByRole('button', { name: /select all \(1\)/i }).click()
  await waitFor(() => expect(screen.getByText('1 selected')).toBeTruthy())
})

it('shows what a nearer library is hiding, dimmed and inert', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, SHARED_LIBRARY]
  listing.assets = [asset('iris.png')]
  listing.shadowed = [shadowedAsset('iris.png', 'common', 'workspace')]
  // Not renderAssets: a shadowed entry carries the same name as the file
  // that hides it, so the caption is deliberately on the page twice
  render(AssetsPage)

  const tile = await waitFor(() =>
    screen.getByTitle(
      "shadowed by this workspace's iris.png - asset:iris.png resolves to that file",
    ),
  )
  expect(tile.className).toContain('shadowed')
  // Not a button and not tickable: there is no file here to open or act on
  expect(tile.tagName).toBe('DIV')
  expect(screen.queryByLabelText('select iris.png')).toBeTruthy()
  expect(screen.getAllByLabelText(/^select /).length).toBe(1)
})

it('collapses a library section, and remembers it', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, SHARED_LIBRARY]
  listing.assets = [asset('iris.png'), asset('shared/logo.png', 'common')]
  await renderAssets()

  screen.getByRole('button', { name: /This workspace/ }).click()
  await waitFor(() => expect(screen.queryByText('iris.png')).toBeNull())
  expect(screen.getByText('logo.png')).toBeTruthy()

  cleanup()
  await renderAssets('logo.png')
  expect(screen.queryByText('iris.png')).toBeNull()
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
  listing.libraries = [EXAMPLES_LIBRARY]
  listing.assets = [asset('demo/x.png', 'examples')]
  await renderAssets('x.png')

  screen.getByLabelText('show details for demo/x.png').click()

  // By title, not by text: the section header says read-only too
  await waitFor(() =>
    expect(
      screen.getByTitle('read-only: an examples library brought it'),
    ).toBeTruthy(),
  )
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

// Where an upload lands is the one thing about it that cannot be changed
// afterwards, so it is the section's own button rather than a destination
// pick that can be left pointing anywhere

function chooseFile(name = 'kept.png') {
  vi.stubGlobal(
    'prompt',
    vi.fn(() => name),
  )
  const input = document.querySelector('input[type="file"]') as HTMLInputElement
  const file = new File(['x'], name, { type: 'image/png' })
  Object.defineProperty(input, 'files', { value: [file], configurable: true })
  input.dispatchEvent(new Event('change', { bubbles: true }))
  return file
}

it('uploads into the workspace from the workspace section', async () => {
  await renderAssets()

  screen.getByRole('button', { name: /^Upload$/ }).click()
  const file = chooseFile()

  await waitFor(() => expect(uploadMedia).toHaveBeenCalledTimes(1))
  expect(uploadMedia).toHaveBeenCalledWith(file, 'kept.png', false)
})

it('uploads into the shared library from the shared section', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, SHARED_LIBRARY]
  await renderAssets()

  screen.getByRole('button', { name: /Upload to shared/ }).click()
  const file = chooseFile()

  await waitFor(() => expect(uploadMedia).toHaveBeenCalledTimes(1))
  expect(uploadMedia).toHaveBeenCalledWith(file, 'kept.png', true)
})

// Deleting a shared asset is not deleting a workspace one: it goes for
// every workspace under the root, and the confirm has to say so

it('says what a shared delete costs', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, SHARED_LIBRARY]
  listing.assets = [asset('shared/logo.png', 'common')]
  await renderAssets('logo.png')
  screen.getByLabelText('show details for shared/logo.png').click()
  await waitFor(() =>
    screen.getByLabelText('delete this asset from the library').click(),
  )

  const dialog = await waitFor(() => screen.getByRole('alertdialog'))
  expect(dialog.textContent).toContain(
    'Delete asset:shared/logo.png from the shared library? Every workspace ' +
      'under this root loses it, and any workflow still carrying that ' +
      'reference stops loading.',
  )
  // Answer it: an unanswered confirm is module state the next render sees
  await answerConfirm(false)
})

it('says nothing about a shared library when nothing shared is picked', async () => {
  await renderAssets()

  screen.getByRole('button', { name: /select all \(2\)/i }).click()
  await waitFor(() => screen.getByRole('button', { name: /^delete$/i }))
  screen.getByRole('button', { name: /^delete$/i }).click()

  const dialog = await waitFor(() => screen.getByRole('alertdialog'))
  expect(dialog.textContent).toContain(
    'Delete 2 assets? Any workflow still carrying one of those references ' +
      'stops loading.',
  )
  await answerConfirm(false)
})

it('counts the shared assets in a bulk delete confirm', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, SHARED_LIBRARY]
  listing.assets = [
    asset('iris.png'),
    asset('shared/logo.png', 'common'),
    asset('shared/mark.png', 'common'),
  ]
  await renderAssets()

  screen.getByRole('button', { name: /select all \(3\)/i }).click()
  await waitFor(() => screen.getByRole('button', { name: /^delete$/i }))
  screen.getByRole('button', { name: /^delete$/i }).click()

  const dialog = await waitFor(() => screen.getByRole('alertdialog'))
  expect(dialog.textContent).toContain(
    'Delete 3 assets? 2 are in the shared library and go away for every ' +
      'workspace. Any workflow still carrying one of those references stops ' +
      'loading.',
  )
  await answerConfirm(false)
})
