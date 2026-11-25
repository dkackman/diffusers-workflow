import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
// Hoisted above the imports for the reason GalleryPage.test.ts states: the
// static import below is hoisted too, and must see an initialized mock
import AssetsPage from './AssetsPage.svelte'
import ConfirmDialog from '../ui/ConfirmDialog.svelte'
import type { AssetFile, AssetLibrary, ShadowedAsset } from '../types'
import { DEFAULT_WORKSPACE, workspace } from '../workspace.svelte'
import { api } from '../test/apiMock'
import { feed } from '../test/feedMock'
import { formatExact, formatMtime } from '../format'

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
  writable: origin !== 'examples',
  url: `/inputs/${name}`,
  // A workflow of the workspace's names it, so a workspace's page shows a
  // shared one; the tests about an unused shared asset drop this
  used_by: ['portrait'],
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
  root: '/ws/assets',
  writable: true,
}
const SHARED_LIBRARY: AssetLibrary = {
  origin: 'common',
  root: '/root/common/assets',
  writable: true,
}
const EXAMPLES_LIBRARY: AssetLibrary = {
  origin: 'examples',
  root: '/examples/assets',
  writable: false,
}

const listing = vi.hoisted(() => ({
  assets: [] as AssetFile[],
  workspace: 'default',
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
// The background `media` read, kept apart so the listing's call counts
// stay the listing's
const assetDurations = vi.hoisted(() =>
  vi.fn((opts?: { limit?: number }) => {
    void opts
    return Promise.resolve({ assets: [] as AssetFile[] })
  }),
)
const setNote = vi.hoisted(() => vi.fn())
// What the server holds as the note a save reads first (its etag read)
const noteHeld = vi.hoisted(() => ({ note: null as string | null }))
// Answered at once by default: the note and Used by sit under the recipe's
// read (RecipePanel's children), so a read that never lands hides them
const NO_RECIPE = { metadata: null, job: null }
// The page also passes the workspace; the assertions here are on the
// reference alone, so the stub forwards only that
const galleryMetadata = vi.hoisted(() =>
  vi.fn<(reference: string) => Promise<unknown>>(() =>
    Promise.resolve({ metadata: null, job: null }),
  ),
)
vi.mock('../api', async () =>
  (await import('../test/apiMock')).apiModule({
    setNote: (t: string, n: string | null) => setNote(t, n),
  }),
)
api.listAssets.mockImplementation(
  (opts?: { media?: boolean; limit?: number }) =>
    opts?.media ? assetDurations(opts) : listAssets(),
)
api.noteForSave.mockImplementation((target: string) =>
  Promise.resolve({ target, note: noteHeld.note, etag: '"e"' }),
)
api.galleryMetadata.mockImplementation((reference: string) =>
  galleryMetadata(reference),
)
api.getJob.mockImplementation(() => new Promise(() => {}))
api.deleteAsset.mockImplementation(() => Promise.resolve())
api.archiveAssets.mockImplementation(() => Promise.resolve())
api.uploadMedia.mockImplementation(() =>
  Promise.resolve({ reference: 'asset:uploads/x.png' }),
)

vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))

const notifyError = vi.hoisted(() => vi.fn())
const notifySuccess = vi.hoisted(() => vi.fn())
const notifyFailure = vi.hoisted(() => vi.fn())
vi.mock('../toast', () => ({
  notify: {
    error: notifyError,
    success: notifySuccess,
    failure: notifyFailure,
  },
}))

beforeEach(() => {
  // The page's own place, with nothing in its query: the text filter and
  // the open asset live there, and a test that set one would hand it to
  // whatever renders next. A hash the router does not route (none) is
  // sent to the overview, under the viewer
  history.replaceState(null, '', `${location.pathname}#/ws/default/assets`)
  listing.assets = [asset('iris.png'), asset('cast/priya.jpg')]
  listing.libraries = [WORKSPACE_LIBRARY]
  listing.shadowed = []
  api.deleteAsset.mockResolvedValue(undefined)
  api.archiveAssets.mockResolvedValue(undefined)
})
afterEach(async () => {
  cleanup()
  listAssets.mockClear()
  assetDurations.mockClear()
  // Reset, not clear: one test sets an implementation, and mockClear
  // leaves it in place for whatever runs next
  api.deleteAsset.mockReset()
  api.archiveAssets.mockReset()
  api.uploadMedia.mockClear()
  galleryMetadata.mockReset()
  galleryMetadata.mockImplementation(() => Promise.resolve(NO_RECIPE))
  setNote.mockReset()
  notifyError.mockClear()
  notifySuccess.mockClear()
  notifyFailure.mockClear()
  vi.unstubAllGlobals()
  // The collapse state is persisted, so a test that collapses a section
  // would otherwise hand it to whatever renders next
  localStorage.clear()
  workspace.current = DEFAULT_WORKSPACE
})

async function renderAssets(first = 'iris.png', shared = false) {
  render(ConfirmDialog)
  render(AssetsPage, { shared })
  await waitFor(() => expect(screen.getByText(first)).toBeTruthy())
}

// A key press starts at the focused element and bubbles to the document
// and the window - where an overlay and the page each listen
function pressEscape() {
  ;(document.activeElement ?? document.body).dispatchEvent(
    new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }),
  )
}

/** The viewer, open on `name` */
const viewer = (name: string) =>
  screen.findByRole('dialog', { name: name.split('/').pop() })

/** Opens an asset in the viewer by clicking its tile, and waits for it */
async function openAsset(name: string) {
  // The open pushes a hash entry: wait for its hashchange (and the
  // viewer's own listener on it) here, not in a later test
  await hashEvent(
    () => void fireEvent.click(screen.getByLabelText(`Open ${name}`)),
  )
  return viewer(name)
}

/** Steps the viewer to the next or previous asset, and waits for it */
async function step(to: 'Next file' | 'Previous file', name: string) {
  await fireEvent.click(screen.getByRole('button', { name: to }))
  return viewer(name)
}

const hashEvent = (change: () => void) =>
  new Promise<void>((resolve) => {
    window.addEventListener('hashchange', () => resolve(), { once: true })
    change()
  })

const hashOpen = () =>
  new URLSearchParams(location.hash.split('?')[1]).get('open')

async function answerConfirm(accept: boolean) {
  const dialog = await waitFor(() => screen.getByRole('alertdialog'))
  within(dialog)
    .getByRole('button', { name: accept ? /^delete$/i : /^cancel$/i })
    .click()
}

it('fetches the asset library once on mount, and again when assets move, without blanking', async () => {
  await renderAssets()
  for (let i = 0; i < 3; i++) await new Promise((r) => setTimeout(r, 10))
  expect(listAssets).toHaveBeenCalledTimes(1)

  listing.assets = [
    asset('iris.png'),
    asset('cast/priya.jpg'),
    asset('new.png'),
  ]
  feed.move('default', 'assets')
  // The list stays up while the reload is asked
  expect(screen.getByText('iris.png')).toBeTruthy()
  await waitFor(() => expect(screen.getByText('new.png')).toBeTruthy())
  expect(listAssets).toHaveBeenCalledTimes(2)
})

it('reloads nothing when another kind moves, or assets move in another workspace', async () => {
  await renderAssets()
  feed.move('default', 'prompts')
  feed.move('other', 'assets')
  for (let i = 0; i < 3; i++) await new Promise((r) => setTimeout(r, 10))
  expect(listAssets).toHaveBeenCalledTimes(1)
})

it('a quiet reload that fails keeps the list and shows no error', async () => {
  await renderAssets()
  listAssets.mockImplementationOnce(() => Promise.reject(new Error('boom')))
  feed.move('default', 'assets')
  await waitFor(() => expect(listAssets).toHaveBeenCalledTimes(2))
  for (let i = 0; i < 3; i++) await new Promise((r) => setTimeout(r, 10))
  expect(screen.getByText('iris.png')).toBeTruthy()
  expect(screen.queryByText(/boom/)).toBeNull()
})

it('keeps the picks across a quiet reload, even for an asset briefly missing', async () => {
  await renderAssets()
  screen.getByRole('button', { name: /select all \(2\)/i }).click()
  await waitFor(() => expect(screen.getByText('2 selected')).toBeTruthy())

  // New asset arrives, one picked asset is briefly omitted
  listing.assets = [asset('iris.png'), asset('new.png')]
  feed.move('default', 'assets')
  await waitFor(() => expect(screen.getByText('new.png')).toBeTruthy())
  expect(screen.getByText('1 selected')).toBeTruthy()

  // It comes back on the next move, still picked
  listing.assets = [
    asset('iris.png'),
    asset('cast/priya.jpg'),
    asset('new.png'),
  ]
  feed.move('default', 'assets')
  await waitFor(() => expect(screen.getByText('2 selected')).toBeTruthy())
})

it('shows every asset, including one in a folder', async () => {
  await renderAssets()

  expect(screen.getByText('iris.png')).toBeTruthy()
  expect(screen.getByText('priya.jpg')).toBeTruthy()
})

it('an asset opens in the viewer with its reference and a download', async () => {
  await renderAssets()

  const sheet = await openAsset('iris.png')

  expect(sheet).toHaveTextContent('1 of 2')
  expect(hashOpen()).toBe('iris.png')
  const ref = within(sheet).getByText('asset:iris.png')
  expect(ref.closest('.refchip')).not.toBeNull()
  expect(
    within(sheet).getByTitle('Copy the reference a workflow argument takes'),
  ).toBeTruthy()
  const download = within(sheet).getByRole('link', { name: 'Download' })
  expect(download.getAttribute('href')).toBe('/inputs/iris.png')
  expect(download.hasAttribute('download')).toBe(true)
  // the asset's kind, size and date, as the old detail said them
  expect(sheet).toHaveTextContent('image · 2 KB')
  // the viewer walks the open library in the grid's order
  await step('Next file', 'cast/priya.jpg')
  expect(
    within(await viewer('priya.jpg')).getByText('asset:cast/priya.jpg'),
  ).toBeTruthy()
})

it('Back closes the viewer, and a Close on a link replaces open= away', async () => {
  await renderAssets()
  const before = history.length
  await openAsset('iris.png')
  // a pushed entry, so Back closes it
  expect(history.length).toBe(before + 1)
  await hashEvent(() => history.back())
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  expect(hashOpen()).toBeNull()

  cleanup()
  await hashEvent(
    () => (location.hash = '#/ws/default/assets?open=cast%2Fpriya.jpg'),
  )
  await renderAssets()
  expect(await viewer('priya.jpg')).toHaveTextContent('2 of 2')
  // a deep link has no pushed entry, so Close replaces open= away
  // (replaceState fires no hashchange, so there is nothing to wait for)
  await fireEvent.click(screen.getByRole('button', { name: 'Close' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  expect(location.hash).toBe('#/ws/default/assets')
})

it('open= survives Back', async () => {
  await renderAssets()
  await openAsset('cast/priya.jpg')
  // leave for another page, then come Back to the open asset
  cleanup()
  await hashEvent(() => (location.hash = '#/ws/default/jobs'))
  await hashEvent(() => history.back())
  expect(hashOpen()).toBe('cast/priya.jpg')
  await renderAssets()
  expect(await viewer('priya.jpg')).toHaveTextContent('2 of 2')
  // the entry the viewer pushed: Close goes Back, onto the page alone
  await hashEvent(
    () => void fireEvent.click(screen.getByRole('button', { name: 'Close' })),
  )
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  expect(location.hash).toBe('#/ws/default/assets')
})

it('an open= naming a missing asset says so', async () => {
  location.hash = '#/ws/default/assets?open=gone%2Fold.png'
  await renderAssets()
  const notice = await screen.findByText(
    'old.png is not in this view any more.',
  )
  expect(screen.queryByRole('dialog')).toBeNull()
  await waitFor(() => expect(hashOpen()).toBeNull())
  await fireEvent.click(within(notice).getByRole('button', { name: 'Dismiss' }))
  expect(screen.queryByText(/is not in this view any more/)).toBeNull()
})

it('open= under a q= that filters it out is said, and unsaid when the filter goes', async () => {
  location.hash = '#/ws/default/assets?q=iris&open=cast%2Fpriya.jpg'
  await renderAssets()
  await screen.findByText('priya.jpg is not in this view any more.')
  expect(screen.queryByRole('dialog')).toBeNull()
  await waitFor(() => expect(hashOpen()).toBeNull())
  // the filter is the person's: kept
  expect(new URLSearchParams(location.hash.split('?')[1]).get('q')).toBe('iris')
  await fireEvent.input(screen.getByPlaceholderText('filter…'), {
    target: { value: '' },
  })
  await waitFor(() =>
    expect(screen.queryByText(/is not in this view any more/)).toBeNull(),
  )
})

it('the durations read asks for only the newest 200 assets', async () => {
  await renderAssets()
  await waitFor(() => expect(assetDurations).toHaveBeenCalled())
  expect(assetDurations).toHaveBeenCalledWith({ media: true, limit: 200 })
})

it('an open= naming an asset in a shut folder opens it and the folder', async () => {
  localStorage.setItem(
    'dw-collapsed-asset-folders-workspace',
    JSON.stringify({ cast: true }),
  )
  location.hash = '#/ws/default/assets?open=cast%2Fpriya.jpg'
  await renderAssets()
  expect(await viewer('priya.jpg')).toHaveTextContent('2 of 2')
  expect(screen.getByLabelText('Open cast/priya.jpg')).toBeTruthy()
})

it('says which job it was saved from', async () => {
  galleryMetadata.mockImplementation(() =>
    Promise.resolve({
      metadata: null,
      job: { id: 'j1', status: 'succeeded', workflow: 'portrait' },
      version: 3,
    }),
  )
  await renderAssets()
  const sheet = await openAsset('iris.png')
  const line = await within(sheet).findByText(/Saved from/)
  expect(line.textContent?.trim()).toBe('Saved from portrait v3')
  expect(galleryMetadata).toHaveBeenCalledWith('asset:iris.png')
})

it('says nothing about where it came from for an asset with no job on record', async () => {
  await renderAssets()
  const sheet = await openAsset('iris.png')
  await within(sheet).findByText('No recipe is saved with this file.')
  expect(within(sheet).queryByText(/Saved from/)).toBeNull()
})

it('shows the asset note', async () => {
  listing.assets = [{ ...asset('iris.png'), note: 'the hero face' }]
  await renderAssets()
  const sheet = await openAsset('iris.png')
  await within(sheet).findByText('Your note on this asset')
  // the tile carries it too, so look in the viewer's aside
  expect(
    within(sheet.querySelector('.asset-aside') as HTMLElement).getByText(
      'the hero face',
    ),
  ).toBeTruthy()
})

it("the viewer's aside says the kind, size and when the asset was written, exact, to the second on hover", async () => {
  await renderAssets()
  const sheet = await openAsset('iris.png')
  const aside = sheet.querySelector('.asset-aside') as HTMLElement
  const time = aside.querySelector('time') as HTMLElement
  expect(time.closest('p')?.textContent?.replace(/\s+/g, ' ').trim()).toBe(
    `image · 2 KB · ${formatMtime(1)}`,
  )
  expect(time.getAttribute('title')).toBe(formatExact(1))
})

it('the note and Used by show while the recipe read fails or never lands', async () => {
  galleryMetadata.mockRejectedValueOnce(new Error('boom'))
  await renderAssets()
  const sheet = await openAsset('iris.png')
  await within(sheet).findByText(/Couldn't load what made this file/)
  expect(within(sheet).getByRole('button', { name: /edit note/i })).toBeTruthy()
  expect(within(sheet).getByText('Used by', { exact: false })).toBeTruthy()
})

it('the note and Used by show while the recipe read is still pending', async () => {
  galleryMetadata.mockImplementation(() => new Promise(() => {}))
  await renderAssets()
  const sheet = await openAsset('iris.png')
  expect(within(sheet).getByRole('button', { name: /edit note/i })).toBeTruthy()
  expect(within(sheet).getByText('Used by', { exact: false })).toBeTruthy()
})

it('a failed recipe read says so, with Retry, rather than nothing', async () => {
  galleryMetadata.mockRejectedValueOnce(new Error('boom'))
  await renderAssets()
  const sheet = await openAsset('iris.png')
  await within(sheet).findByText(/Couldn't load what made this file/)
  galleryMetadata.mockImplementation(() =>
    Promise.resolve({ metadata: null, job: { id: 'j1', workflow: 'p' } }),
  )
  await fireEvent.click(within(sheet).getByRole('button', { name: 'Retry' }))
  await waitFor(() =>
    expect(within(sheet).getByText(/Saved from p/)).toBeTruthy(),
  )
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

it('opens a shut shadowed section while a filter matches inside it', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, SHARED_LIBRARY]
  listing.assets = [asset('iris.png')]
  listing.shadowed = [shadowedAsset('iris.png', 'common', 'workspace')]
  render(AssetsPage)
  const label =
    "shadowed by this workspace's iris.png - asset:iris.png resolves to that file"
  await waitFor(() => screen.getByTitle(label))

  screen.getByTitle('collapse the shadowed entries').click()
  await waitFor(() => expect(screen.queryByTitle(label)).toBeNull())

  const filter = screen.getByPlaceholderText('filter…') as HTMLInputElement
  filter.value = 'iris'
  filter.dispatchEvent(new Event('input', { bubbles: true }))
  await waitFor(() => expect(screen.getByTitle(label)).toBeTruthy())
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
  const sheet = await openAsset('iris.png')
  expect(
    within(sheet)
      .getByRole('button', { name: 'Delete' })
      .classList.contains('danger'),
  ).toBe(true)
  within(sheet).getByRole('button', { name: 'Delete' }).click()

  await answerConfirm(true)

  await waitFor(() => expect(api.deleteAsset).toHaveBeenCalledWith('iris.png'))
  // gone from the listing, the viewer steps on to the one left
  listing.assets = [asset('cast/priya.jpg')]
  feed.move('default', 'assets')
  expect(await viewer('priya.jpg')).toHaveTextContent('1 of 1')
})

it('deletes nothing when the confirmation is declined', async () => {
  await renderAssets()
  const sheet = await openAsset('iris.png')
  within(sheet).getByRole('button', { name: 'Delete' }).click()

  await answerConfirm(false)

  await new Promise((r) => setTimeout(r, 10))
  expect(api.deleteAsset).not.toHaveBeenCalled()
})

it('a read-only asset offers no delete', async () => {
  listing.libraries = [EXAMPLES_LIBRARY]
  listing.assets = [asset('demo/x.png', 'examples')]
  await renderAssets('x.png')

  const sheet = await openAsset('demo/x.png')

  const mark = within(sheet).getByTitle(
    'Read-only: an examples library brought it',
  )
  expect(mark.textContent?.trim()).toBe('Read-only')
  expect(within(sheet).queryByRole('button', { name: 'Delete' })).toBeNull()
  // the reference and the download are still there
  expect(within(sheet).getByText('asset:demo/x.png')).toBeTruthy()
  expect(within(sheet).getByRole('link', { name: 'Download' })).toBeTruthy()
})

it('filters by name', async () => {
  await renderAssets()
  const filter = screen.getByPlaceholderText('filter…') as HTMLInputElement
  filter.value = 'priya'
  filter.dispatchEvent(new Event('input', { bubbles: true }))

  await waitFor(() => expect(screen.queryByText('iris.png')).toBeNull())
  expect(screen.getByText('priya.jpg')).toBeTruthy()
})

it('refetches when the workspace changes, and watches only the new one', async () => {
  await renderAssets()
  expect(listAssets).toHaveBeenCalledTimes(1)

  workspace.current = 'other'

  await waitFor(() => expect(listAssets).toHaveBeenCalledTimes(2))
  feed.move('default', 'assets')
  for (let i = 0; i < 3; i++) await new Promise((r) => setTimeout(r, 10))
  expect(listAssets).toHaveBeenCalledTimes(2)
  feed.move('other', 'assets')
  await waitFor(() => expect(listAssets).toHaveBeenCalledTimes(3))
})

// Escape means "close the thing on top". A confirm dialog answers it
// itself, so the viewer must not also close underneath it
it('leaves the viewer open when Escape answers a confirm dialog', async () => {
  await renderAssets()
  const sheet = await openAsset('iris.png')

  within(sheet).getByRole('button', { name: 'Delete' }).click()
  await waitFor(() => expect(screen.getByRole('alertdialog')).toBeTruthy())

  pressEscape()

  await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
  expect(screen.getByRole('dialog', { name: 'iris.png' })).toBeTruthy()
})

// The gallery's bulk actions, for the input side: tick files in the grid,
// then download or delete the lot in one go

it('downloads the ticked assets as one zip', async () => {
  await renderAssets()

  screen.getByLabelText('select iris.png').click()
  screen.getByLabelText('select cast/priya.jpg').click()
  await waitFor(() =>
    screen.getByRole('button', { name: /download the selected/i }),
  )
  screen.getByRole('button', { name: /download the selected/i }).click()

  await waitFor(() => expect(api.archiveAssets).toHaveBeenCalledTimes(1))
  // The order the grid shows them in, not the order they were ticked
  expect(api.archiveAssets).toHaveBeenCalledWith(['iris.png', 'cast/priya.jpg'])
})

it('a double click on Download sends one request', async () => {
  await renderAssets()

  screen.getByLabelText('select iris.png').click()
  await waitFor(() =>
    screen.getByRole('button', { name: /download the selected/i }),
  )
  const download = screen.getByRole('button', {
    name: /download the selected/i,
  })
  // Both clicks land before the page renders the first one's busy state
  download.click()
  download.click()

  await waitFor(() => expect(api.archiveAssets).toHaveBeenCalled())
  await new Promise((r) => setTimeout(r, 20))
  expect(api.archiveAssets).toHaveBeenCalledTimes(1)
})

it('deletes every ticked asset and refetches once', async () => {
  await renderAssets()

  screen.getByLabelText('select iris.png').click()
  screen.getByLabelText('select cast/priya.jpg').click()
  await waitFor(() =>
    screen.getByRole('button', { name: /^delete the selected/i }),
  )
  screen.getByRole('button', { name: /^delete the selected/i }).click()
  await answerConfirm(true)

  await waitFor(() => expect(api.deleteAsset).toHaveBeenCalledTimes(2))
  expect(api.deleteAsset.mock.calls.map((c) => c[0])).toEqual([
    'iris.png',
    'cast/priya.jpg',
  ])
})

it('a double press deletes the picked once', async () => {
  await renderAssets()

  screen.getByLabelText('select iris.png').click()
  screen.getByLabelText('select cast/priya.jpg').click()
  const button = await screen.findByRole('button', {
    name: /^delete the selected/i,
  })
  button.click()
  button.click()
  await answerConfirm(true)

  await waitFor(() => expect(api.deleteAsset).toHaveBeenCalled())
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 10))
  expect(api.deleteAsset.mock.calls.map((c) => c[0])).toEqual([
    'iris.png',
    'cast/priya.jpg',
  ])
  expect(screen.queryByRole('alertdialog')).toBeNull()
})

// A read-only asset answers 403. Whatever could not go stays ticked, so a
// retry needs no re-ticking and the failure is visible rather than dropped
it('keeps an asset that would not delete in the selection', async () => {
  api.deleteAsset.mockImplementation((name: string) =>
    name === 'iris.png'
      ? Promise.reject(new Error('read-only'))
      : Promise.resolve(),
  )
  await renderAssets()

  screen.getByLabelText('select iris.png').click()
  screen.getByLabelText('select cast/priya.jpg').click()
  await waitFor(() =>
    screen.getByRole('button', { name: /^delete the selected/i }),
  )
  screen.getByRole('button', { name: /^delete the selected/i }).click()
  await answerConfirm(true)

  await waitFor(() => expect(notifyFailure).toHaveBeenCalled())
  // Why, through describeError's copy, with the names in Copy details
  const [doing, error, id, detail] = notifyFailure.mock.calls[0]
  expect(doing).toBe('delete 1 of 2 assets')
  expect((error as Error).message).toBe('read-only')
  expect(id).toBeUndefined()
  expect(detail).toBe('iris.png: Something went wrong')
  expect(notifyError).not.toHaveBeenCalled()
  expect(screen.getByText('1 selected')).toBeTruthy()
  expect(
    (screen.getByLabelText('select iris.png') as HTMLInputElement).checked,
  ).toBe(true)
})

// The viewer is on top of the grid, so it takes the first Escape; the
// selection behind it clears on the next
it('Escape closes the viewer, then clears the selection', async () => {
  await renderAssets()
  screen.getByLabelText('select iris.png').click()
  await waitFor(() => expect(screen.getByText('1 selected')).toBeTruthy())
  await openAsset('iris.png')

  // the viewer pushed its entry, so Escape goes Back
  await hashEvent(pressEscape)
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  expect(screen.getByText('1 selected')).toBeTruthy()

  pressEscape()
  await waitFor(() => expect(screen.queryByText('1 selected')).toBeNull())
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

  await waitFor(() => expect(api.uploadMedia).toHaveBeenCalledTimes(1))
  expect(api.uploadMedia).toHaveBeenCalledWith(file, 'kept.png', false, {
    createOnly: true,
  })
})

it('uploads into the shared library from the shared section', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, SHARED_LIBRARY]
  listing.assets = [asset('shared/logo.png', 'common')]
  await renderAssets('logo.png', true)

  screen.getByRole('button', { name: /Upload to shared/ }).click()
  const file = chooseFile()

  await waitFor(() => expect(api.uploadMedia).toHaveBeenCalledTimes(1))
  expect(api.uploadMedia).toHaveBeenCalledWith(file, 'kept.png', true, {
    createOnly: true,
  })
})

// Deleting a shared asset is not deleting a workspace one: it goes for
// every workspace under the root, and the confirm has to say so

it('says what a shared delete costs', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, SHARED_LIBRARY]
  listing.assets = [asset('shared/logo.png', 'common')]
  await renderAssets('logo.png')
  const sheet = await openAsset('shared/logo.png')
  within(sheet).getByRole('button', { name: 'Delete' }).click()

  const dialog = await waitFor(() => screen.getByRole('alertdialog'))
  expect(dialog.textContent).toContain('Delete asset:shared/logo.png?')
  expect(dialog.textContent).toContain(
    'This removes it on disk. It is in the shared asset library, so every ' +
      'workspace under this root loses it, and any workflow still carrying ' +
      'that reference stops loading.',
  )
  // Answer it: an unanswered confirm is module state the next render sees
  await answerConfirm(false)
})

it('says nothing about a shared library when nothing shared is picked', async () => {
  await renderAssets()

  screen.getByRole('button', { name: /select all \(2\)/i }).click()
  await waitFor(() =>
    screen.getByRole('button', { name: /^delete the selected/i }),
  )
  screen.getByRole('button', { name: /^delete the selected/i }).click()

  const dialog = await waitFor(() => screen.getByRole('alertdialog'))
  expect(dialog.textContent).toContain('Delete 2 assets?')
  expect(dialog.textContent).toContain(
    'This removes them on disk. Any workflow still carrying one of those ' +
      'references stops loading.',
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
  await waitFor(() =>
    screen.getByRole('button', { name: /^delete the selected/i }),
  )
  screen.getByRole('button', { name: /^delete the selected/i }).click()

  const dialog = await waitFor(() => screen.getByRole('alertdialog'))
  expect(dialog.textContent).toContain('Delete 3 assets?')
  expect(dialog.textContent).toContain(
    'This removes them on disk. 2 of them are in the shared asset library and go ' +
      'away for every workspace. Any workflow still carrying one of those ' +
      'references stops loading.',
  )
  expect(
    within(dialog)
      .getByRole('button', { name: /^delete$/i })
      .classList.contains('danger'),
  ).toBe(true)
  await answerConfirm(false)
})

it('the shared view hides the workspace section and makes shared upload the filled button', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, SHARED_LIBRARY, EXAMPLES_LIBRARY]
  listing.assets = [
    asset('iris.png'),
    asset('shared/logo.png', 'common'),
    asset('demo/x.png', 'examples'),
  ]
  render(AssetsPage, { shared: true })
  const upload = await screen.findByRole('button', {
    name: /upload to shared/i,
  })
  expect(screen.queryByText('This workspace')).toBeNull()
  expect(upload.classList.contains('quiet')).toBe(false)
  // The hint must not describe a workspace section the view does not show
  expect(screen.queryByText(/only this workspace's own section/)).toBeNull()
  expect(screen.getByText(/an examples library is read-only/)).toBeTruthy()
})

// A workspace's page shows the shared library only as far as the workspace's
// own workflows reach into it; the whole of it is the Shared assets page's

it("leaves out a shared asset none of the workspace's workflows use", async () => {
  listing.libraries = [WORKSPACE_LIBRARY, SHARED_LIBRARY]
  listing.assets = [
    asset('iris.png'),
    asset('cast/priya.png', 'common'),
    { ...asset('unused.png', 'common'), used_by: undefined },
  ]
  await renderAssets()

  expect(screen.getByText('priya.png')).toBeTruthy()
  expect(screen.queryByText('unused.png')).toBeNull()
  expect(screen.getByText(/Shared library, used here/)).toBeTruthy()
  // The rest are a link away, counted
  const link = screen.getByRole('link', { name: /shared/ })
  expect(link.getAttribute('href')).toContain('shared/assets')
  expect(link.textContent).toContain('2')
})

it('drops the shared section when the workspace uses nothing from it', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, SHARED_LIBRARY]
  listing.assets = [
    asset('iris.png'),
    { ...asset('unused.png', 'common'), used_by: undefined },
  ]
  await renderAssets()

  const labels = [...document.querySelectorAll('.library')].map(
    (node) => node.textContent,
  )
  expect(labels.some((label) => label?.includes('Shared library'))).toBe(false)
  expect(screen.queryByRole('button', { name: /Upload to shared/ })).toBeNull()
})

it('lists every shared asset on the Shared assets page', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, SHARED_LIBRARY]
  listing.assets = [
    asset('iris.png'),
    { ...asset('unused.png', 'common'), used_by: undefined },
  ]
  await renderAssets('unused.png', true)

  expect(screen.queryByText('iris.png')).toBeNull()
  expect(screen.getByRole('button', { name: /Upload to shared/ })).toBeTruthy()
})

it('names the workflows that use an asset beside it in the viewer', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, SHARED_LIBRARY]
  listing.assets = [
    { ...asset('cast/priya.png', 'common'), used_by: ['portrait', 'ep/two'] },
  ]
  await renderAssets('priya.png')
  const sheet = await openAsset('cast/priya.png')

  const link = await within(sheet).findByRole('link', { name: 'ep/two' })
  expect(link.getAttribute('href')).toContain('workflows/ep/two')
  expect(link.closest('p')?.textContent).toMatch(/^\s*Used by\s+portrait,/)
})

it('keeps a saved note on the asset when it is selected again', async () => {
  setNote.mockResolvedValue({
    target: 'asset:iris.png',
    note: 'hero shot',
    note_source: null,
  })
  await renderAssets()
  await openAsset('iris.png')
  await fireEvent.click(
    await screen.findByRole('button', { name: /edit note/i }),
  )
  await fireEvent.input(screen.getByLabelText('Your note on this asset'), {
    target: { value: 'hero shot' },
  })
  await fireEvent.click(screen.getByRole('button', { name: /save/i }))
  await waitFor(() => expect(screen.getByText('hero shot')).toBeTruthy())
  await step('Next file', 'cast/priya.jpg')
  await waitFor(() => expect(screen.queryByText('hero shot')).toBeNull())
  await step('Previous file', 'iris.png')
  await waitFor(() => expect(screen.getByText('hero shot')).toBeTruthy())
})

it('a drop on the shared page uploads to the shared library without a button click', async () => {
  listing.libraries = [WORKSPACE_LIBRARY, SHARED_LIBRARY]
  listing.assets = [asset('shared/logo.png', 'common')]
  await renderAssets('logo.png', true)

  const file = new File(['x'], 'dropped.png', { type: 'image/png' })
  const drop = new Event('drop', { bubbles: true, cancelable: true })
  Object.defineProperty(drop, 'dataTransfer', {
    value: { types: ['Files'], files: [file] },
  })
  document.querySelector('.dropzone')!.dispatchEvent(drop)

  await waitFor(() => expect(api.uploadMedia).toHaveBeenCalledTimes(1))
  expect(api.uploadMedia).toHaveBeenCalledWith(file, 'dropped.png', true, {
    createOnly: true,
  })
})

it("a dropped batch is uploaded under the files' names", async () => {
  await renderAssets()
  const a = new File(['x'], 'one.png', { type: 'image/png' })
  const b = new File(['x'], 'two.png', { type: 'image/png' })
  const drop = new Event('drop', { bubbles: true, cancelable: true })
  Object.defineProperty(drop, 'dataTransfer', {
    value: { types: ['Files'], files: [a, b] },
  })
  document.querySelector('.dropzone')!.dispatchEvent(drop)

  await waitFor(() => expect(api.uploadMedia).toHaveBeenCalledTimes(2))
  expect(api.uploadMedia).toHaveBeenCalledWith(a, 'one.png', false, {
    createOnly: true,
  })
  expect(api.uploadMedia).toHaveBeenCalledWith(b, 'two.png', false, {
    createOnly: true,
  })
})

it('shows a skeleton while loading, not the empty state', async () => {
  listAssets.mockImplementationOnce(() => new Promise(() => {}))
  render(AssetsPage, { shared: false })
  expect(
    screen.getByRole('status', { name: 'loading the asset library' }),
  ).toBeTruthy()
  expect(screen.queryByText(/Nothing in the asset library/)).toBeNull()
})

it('a failed first load offers Retry and Retry loads', async () => {
  listAssets.mockImplementationOnce(() => Promise.reject(new Error('boom')))
  render(AssetsPage, { shared: false })
  await screen.findByText(/Couldn't load the asset library/)
  await fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await waitFor(() => expect(screen.getByText('iris.png')).toBeTruthy())
})

// Where an asset came from, as the viewer's aside says it
const keptFrom = (workflow: string) => ({
  metadata: null,
  job: { id: 'j1', workflow },
  version: null,
})

it('switching assets never shows the previous asset provenance', async () => {
  galleryMetadata.mockImplementation((reference) =>
    reference === 'asset:iris.png'
      ? Promise.resolve(keptFrom('paint'))
      : new Promise(() => {}),
  )
  await renderAssets()
  await openAsset('iris.png')
  await screen.findByText('Saved from paint')
  await step('Next file', 'cast/priya.jpg')
  await waitFor(() =>
    expect(screen.getByText('asset:cast/priya.jpg')).toBeTruthy(),
  )
  expect(screen.queryByText('Saved from paint')).toBeNull()
})

it("a failed recipe read for the next asset is a load failure, not a refresh line over the last one's", async () => {
  galleryMetadata.mockImplementation((reference) =>
    reference === 'asset:iris.png'
      ? Promise.resolve(keptFrom('paint'))
      : Promise.reject(new Error('boom')),
  )
  await renderAssets()
  await openAsset('iris.png')
  await screen.findByText('Saved from paint')
  await step('Next file', 'cast/priya.jpg')
  await screen.findByText(/Couldn't load what made this file/)
  expect(screen.queryByText(/Couldn't refresh/)).toBeNull()
  expect(screen.queryByText('Saved from paint')).toBeNull()
})

async function saveIrisNote(note: string) {
  setNote.mockResolvedValue({
    target: 'asset:iris.png',
    note,
    note_source: null,
  })
  await fireEvent.click(
    await screen.findByRole('button', { name: /edit note/i }),
  )
  await fireEvent.input(screen.getByLabelText('Your note on this asset'), {
    target: { value: note },
  })
  await fireEvent.click(screen.getByRole('button', { name: /save/i }))
  await waitFor(() => expect(screen.getByText(note)).toBeTruthy())
}

it('saving a note does not read the recipe again', async () => {
  galleryMetadata.mockImplementation(() => Promise.resolve(keptFrom('paint')))
  await renderAssets()
  await openAsset('iris.png')
  await screen.findByText('Saved from paint')
  const reads = galleryMetadata.mock.calls.length
  await saveIrisNote('hero shot')
  for (let i = 0; i < 3; i++) await new Promise((r) => setTimeout(r, 10))
  expect(galleryMetadata.mock.calls.length).toBe(reads)
})

it('a listing asked for before a note save cannot drop the note', async () => {
  await renderAssets()
  await openAsset('iris.png')
  // A read starts, and is held until after the save
  let answer: (value: unknown) => void = () => {}
  listAssets.mockImplementationOnce(
    () => new Promise((resolve) => (answer = resolve as never)),
  )
  feed.move('default', 'assets')
  await waitFor(() => expect(listAssets).toHaveBeenCalledTimes(2))
  await saveIrisNote('hero shot')
  // It lands carrying the note from before the save
  answer({ ...listing, folders: [] })
  for (let i = 0; i < 3; i++) await new Promise((r) => setTimeout(r, 10))
  await step('Next file', 'cast/priya.jpg')
  await waitFor(() => expect(screen.queryByText('hero shot')).toBeNull())
  await step('Previous file', 'iris.png')
  await waitFor(() => expect(screen.getByText('hero shot')).toBeTruthy())
})

it('a listing asked for after a note save shows a note changed elsewhere', async () => {
  await renderAssets()
  await openAsset('iris.png')
  await saveIrisNote('hero shot')
  listing.assets = [
    { ...asset('iris.png'), note: 'from an agent' },
    asset('cast/priya.jpg'),
  ]
  feed.move('default', 'assets')
  await waitFor(() => expect(listAssets).toHaveBeenCalledTimes(2))
  for (let i = 0; i < 3; i++) await new Promise((r) => setTimeout(r, 10))
  await step('Next file', 'cast/priya.jpg')
  await waitFor(() => expect(screen.queryByText('hero shot')).toBeNull())
  await step('Previous file', 'iris.png')
  await waitFor(() => expect(screen.getByText('from an agent')).toBeTruthy())
})

it("the user's own note is not marked as another writer's change", async () => {
  await renderAssets()
  const tile = (name: string) =>
    screen.getByLabelText(`Open ${name}`).closest('.cellwrap') as HTMLElement
  await openAsset('iris.png')
  await saveIrisNote('hero shot')
  // the save's own move of the feed: the server now carries the note, and
  // an agent changed priya's
  listing.assets = [
    { ...asset('iris.png'), note: 'hero shot' },
    { ...asset('cast/priya.jpg'), note: 'from an agent' },
  ]
  feed.move('default', 'assets')
  await waitFor(() =>
    expect(tile('cast/priya.jpg').classList).toContain('row-changed'),
  )
  expect(tile('iris.png').classList).not.toContain('row-changed')
})

it('the asset grid shows thumbnails, never the original or a video', async () => {
  listing.assets = [
    asset('iris.png'),
    asset('cast/walk.mp4', 'workspace', 'video'),
  ]
  const { container } = render(AssetsPage)
  await waitFor(() => expect(container.querySelectorAll('img').length).toBe(2))
  expect(container.querySelector('video')).toBeNull()
  const srcs = [...container.querySelectorAll('img')].map((img) =>
    img.getAttribute('src'),
  )
  for (const src of srcs) {
    expect(src).toMatch(/^\/api\/gallery\/asset(:|%3A).*\/thumbnail/)
    expect(src).not.toContain('/inputs/')
  }
  expect(container.querySelector('.videobadge')).not.toBeNull()
})

it('asset clips and sounds show their length, from one background read', async () => {
  listing.assets = [
    asset('iris.png'),
    asset('cast/walk.mp4', 'workspace', 'video'),
    asset('hum.wav', 'workspace', 'audio'),
  ]
  assetDurations.mockResolvedValueOnce({
    assets: [
      { ...asset('cast/walk.mp4', 'workspace', 'video'), duration_seconds: 3 },
      { ...asset('hum.wav', 'workspace', 'audio'), duration_seconds: 2.5 },
    ],
  })
  const { container } = render(AssetsPage)
  await waitFor(() =>
    expect(container.querySelector('.tile-time')?.textContent).toBe('3.0s'),
  )
  expect(container.querySelector('.tile-length')?.textContent).toBe('2.5s')
  expect(assetDurations).toHaveBeenCalledTimes(1)
  expect(listAssets).toHaveBeenCalledTimes(1)
})

it('an empty asset library says where to go', async () => {
  listing.assets = []
  render(AssetsPage, { shared: false })
  const link = await screen.findByRole('link', { name: 'Gallery' })
  expect(link.getAttribute('href')).toMatch(/^#\/ws\/[^/]+\/gallery$/)
})

const tileOf = (name: string) =>
  screen.getByLabelText(`Open ${name}`).closest('.cellwrap') as HTMLElement

it("an upload marks nothing and keeps an agent's fresh mark", async () => {
  await renderAssets()
  // An agent changed priya's tile: it is marked
  listing.assets = [
    asset('iris.png'),
    { ...asset('cast/priya.jpg'), size: 4096 },
  ]
  feed.move('default', 'assets')
  await waitFor(() =>
    expect(tileOf('cast/priya.jpg').classList).toContain('row-changed'),
  )
  // The page's own upload: the server now lists it
  api.uploadMedia.mockResolvedValueOnce({ reference: 'asset:uploads/kept.png' })
  listing.assets = [...listing.assets, asset('uploads/kept.png')]
  screen.getByRole('button', { name: /^Upload$/ }).click()
  chooseFile()
  await waitFor(() => expect(screen.getByText('kept.png')).toBeTruthy())
  // Held as read: the page never went loud, and nothing else was marked
  expect(screen.queryByLabelText(/loading/i)).toBeNull()
  expect(tileOf('uploads/kept.png').classList).not.toContain('row-changed')
  expect(tileOf('cast/priya.jpg').classList).toContain('row-changed')
  // The feed's echo of that write reads the same listing: still unmarked
  feed.move('default', 'assets')
  await waitFor(() => expect(listAssets).toHaveBeenCalledTimes(4))
  await new Promise((r) => setTimeout(r, 10))
  expect(tileOf('uploads/kept.png').classList).not.toContain('row-changed')
})

it('a bulk delete adopts', async () => {
  await renderAssets()
  const reads = listAssets.mock.calls.length
  screen.getByLabelText('select iris.png').click()
  screen.getByLabelText('select cast/priya.jpg').click()
  await waitFor(() =>
    screen.getByRole('button', { name: /^delete the selected/i }),
  )
  screen.getByRole('button', { name: /^delete the selected/i }).click()
  await answerConfirm(true)
  await waitFor(() => expect(screen.queryByText('iris.png')).toBeNull())
  expect(screen.queryByText('priya.jpg')).toBeNull()
  // The held list lost what went; no listing was read for it
  await new Promise((r) => setTimeout(r, 10))
  expect(listAssets.mock.calls.length).toBe(reads)
})

const tileNames = () =>
  [...document.querySelectorAll('.cellwrap .tile-caption')].map(
    (c) => c.textContent,
  )

it('AssetsPage says when each asset changed', async () => {
  const now = Math.floor(Date.now() / 1000)
  listing.assets = [{ ...asset('iris.png'), mtime: now - 3 * 86_400 }]
  await renderAssets()
  const stamp = screen
    .getByRole('button', { name: 'Open iris.png' })
    .querySelector('.tile-meta time')!
  expect(stamp.textContent?.trim()).toBe('changed 3 days ago')
  expect(stamp.getAttribute('title')).toBe(formatExact(now - 3 * 86_400))
})

it('the newest-first sort', async () => {
  const now = Math.floor(Date.now() / 1000)
  listing.assets = [
    { ...asset('a.png'), mtime: now - 86_400 },
    { ...asset('b.png'), mtime: now - 60 },
    { ...asset('c.png'), mtime: now - 3600 },
  ]
  await renderAssets('a.png')
  expect(tileNames()).toEqual(['a.png', 'b.png', 'c.png'])
  await fireEvent.click(screen.getByRole('button', { name: 'Newest' }))
  await waitFor(() => expect(tileNames()).toEqual(['b.png', 'c.png', 'a.png']))
  expect(location.hash).toContain('sort=newest')
})

it('Newest names a foldered asset by its leaf, its folder a chip', async () => {
  const now = Math.floor(Date.now() / 1000)
  listing.assets = [
    { ...asset('cast/priya.jpg'), mtime: now - 60 },
    { ...asset('people/edit.png'), mtime: now - 120 },
    { ...asset('pets/edit.png'), mtime: now - 3600 },
    { ...asset('iris.png'), mtime: now - 86_400 },
  ]
  await renderAssets()
  const chips = () =>
    [...document.querySelectorAll('.cellwrap .cardfolder')].map((c) =>
      c.textContent?.trim(),
    )
  // in folders, the heading says the folder: no chip
  expect(chips()).toEqual([])
  await fireEvent.click(screen.getByRole('button', { name: 'Newest' }))
  // a leaf no other asset shares is the leaf; two that collide stay
  // distinct, as viewLabels has them
  await waitFor(() =>
    expect(tileNames()).toEqual([
      'priya.jpg',
      'people/edit.png',
      'pets/edit.png',
      'iris.png',
    ]),
  )
  expect(chips()).toEqual(['cast', 'people', 'pets'])
})
