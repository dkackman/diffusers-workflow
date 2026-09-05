import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
// Hoisted above the imports so the static import of the component below -
// itself hoisted - sees an initialized mock. Importing the component inside
// the test instead would charge its (multi-second) compile to the test timeout
import GalleryPage from './GalleryPage.svelte'
import type { GalleryFile } from '../types'

const file = (name: string): GalleryFile => ({
  name,
  folder: name.includes('/') ? name.split('/')[0] : '',
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
  listing.files = ['a.png', 'b.png', 'demo/c.png'].map(file)
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
})

/** The grid's per-file selection checkbox. */
const checkbox = (name: string) =>
  screen.getByRole('checkbox', { name: `select ${name}` })

async function renderGallery(first = 'a.png') {
  render(GalleryPage)
  await waitFor(() =>
    expect(screen.getByLabelText(`select ${first}`)).toBeTruthy(),
  )
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
  listing.files = ['keep-a.png', 'keep-b.png', 'other.png'].map(file)
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
  vi.stubGlobal('confirm', () => true)
  await renderGallery()

  checkbox('a.png').click()
  checkbox('b.png').click()
  await waitFor(() => expect(screen.getByText('2 selected')).toBeTruthy())
  screen.getByRole('button', { name: /^delete/i }).click()

  await waitFor(() =>
    expect(screen.queryByLabelText('select a.png')).toBeNull(),
  )
  expect(screen.queryByLabelText('select b.png')).toBeNull()
  expect(screen.queryByText(/selected/)).toBeNull()
  expect(deleteOutput).toHaveBeenCalledTimes(2)
})

it('keeps a file that failed to delete selected and reports it', async () => {
  vi.stubGlobal('confirm', () => true)
  deleteOutput.mockImplementation((name: string) =>
    name === 'b.png' ? Promise.reject(new Error('busy')) : Promise.resolve(),
  )
  await renderGallery()

  checkbox('a.png').click()
  checkbox('b.png').click()
  await waitFor(() => expect(screen.getByText('2 selected')).toBeTruthy())
  screen.getByRole('button', { name: /^delete/i }).click()

  await waitFor(() => expect(screen.getByText('1 selected')).toBeTruthy())
  expect(screen.queryByLabelText('select a.png')).toBeNull()
  expect(screen.getByLabelText('select b.png')).toBeTruthy()
  expect(notifyError).toHaveBeenCalled()
})

it('does not delete anything when the confirmation is declined', async () => {
  vi.stubGlobal('confirm', () => false)
  await renderGallery()

  checkbox('a.png').click()
  await waitFor(() => expect(screen.getByText('1 selected')).toBeTruthy())
  screen.getByRole('button', { name: /^delete/i }).click()

  await new Promise((r) => setTimeout(r, 10))
  expect(deleteOutput).not.toHaveBeenCalled()
})

it('drops a file deleted from the detail panel out of the selection', async () => {
  vi.stubGlobal('confirm', () => true)
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

  await waitFor(() => expect(screen.getByText('1 selected')).toBeTruthy())
  expect(screen.getByLabelText('select b.png')).toBeTruthy()
})
