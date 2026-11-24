import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import AssetDetail from './AssetDetail.svelte'
import type { AssetFile } from '../types'

const provenance = vi.hoisted(() => ({
  response: null as Record<string, unknown> | null,
}))
const galleryMetadata = vi.hoisted(() =>
  vi.fn<(name: string) => Promise<Record<string, unknown> | null>>(() =>
    Promise.resolve(provenance.response),
  ),
)
vi.mock('../api', () => ({
  api: { galleryMetadata: (name: string) => galleryMetadata(name) },
}))

vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))

afterEach(() => {
  cleanup()
  galleryMetadata.mockClear()
})

const asset: AssetFile = {
  name: 'face.png',
  folder: '',
  reference: 'asset:face.png',
  kind: 'image',
  size: 10,
  mtime: 1,
  url: '/assets/face.png',
  origin: 'workspace',
  writable: true,
}

const props = { asset, busy: false, onremove: () => {}, onclose: () => {} }

it('says which workflow and version a kept asset came from', async () => {
  provenance.response = {
    job: { id: 'j1', status: 'succeeded', workflow: 'portrait' },
    version: 3,
  }
  render(AssetDetail, props)
  await waitFor(() => expect(screen.getByText(/Saved from/)).toBeTruthy())
  expect(screen.getByText(/Saved from/).textContent?.trim()).toBe(
    'Saved from portrait v3',
  )
  expect(galleryMetadata).toHaveBeenCalledWith('asset:face.png')
})

it('says nothing for an asset with no recorded provenance', async () => {
  provenance.response = { job: null, version: null }
  render(AssetDetail, props)
  await waitFor(() => expect(galleryMetadata).toHaveBeenCalled())
  await new Promise((r) => setTimeout(r, 0))
  expect(screen.queryByText(/Saved from/)).toBeNull()
})

it('shows the asset note', () => {
  render(AssetDetail, { ...props, asset: { ...asset, note: 'the hero face' } })
  expect(screen.getByText('the hero face')).toBeTruthy()
  expect(screen.getByText('Your note on this asset')).toBeTruthy()
})

it('a failed provenance read says so, with Retry, rather than nothing', async () => {
  galleryMetadata.mockRejectedValueOnce(new Error('boom'))
  render(AssetDetail, props)
  await screen.findByText(/Couldn't load where this asset came from/)
  provenance.response = { job: { id: 'j1', status: 'ok', workflow: 'p' } }
  await fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await waitFor(() => expect(screen.getByText(/Saved from p/)).toBeTruthy())
})
