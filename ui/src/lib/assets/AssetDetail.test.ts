import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
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
  await waitFor(() => expect(screen.getByText(/kept from/)).toBeTruthy())
  expect(screen.getByText(/kept from/).textContent?.trim()).toBe(
    'kept from portrait v3',
  )
  expect(galleryMetadata).toHaveBeenCalledWith('asset:face.png')
})

it('says nothing for an asset with no recorded provenance', async () => {
  provenance.response = { job: null, version: null }
  render(AssetDetail, props)
  await waitFor(() => expect(galleryMetadata).toHaveBeenCalled())
  await new Promise((r) => setTimeout(r, 0))
  expect(screen.queryByText(/kept from/)).toBeNull()
})
