import { render } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
// Hoisted above the imports so the static import of the component below -
// itself hoisted - sees an initialized mock. Importing the component inside
// the test instead would charge its (multi-second) compile to the test timeout
import GalleryPage from './GalleryPage.svelte'

// Resolves on a macrotask so that a re-triggering effect cannot starve the
// timers this test waits on - the failure then shows as a call count
const gallery = vi.hoisted(() =>
  vi.fn(
    () =>
      new Promise<{ files: never[] }>((resolve) =>
        setTimeout(() => resolve({ files: [] }), 0),
      ),
  ),
)
vi.mock('../api', () => ({
  api: {
    gallery: () => gallery(),
    galleryThumbnailUrl: (name: string) => `/thumb/${name}`,
    galleryUrl: (name: string) => `/outputs/${name}`,
    deleteOutput: vi.fn(),
    galleryMetadata: vi.fn(),
  },
}))

afterEach(() => gallery.mockClear())

it('fetches the gallery listing exactly once on mount', async () => {
  render(GalleryPage)
  // Let the request settle and any (wrongly) re-triggered effects run
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 10))
  expect(gallery).toHaveBeenCalledTimes(1)
})
