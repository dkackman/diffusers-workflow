import { fireEvent, render } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import MediaArgumentInput from './MediaArgumentInput.svelte'

const uploadMedia = vi.fn()
vi.mock('../api', () => ({ api: { uploadMedia: (f: File) => uploadMedia(f) } }))

const originals = {
  create: URL.createObjectURL,
  revoke: URL.revokeObjectURL,
}
afterEach(() => {
  URL.createObjectURL = originals.create
  URL.revokeObjectURL = originals.revoke
})

it('revokes the local preview blob URL when the component is destroyed', async () => {
  const createObjectURL = vi.fn(() => 'blob:preview-1')
  const revokeObjectURL = vi.fn()
  URL.createObjectURL = createObjectURL
  URL.revokeObjectURL = revokeObjectURL
  uploadMedia.mockRejectedValueOnce(new Error('upload failed'))

  const { container, unmount } = render(MediaArgumentInput, {
    id: 'arg-image',
    kind: 'image',
    location: '',
    onchange: () => {},
  })
  const input = container.querySelector(
    'input[type="file"]',
  ) as HTMLInputElement
  const file = new File(['x'], 'cat.png', { type: 'image/png' })
  Object.defineProperty(input, 'files', { value: [file] })
  await fireEvent.change(input)
  await new Promise((r) => setTimeout(r, 0))

  expect(createObjectURL).toHaveBeenCalledTimes(1)
  expect(revokeObjectURL).not.toHaveBeenCalled()
  unmount()
  expect(revokeObjectURL).toHaveBeenCalledWith('blob:preview-1')
}, 20000)

it('previews the file an asset: reference names', async () => {
  const { assetLibrary } = await import('../assetlib.svelte')
  const { workspace } = await import('../workspace.svelte')
  assetLibrary.workspace = workspace.current
  assetLibrary.items = [
    {
      name: 'cast/priya.png',
      reference: 'asset:cast/priya.png',
      folder: 'cast',
      kind: 'image',
      size: 1,
      mtime: 1,
      origin: 'common',
      writable: true,
      url: '/inputs/cast/priya.png',
    },
  ]
  const { container } = render(MediaArgumentInput, {
    id: 'arg-image',
    kind: 'image',
    location: 'asset:cast/priya.png',
    onchange: () => {},
  })
  expect(container.querySelector('.preview img')?.getAttribute('src')).toBe(
    '/inputs/cast/priya.png',
  )
})
