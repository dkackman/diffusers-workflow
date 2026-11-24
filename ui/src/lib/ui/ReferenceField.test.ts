import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { createRawSnippet } from 'svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import ReferenceField from './ReferenceField.svelte'
import type { AssetFile } from '../types'

const uploadMedia = vi.fn()
vi.mock('../api', () => ({
  api: {
    uploadMedia: (f: File) => uploadMedia(f),
    // The listings are set by hand below; a fetch never lands
    listAssets: () => new Promise(() => {}),
    gallery: () => new Promise(() => {}),
  },
}))
const chooseDialog = vi.hoisted(() => vi.fn())
vi.mock('../confirm.svelte', async (orig) => ({
  ...(await orig<object>()),
  chooseDialog: (...args: unknown[]) => chooseDialog(...args),
}))
const notifySuccess = vi.hoisted(() => vi.fn())
vi.mock('../toast', async (orig) => {
  const mod = await orig<{ notify: object }>()
  return { ...mod, notify: { ...mod.notify, success: notifySuccess } }
})

const originals = {
  create: URL.createObjectURL,
  revoke: URL.revokeObjectURL,
}
beforeEach(async () => {
  uploadMedia.mockReset()
  notifySuccess.mockReset()
  const { assetLibrary } = await import('../assetlib.svelte')
  const { outputLibrary } = await import('../outputlib.svelte')
  const { workspace } = await import('../workspace.svelte')
  assetLibrary.workspace = workspace.current
  assetLibrary.items = [
    asset('cast/priya.png', 'image'),
    asset('clips/walk.mp4', 'video'),
  ]
  outputLibrary.workspace = workspace.current
  outputLibrary.items = []
})
afterEach(() => {
  cleanup()
  URL.createObjectURL = originals.create
  URL.revokeObjectURL = originals.revoke
})

function asset(name: string, kind: AssetFile['kind']): AssetFile {
  return {
    name,
    reference: `asset:${name}`,
    folder: name.split('/')[0],
    kind,
    size: 1,
    mtime: 1,
    origin: 'common',
    writable: true,
    url: `/inputs/${name}`,
  } as AssetFile
}

function field(props: Record<string, unknown> = {}) {
  return render(ReferenceField, {
    id: 'arg-image',
    kind: 'image',
    value: '',
    onchange: () => {},
    ...props,
  })
}

async function pickFile(container: HTMLElement, file: File) {
  const input = container.querySelector(
    'input[type="file"]',
  ) as HTMLInputElement
  Object.defineProperty(input, 'files', { value: [file], configurable: true })
  await fireEvent.change(input)
}

const png = () => new File(['x'], 'cat.png', { type: 'image/png' })
const tick = () => new Promise((r) => setTimeout(r, 0))

it('revokes the local preview blob URL when the component is destroyed', async () => {
  const createObjectURL = vi.fn(() => 'blob:preview-1')
  const revokeObjectURL = vi.fn()
  URL.createObjectURL = createObjectURL
  URL.revokeObjectURL = revokeObjectURL
  uploadMedia.mockRejectedValueOnce(new Error('upload failed'))

  const { container, unmount } = field()
  await pickFile(container, png())
  await tick()

  expect(createObjectURL).toHaveBeenCalledTimes(1)
  expect(revokeObjectURL).not.toHaveBeenCalled()
  unmount()
  expect(revokeObjectURL).toHaveBeenCalledWith('blob:preview-1')
}, 20000)

it('a failed upload says why in plain words, the server text folded behind Details', async () => {
  URL.createObjectURL = vi.fn(() => 'blob:preview-2')
  URL.revokeObjectURL = vi.fn()
  const { ApiError } = await import('../apiErrors')
  const raw = 'OSError: [Errno 28] No space left on device: /assets/cat.png'
  uploadMedia.mockRejectedValueOnce(new ApiError(raw, 500))
  const { container } = field()
  await pickFile(container, png())
  const alert = await screen.findByRole('alert')
  expect(alert.querySelector('.title')?.textContent).toBe(
    "Couldn't upload that file: The server hit an error",
  )
  expect(alert.querySelector('details')?.open).toBe(false)
  expect(alert.querySelector('details pre')?.textContent).toBe(raw)
  expect(container.textContent).toContain(
    'The path above can still be typed by hand.',
  )
}, 20000)

it('previews the file an asset: reference names', () => {
  const { container } = field({ value: 'asset:cast/priya.png' })
  const tile = container.querySelector('.preview .filetile.tile-strip')
  expect(tile?.querySelector('.tile-caption')?.textContent).toBe('priya.png')
  expect(tile?.querySelector('img.tile-pic')?.getAttribute('src')).toContain(
    'cast/priya.png',
  )
})

it('a skipped upload keeps the field value and the preview, and says so', async () => {
  let n = 0
  URL.createObjectURL = vi.fn(() => `blob:p-${++n}`)
  const revokeObjectURL = vi.fn()
  URL.revokeObjectURL = revokeObjectURL
  const { ApiError } = await import('../apiErrors')
  uploadMedia.mockRejectedValueOnce(new ApiError('exists', 409, 'exists'))
  chooseDialog.mockResolvedValueOnce('skip')
  const onchange = vi.fn()
  const { container } = field({ value: 'asset:cast/priya.png', onchange })
  await pickFile(container, png())
  await tick()
  expect(onchange).not.toHaveBeenCalled()
  expect(notifySuccess).toHaveBeenCalledWith('Skipped 1 existing file: cat.png')
  expect(revokeObjectURL).toHaveBeenCalledWith('blob:p-1')
  expect(
    container.querySelector('.preview img.tile-pic')?.getAttribute('src'),
  ).toContain('cast/priya.png')
})

it('a finished upload writes its reference into the field', async () => {
  URL.createObjectURL = vi.fn(() => 'blob:p')
  URL.revokeObjectURL = vi.fn()
  uploadMedia.mockResolvedValueOnce({
    reference: 'asset:uploads/cat.png',
    url: '/inputs/uploads/cat.png',
  })
  const onchange = vi.fn()
  const { container } = field({ onchange })
  await pickFile(container, png())
  await tick()
  expect(onchange).toHaveBeenCalledWith('asset:uploads/cat.png')
})

it('a reference to a missing file keeps its text and shows the glyph', async () => {
  const onchange = vi.fn()
  const { container } = field({ value: 'output:shots/r9/gone.png', onchange })
  const input = screen.getByRole('combobox') as HTMLInputElement
  expect(input.value).toBe('output:shots/r9/gone.png')
  const tile = container.querySelector('.preview .filetile')
  expect(tile?.querySelector('.tile-glyph')).not.toBeNull()
  expect(container.querySelector('img')).toBeNull()
  await tick()
  expect(onchange).not.toHaveBeenCalled()
  expect(input.value).toBe('output:shots/r9/gone.png')
})

it('a step result or a variable shows nothing and keeps its text', () => {
  const { container } = field({ value: 'previous_result:still' })
  expect(container.querySelector('.preview')).toBeNull()
  expect((screen.getByRole('combobox') as HTMLInputElement).value).toBe(
    'previous_result:still',
  )
})

it('a typed URL is kept as typed', async () => {
  const url = 'https://example.com/a%20cat.png?w=1'
  const onchange = vi.fn()
  const { container, rerender } = field({ onchange })
  const input = screen.getByRole('combobox') as HTMLInputElement
  await fireEvent.input(input, { target: { value: url } })
  await fireEvent.change(input, { target: { value: url } })
  expect(onchange).toHaveBeenLastCalledWith(url)
  await rerender({ value: url })
  expect(input.value).toBe(url)
  const img = container.querySelector('.preview img.tile-pic') as HTMLElement
  expect(img.getAttribute('src')).toBe(url)
  // A URL that serves no picture falls back to the glyph, the text untouched
  await fireEvent.error(img)
  expect(container.querySelector('.preview img')).toBeNull()
  expect(container.querySelector('.preview .tile-glyph')).not.toBeNull()
  expect(input.value).toBe(url)
  expect(onchange).toHaveBeenCalledTimes(1)
})

it('an upload that finishes after the field changed leaves it alone', async () => {
  URL.createObjectURL = vi.fn(() => 'blob:slow')
  URL.revokeObjectURL = vi.fn()
  let finish: (v: unknown) => void = () => {}
  uploadMedia.mockReturnValueOnce(new Promise((r) => (finish = r)))
  const onchange = vi.fn()
  const { container, rerender } = field({ value: 'asset:old.png', onchange })
  await pickFile(container, png())
  // Meanwhile the field is cleared
  await rerender({ value: '' })
  finish({ reference: 'asset:uploads/cat.png', url: '/inputs/uploads/cat.png' })
  await tick()
  await tick()
  expect(onchange).not.toHaveBeenCalled()
  expect(notifySuccess).toHaveBeenCalledWith(
    'Uploaded as asset:uploads/cat.png. The field changed meanwhile, so it was left as it is.',
  )
  expect(container.querySelector('.preview')).toBeNull()
})

it('an upload that finishes after the field is gone writes nothing', async () => {
  URL.createObjectURL = vi.fn(() => 'blob:slow')
  URL.revokeObjectURL = vi.fn()
  let finish: (v: unknown) => void = () => {}
  uploadMedia.mockReturnValueOnce(new Promise((r) => (finish = r)))
  const onchange = vi.fn()
  const { container, unmount } = field({ onchange })
  await pickFile(container, png())
  unmount()
  finish({ reference: 'asset:uploads/cat.png', url: '/inputs/uploads/cat.png' })
  await tick()
  await tick()
  expect(onchange).not.toHaveBeenCalled()
  expect(notifySuccess).toHaveBeenCalledOnce()
})

it('refuses a dropped file of the wrong kind', async () => {
  const { container } = field()
  const zone = container.querySelector('.reference') as HTMLElement
  const sound = new File(['x'], 'song.wav', { type: 'audio/wav' })
  await fireEvent.drop(zone, {
    dataTransfer: { types: ['Files'], files: [sound] },
  })
  expect(container.textContent).toContain("That isn't an image file.")
  expect(uploadMedia).not.toHaveBeenCalled()
  // Picked through the file input, likewise
  const other = field({ id: 'arg-video', kind: 'video' })
  await pickFile(other.container, png())
  expect(other.container.textContent).toContain("That isn't a video file.")
  expect(uploadMedia).not.toHaveBeenCalled()
})

it('the preview is a tile, never a raw img or video', () => {
  for (const value of [
    'asset:cast/priya.png',
    'asset:clips/walk.mp4',
    'https://example.com/cat.png',
    '/media/clip.mp4',
  ]) {
    const { container, unmount } = field({
      value,
      kind: value.endsWith('.mp4') ? 'video' : 'image',
    })
    expect(container.querySelector('video')).toBeNull()
    for (const img of container.querySelectorAll('img'))
      expect(img.closest('.filetile')).not.toBeNull()
    expect(container.querySelector('.preview .filetile')).not.toBeNull()
    unmount()
  }
})

it('names its buttons and placeholder for the kind, and renders its children after them', () => {
  const children = createRawSnippet(() => ({
    render: () => '<button type="button">Enhance…</button>',
  }))
  field({ kind: 'audio', children })
  const input = screen.getByRole('combobox')
  expect(input.getAttribute('placeholder')).toBe(
    'type, drop or choose an audio',
  )
  const choose = screen.getByRole('button', { name: 'Choose…' })
  expect(choose.getAttribute('title')).toBe(
    'Choose an audio from the assets or earlier jobs',
  )
  const upload = screen.getByRole('button', { name: 'Upload…' })
  expect(upload.getAttribute('title')).toBe(
    'Upload an audio from this computer, or drop one here',
  )
  const buttons = screen
    .getAllByRole('button')
    .map((b) => b.textContent?.trim())
  expect(buttons.slice(-3)).toEqual(['Choose…', 'Upload…', 'Enhance…'])
  expect(
    document.querySelector('input[type="file"]')?.getAttribute('accept'),
  ).toBe('audio/*')
})
