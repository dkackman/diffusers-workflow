import { beforeEach, expect, it, vi } from 'vitest'
import { ApiError } from '../apiErrors'
import { isUploadable, uploadAll } from './uploads'

const uploadMedia = vi.hoisted(() => vi.fn())
vi.mock('../api', () => ({
  api: { uploadMedia: (...args: unknown[]) => uploadMedia(...args) },
}))
const chooseDialog = vi.hoisted(() => vi.fn())
vi.mock('../confirm.svelte', () => ({
  chooseDialog: (...args: unknown[]) => chooseDialog(...args),
}))
const notifyError = vi.hoisted(() => vi.fn())
vi.mock('../toast', () => ({ notify: { error: notifyError } }))

const file = (name: string) => new File(['x'], name)
const exists = () => new ApiError('exists', 409, 'exists')
const ok = (name: string) => ({ reference: `asset:${name}`, url: `/u/${name}` })

beforeEach(() => {
  uploadMedia.mockReset()
  chooseDialog.mockReset()
  notifyError.mockReset()
  uploadMedia.mockImplementation((f: File, name?: string) =>
    Promise.resolve(ok(name ?? f.name)),
  )
})

it('sends each file under its own name, create-only', async () => {
  const a = file('a.png')
  const b = file('b.png')
  const result = await uploadAll([a, b], { shared: true, folder: 'cast' })
  expect(uploadMedia).toHaveBeenCalledWith(a, 'cast/a.png', true, {
    createOnly: true,
  })
  expect(uploadMedia).toHaveBeenCalledWith(b, 'cast/b.png', true, {
    createOnly: true,
  })
  expect(result.uploaded).toHaveLength(2)
  expect(chooseDialog).not.toHaveBeenCalled()
})

it('asks once for a batch with existing names', async () => {
  uploadMedia.mockImplementation((f: File, name: string, _s, pre) =>
    pre?.createOnly && f.name !== 'c.png'
      ? Promise.reject(exists())
      : Promise.resolve(ok(name)),
  )
  chooseDialog.mockResolvedValue('replace')
  const result = await uploadAll(
    [file('a.png'), file('b.png'), file('c.png')],
    { shared: false },
  )
  expect(chooseDialog).toHaveBeenCalledTimes(1)
  expect(chooseDialog.mock.calls[0][0]).toBe(
    '2 of 3 files already exist in the library.',
  )
  const labels = chooseDialog.mock.calls[0][1].map(
    (c: { label: string }) => c.label,
  )
  expect(labels).toEqual(['Replace them', 'Keep both', 'Skip them'])
  expect(result.uploaded).toHaveLength(3)
  // Replaced without a precondition
  expect(uploadMedia).toHaveBeenCalledWith(
    expect.any(File),
    'a.png',
    false,
    undefined,
  )
})

it('keep both renames with freeName', async () => {
  uploadMedia.mockImplementation((f: File, name: string, _s, pre) =>
    pre?.createOnly && name === 'photo.jpg'
      ? Promise.reject(exists())
      : Promise.resolve(ok(name)),
  )
  chooseDialog.mockResolvedValue('keep-both')
  const result = await uploadAll([file('photo.jpg')], {
    shared: false,
    taken: new Set(['photo.jpg', 'photo-2.jpg']),
  })
  expect(uploadMedia).toHaveBeenLastCalledWith(
    expect.any(File),
    'photo-3.jpg',
    false,
    { createOnly: true },
  )
  expect(result.uploaded).toHaveLength(1)
})

it('keep both retries when the chosen name is taken too', async () => {
  uploadMedia.mockImplementation((f: File, name: string, _s, pre) =>
    pre?.createOnly && name !== 'p-3.jpg'
      ? Promise.reject(exists())
      : Promise.resolve(ok(name)),
  )
  chooseDialog.mockResolvedValue('keep-both')
  const result = await uploadAll([file('p.jpg')], { shared: false })
  expect(result.uploaded).toHaveLength(1)
  expect(result.failed).toEqual([])
})

it('skip leaves existing assets untouched', async () => {
  uploadMedia.mockRejectedValue(exists())
  chooseDialog.mockResolvedValue('skip')
  const result = await uploadAll([file('a.png')], { shared: false })
  expect(uploadMedia).toHaveBeenCalledTimes(1)
  expect(result.skipped).toEqual(['a.png'])
  expect(result.uploaded).toEqual([])
})

it('a dismissed dialog skips', async () => {
  uploadMedia.mockRejectedValue(exists())
  chooseDialog.mockResolvedValue(null)
  const result = await uploadAll([file('a.png')], { shared: false })
  expect(result.skipped).toEqual(['a.png'])
  expect(uploadMedia).toHaveBeenCalledTimes(1)
})

it('unsupported files are not sent', async () => {
  const result = await uploadAll([file('notes.txt'), file('a.PNG')], {
    shared: false,
  })
  expect(uploadMedia).toHaveBeenCalledTimes(1)
  expect(notifyError).toHaveBeenCalledWith(
    'Skipped notes.txt: not an image, video, audio or LUT file',
  )
  expect(result.uploaded).toHaveLength(1)
  expect(isUploadable(file('x.cube'))).toBe(true)
  expect(isUploadable(file('x.json'))).toBe(false)
})

it('other failures are reported per file', async () => {
  uploadMedia.mockRejectedValue(new Error('boom'))
  const result = await uploadAll([file('a.png')], { shared: false })
  expect(result.failed).toHaveLength(1)
  expect(result.failed[0].name).toBe('a.png')
})
