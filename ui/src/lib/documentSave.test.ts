import { beforeEach, expect, it, vi } from 'vitest'
import { saveDocument, freeName } from './documentSave'
import type { DocumentEditor } from './editorShell.svelte'
import { ApiError } from './apiErrors'
import { chooseDialog } from './confirm.svelte'

vi.mock('./api', () => ({ api: {} }))
const notifyError = vi.hoisted(() => vi.fn())
vi.mock('./toast', () => ({ notify: { error: notifyError } }))
vi.mock('./confirm.svelte', () => ({ chooseDialog: vi.fn() }))

function fakeEditor(overrides: Partial<DocumentEditor> = {}) {
  return {
    flush: vi.fn(),
    canWrite: true,
    preconditions: vi.fn(() => ({ ifMatch: '"old"' })),
    markSaved: vi.fn(),
    serialized: vi.fn(() => '{"text":"sent"}'),
    etag: '"old"',
    loadedPath: 'hero',
    writable: true,
    ...overrides,
  } as unknown as DocumentEditor
}

const conflict = (code: string) =>
  new ApiError('conflict', 409, code, { code, message: 'conflict' })

const REPLACE = [
  { label: 'Replace', value: 'replace', danger: true },
  { label: 'Cancel', value: 'cancel' },
]

beforeEach(() => vi.clearAllMocks())

it("saves with the editor's preconditions and records the etag", async () => {
  const ed = fakeEditor()
  const write = vi.fn(async () => ({ etag: '"new"' }))
  const outcome = await saveDocument(ed, { path: 'hero', label: 'hero', write })
  expect(outcome).toBe('saved')
  expect(ed.flush).toHaveBeenCalled()
  expect(ed.preconditions).toHaveBeenCalledWith('hero')
  expect(write).toHaveBeenCalledWith({ ifMatch: '"old"' })
  expect(ed.markSaved).toHaveBeenCalledWith('{"text":"sent"}')
  expect(ed.etag).toBe('"new"')
  expect(ed.loadedPath).toBe('hero')
  expect(chooseDialog).not.toHaveBeenCalled()
})

it('replaces on an exists conflict when the user chooses Replace', async () => {
  const ed = fakeEditor({ loadedPath: null })
  const write = vi
    .fn()
    .mockRejectedValueOnce(conflict('exists'))
    .mockResolvedValueOnce({ etag: '"new"' })
  vi.mocked(chooseDialog).mockResolvedValueOnce('replace')
  const outcome = await saveDocument(ed, {
    path: 'portraits/hero',
    label: 'portraits/hero',
    write,
  })
  expect(outcome).toBe('saved')
  expect(chooseDialog).toHaveBeenCalledWith(
    'portraits/hero already exists. Replace it?',
    REPLACE,
  )
  expect(write).toHaveBeenLastCalledWith({})
  expect(ed.etag).toBe('"new"')
  expect(ed.loadedPath).toBe('portraits/hero')
  expect(ed.writable).toBe(true)
})

it('cancelling a conflict leaves the document dirty', async () => {
  const ed = fakeEditor()
  const write = vi.fn().mockRejectedValueOnce(conflict('exists'))
  vi.mocked(chooseDialog).mockResolvedValueOnce(null)
  const outcome = await saveDocument(ed, { path: 'hero', label: 'hero', write })
  expect(outcome).toBe('cancelled')
  expect(write).toHaveBeenCalledTimes(1)
  expect(ed.markSaved).not.toHaveBeenCalled()
  expect(ed.etag).toBe('"old"')
})

it('asks with the stale_write copy', async () => {
  const ed = fakeEditor()
  const write = vi.fn().mockRejectedValueOnce(conflict('stale_write'))
  vi.mocked(chooseDialog).mockResolvedValueOnce('cancel')
  expect(await saveDocument(ed, { path: 'hero', label: 'hero', write })).toBe(
    'cancelled',
  )
  expect(chooseDialog).toHaveBeenCalledWith(
    'hero changed since you opened it, in another tab or by an agent. Replace that version with yours?',
    REPLACE,
  )
})

it('rethrows a non-conflict error', async () => {
  const ed = fakeEditor()
  const failure = new ApiError('bad', 400, 'invalid')
  const write = vi.fn().mockRejectedValueOnce(failure)
  await expect(
    saveDocument(ed, { path: 'hero', label: 'hero', write }),
  ).rejects.toBe(failure)
  // a 409 with some other code is not a write conflict either
  write.mockRejectedValueOnce(conflict('stale_revision'))
  await expect(
    saveDocument(ed, { path: 'hero', label: 'hero', write }),
  ).rejects.toThrow('conflict')
  expect(chooseDialog).not.toHaveBeenCalled()
  expect(ed.markSaved).not.toHaveBeenCalled()
})

it('refuses to save a document whose load failed', async () => {
  const ed = fakeEditor({ canWrite: false })
  const write = vi.fn()
  await expect(
    saveDocument(ed, { path: 'hero', label: 'hero', write }),
  ).rejects.toThrow('not loaded')
  expect(ed.flush).toHaveBeenCalled()
  expect(write).not.toHaveBeenCalled()
})

it('freeName counts past taken names', () => {
  expect(freeName('photo.jpg', new Set())).toBe('photo.jpg')
  expect(freeName('photo.jpg', new Set(['photo.jpg']))).toBe('photo-2.jpg')
  expect(freeName('photo.jpg', new Set(['photo.jpg', 'photo-2.jpg']))).toBe(
    'photo-3.jpg',
  )
  expect(freeName('notes', new Set(['notes']))).toBe('notes-2')
  expect(freeName('a.b/clip', new Set(['a.b/clip']))).toBe('a.b/clip-2')
})

it('refuses to save while the JSON does not parse', async () => {
  const ed = fakeEditor({ jsonParseFailed: true })
  const write = vi.fn(async () => ({ etag: '"new"' }))
  const outcome = await saveDocument(ed, { path: 'hero', label: 'hero', write })
  expect(outcome).toBe('cancelled')
  expect(notifyError).toHaveBeenCalledWith(
    'Fix the JSON before saving',
    'json-parse',
  )
  expect(write).not.toHaveBeenCalled()
  expect(ed.markSaved).not.toHaveBeenCalled()
})
