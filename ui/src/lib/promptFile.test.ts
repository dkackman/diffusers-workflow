import { beforeEach, expect, it, vi } from 'vitest'
import { openPrompt, savePromptAt } from './promptFile'
import type { DocumentEditor } from './editorShell.svelte'
import { api } from './test/apiMock'
import { saveDocument } from './documentSave'

vi.mock('./api', async () => (await import('./test/apiMock')).apiModule())
// The old mock's `savePrompt` was a bare vi.fn(): it answered undefined
api.savePrompt.mockReturnValue(undefined)
vi.mock('./documentSave', () => ({ saveDocument: vi.fn() }))

function fakeEditor() {
  return {
    beginLoad: vi.fn(),
    load: vi.fn(),
    failLoad: vi.fn(),
    serialized: vi.fn(() => '{"text":"on screen"}'),
  } as unknown as DocumentEditor
}

beforeEach(() => vi.clearAllMocks())

it('opens a prompt at the etag, path and library it was read with', async () => {
  const ed = fakeEditor()
  const prompt = { text: 'a hero at dawn' }
  api.getPrompt.mockResolvedValue({
    prompt,
    origin: 'examples',
    writable: false,
    etag: '"p1"',
  } as never)
  expect(await openPrompt(ed, 'portraits/hero')).toBe(prompt)
  expect(ed.beginLoad).toHaveBeenCalled()
  expect(ed.load).toHaveBeenCalledWith(prompt, {
    etag: '"p1"',
    path: 'portraits/hero',
    writable: false,
  })
})

it('a failed read is left on the editor, not opened', async () => {
  const ed = fakeEditor()
  const error = new Error('down')
  api.getPrompt.mockRejectedValue(error)
  expect(await openPrompt(ed, 'w')).toBeNull()
  expect(ed.failLoad).toHaveBeenCalledWith(error)
  expect(ed.load).not.toHaveBeenCalled()
})

it('a read the page has moved on from touches nothing', async () => {
  const ed = fakeEditor()
  api.getPrompt.mockRejectedValue(new Error('late'))
  expect(await openPrompt(ed, 'w', () => true)).toBeNull()
  expect(ed.failLoad).not.toHaveBeenCalled()
  api.getPrompt.mockResolvedValue({ prompt: {} } as never)
  expect(await openPrompt(ed, 'w', () => true)).toBeNull()
  expect(ed.load).not.toHaveBeenCalled()
})

it('saves through saveDocument, writing what is on screen', async () => {
  const ed = fakeEditor()
  vi.mocked(saveDocument).mockResolvedValue('saved')
  expect(await savePromptAt(ed, 'portraits/mine')).toBe('saved')
  const [passed, opts] = vi.mocked(saveDocument).mock.calls[0]
  expect(passed).toBe(ed)
  expect(opts.path).toBe('portraits/mine')
  expect(opts.label).toBe('portraits/mine')
  await opts.write({ createOnly: true })
  expect(api.savePrompt).toHaveBeenCalledWith(
    'portraits/mine',
    { text: 'on screen' },
    { createOnly: true },
  )
})
