import { beforeEach, expect, it, vi } from 'vitest'
import { openWorkflow, saveWorkflowAt } from './workflowFile'
import type { DocumentEditor } from '../editorShell.svelte'
import { api } from '../test/apiMock'
import { saveDocument } from '../documentSave'

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
api.saveWorkflow.mockReturnValue(undefined)
vi.mock('../documentSave', () => ({ saveDocument: vi.fn() }))

function fakeEditor() {
  return {
    beginLoad: vi.fn(),
    load: vi.fn(),
    failLoad: vi.fn(),
    serialized: vi.fn(() => '{"id":"on-screen","steps":[]}'),
  } as unknown as DocumentEditor
}

beforeEach(() => vi.clearAllMocks())

it('opens a workflow at the etag, path and origin it was read with', async () => {
  const ed = fakeEditor()
  api.getWorkflow.mockResolvedValue({
    definition: { id: 'w', steps: [] },
    origin: 'examples',
    writable: false,
    etag: '"e1"',
  } as never)
  expect(await openWorkflow(ed, 'templates/w')).toBe(true)
  expect(ed.beginLoad).toHaveBeenCalled()
  expect(ed.load).toHaveBeenCalledWith(
    { id: 'w', steps: [] },
    { etag: '"e1"', path: 'templates/w', writable: false },
  )
})

it('a failed read is left on the editor, not opened', async () => {
  const ed = fakeEditor()
  const error = new Error('down')
  api.getWorkflow.mockRejectedValue(error)
  expect(await openWorkflow(ed, 'w')).toBe(false)
  expect(ed.failLoad).toHaveBeenCalledWith(error)
  expect(ed.load).not.toHaveBeenCalled()
})

it('a read the page has moved on from touches nothing', async () => {
  const ed = fakeEditor()
  api.getWorkflow.mockRejectedValue(new Error('late'))
  expect(await openWorkflow(ed, 'w', () => true)).toBe(false)
  expect(ed.failLoad).not.toHaveBeenCalled()
  api.getWorkflow.mockResolvedValue({ definition: {} } as never)
  expect(await openWorkflow(ed, 'w', () => true)).toBe(false)
  expect(ed.load).not.toHaveBeenCalled()
})

it('saves through saveDocument, writing what is on screen', async () => {
  const ed = fakeEditor()
  vi.mocked(saveDocument).mockResolvedValue('saved')
  expect(await saveWorkflowAt(ed, 'models/mine')).toBe('saved')
  const [passed, opts] = vi.mocked(saveDocument).mock.calls[0]
  expect(passed).toBe(ed)
  expect(opts.path).toBe('models/mine')
  expect(opts.label).toBe('models/mine')
  await opts.write({ createOnly: true })
  expect(api.saveWorkflow).toHaveBeenCalledWith(
    'models/mine',
    { id: 'on-screen', steps: [] },
    { createOnly: true },
  )
})
