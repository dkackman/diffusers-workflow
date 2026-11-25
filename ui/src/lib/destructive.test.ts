import { expect, it, vi } from 'vitest'
import {
  cacheClearConfirm,
  cacheEntryConfirm,
  destroyConfirm,
  modelDeleteConfirm,
  serverConfirm,
  undoToast,
  workspaceConfirm,
} from './destructive'

const notify = vi.hoisted(() => ({
  undo: vi.fn(),
  failure: vi.fn(),
  error: vi.fn(),
}))
vi.mock('./toast', () => ({ notify }))

it('one file, one starred, several', () => {
  expect(destroyConfirm({ name: 'a.png', noun: 'file' })).toEqual({
    title: 'Delete a.png?',
    message: 'This removes it on disk.',
    confirmLabel: 'Delete',
    danger: true,
  })
  expect(
    destroyConfirm({
      name: 'a.png',
      noun: 'file',
      starred: 1,
      consequence: 'Nothing else changes.',
    }).message,
  ).toBe('This removes it on disk. It is starred. Nothing else changes.')
  const several = destroyConfirm({ count: 3, noun: 'file', starred: 2 })
  expect(several.title).toBe('Delete 3 files?')
  expect(several.message).toBe(
    'This removes them on disk. 2 of them are starred.',
  )
  expect(several.danger).toBe(true)
  expect(destroyConfirm({ count: 2, noun: 'file', starred: 1 }).message).toBe(
    'This removes them on disk. 1 of them is starred.',
  )
  expect(destroyConfirm({ count: 1, noun: 'asset' }).title).toBe(
    'Delete 1 asset?',
  )
  expect(destroyConfirm({ count: 1, noun: 'asset' }).message).toBe(
    'This removes it on disk.',
  )
  expect(
    destroyConfirm({ count: 2, noun: 'library', plural: 'libraries' }).title,
  ).toBe('Delete 2 libraries?')
})

const MB = 1024 * 1024

it('the workspace confirm counts what it removes, starred included', () => {
  const spec = workspaceConfirm(
    'studio',
    {
      workflows: { files: 3, bytes: MB },
      assets: { files: 4, bytes: MB },
      outputs: { files: 12, bytes: 8 * MB },
    },
    2,
  )
  expect(spec).toEqual({
    title: 'Delete workspace "studio"?',
    message:
      'This permanently removes 12 files in outputs/ (2 starred in the gallery), 4 assets and 3 ' +
      'workflows (10.0 MB). The prompt library and the shared asset ' +
      'library are kept.',
    confirmLabel: 'Delete workspace',
    danger: true,
  })
  expect(
    workspaceConfirm('studio', { outputs: { files: 1, bytes: MB } }, 0).message,
  ).toBe(
    'This permanently removes 1 file in outputs/ (1.0 MB). The prompt library and ' +
      'the shared asset library are kept.',
  )
})

it('a folder with no files is left out', () => {
  const spec = workspaceConfirm(
    'studio',
    {
      workflows: { files: 0, bytes: 0 },
      assets: { files: 2, bytes: MB },
      outputs: { files: 0, bytes: 0 },
    },
    0,
  )
  expect(spec.message).toBe(
    'This permanently removes 2 assets (1.0 MB). The prompt library and ' +
      'the shared asset library are kept.',
  )
})

it('a server change asks in the plain style, never danger', () => {
  expect(
    serverConfirm({ title: 'T?', message: 'M.', confirmLabel: 'Go' }),
  ).toEqual({ title: 'T?', message: 'M.', confirmLabel: 'Go', danger: false })
  expect(modelDeleteConfirm('org/m', '2.5')).toEqual({
    title: 'Delete org/m (2.5 GB) from the model cache?',
    message: 'The next workflow that needs it downloads it again.',
    confirmLabel: 'Delete model',
    danger: false,
  })
})

it('an undo toast reverts, and a revert that fails is reported', async () => {
  const revert = vi.fn()
  undoToast('Removed step b', revert)
  expect(notify.undo.mock.calls[0][0]).toBe('Removed step b')
  notify.undo.mock.calls[0][1]()
  await vi.waitFor(() => expect(revert).toHaveBeenCalledTimes(1))
  const bad = new Error('gone')
  undoToast('Starred a.png', () => Promise.reject(bad))
  notify.undo.mock.calls[1][1]()
  await vi.waitFor(() =>
    expect(notify.failure).toHaveBeenCalledWith('undo that', bad),
  )
  undoToast('Removed variable x', () => {
    throw bad
  })
  notify.undo.mock.calls[2][1]()
  await vi.waitFor(() => expect(notify.failure).toHaveBeenCalledTimes(2))
})

it('an undo runs at most once: a second click on its fading toast does nothing', async () => {
  notify.undo.mockClear()
  const revert = vi.fn()
  undoToast('Removed step b', revert)
  const press = notify.undo.mock.calls[0][1]
  press()
  press()
  await vi.waitFor(() => expect(revert).toHaveBeenCalledTimes(1))
  await new Promise((r) => setTimeout(r, 0))
  expect(revert).toHaveBeenCalledTimes(1)
})

it('an undo is refused, with a notice, once its target is no longer open', async () => {
  notify.undo.mockClear()
  let open = 'a'
  const revert = vi.fn()
  const target = { now: () => open, away: 'Nothing undone: elsewhere.' }
  undoToast('Removed step b', revert, target)
  open = 'b'
  notify.undo.mock.calls[0][1]()
  await new Promise((r) => setTimeout(r, 0))
  expect(revert).not.toHaveBeenCalled()
  expect(notify.error).toHaveBeenCalledWith('Nothing undone: elsewhere.')

  undoToast('Removed step c', revert, target)
  notify.undo.mock.calls[1][1]()
  await vi.waitFor(() => expect(revert).toHaveBeenCalledTimes(1))
})

it('a revert that waits asks again whether its target is still open', async () => {
  notify.undo.mockClear()
  notify.error.mockClear()
  let open = 'a'
  let ready: (() => boolean) | undefined
  const target = { now: () => open, away: 'Nothing undone: elsewhere.' }
  undoToast('Removed step b', (r) => void (ready = r), target)
  notify.undo.mock.calls[0][1]()
  await vi.waitFor(() => expect(ready).toBeDefined())
  expect(ready!()).toBe(true)
  // The revert's wait is over, and another target is open by then
  open = 'b'
  expect(ready!()).toBe(false)
  expect(notify.error).toHaveBeenCalledWith('Nothing undone: elsewhere.')
})

it('an undo toast given an id is shown under it, for the caller to dismiss', () => {
  notify.undo.mockClear()
  undoToast('Dropped Shot 3', vi.fn(), undefined, 'request-undo-1')
  expect(notify.undo).toHaveBeenCalledWith(
    'Dropped Shot 3',
    expect.any(Function),
    'request-undo-1',
  )
  undoToast('Removed step b', vi.fn())
  expect(notify.undo.mock.calls[1]).toHaveLength(2)
})

it('the cache confirms name the cache or entry once, in the title, and count files', () => {
  const triton = {
    name: 'triton',
    root: '/cache/triton',
    files: 1,
    cost_of_clearing: 'It is read again.',
  }
  expect(cacheClearConfirm(triton, '1 KB')).toEqual({
    title: 'Clear the triton cache at /cache/triton?',
    message: '1 KB, 1 file. It is read again.',
    confirmLabel: 'Clear',
    danger: false,
  })
  expect(cacheClearConfirm({ ...triton, files: 2 }, '2 KB').message).toBe(
    '2 KB, 2 files. It is read again.',
  )
  expect(
    cacheEntryConfirm('quantized', 'vae of org/flux', '3.00 GB', 'Costly.'),
  ).toEqual({
    title: 'Delete vae of org/flux from the quantized cache?',
    message: '3.00 GB. Costly.',
    confirmLabel: 'Delete entry',
    danger: false,
  })
})

it('an unknown starred count is said', () => {
  expect(
    workspaceConfirm('studio', { outputs: { files: 2, bytes: MB } }, null)
      .message,
  ).toBe(
    'This permanently removes 2 files in outputs/ (1.0 MB). The prompt library and ' +
      "the shared asset library are kept. The starred count couldn't be read.",
  )
})

it('a ledger delete says the ledger', () => {
  expect(
    destroyConfirm({ name: 'this note', noun: 'note', place: 'ledger' })
      .message,
  ).toBe('This removes it from the ledger.')
  expect(destroyConfirm({ name: 'a.png', noun: 'file' }).message).toBe(
    'This removes it on disk.',
  )
})

it('the workspace delete says which count is which', () => {
  const { message } = workspaceConfirm(
    'studio',
    { outputs: { files: 5, bytes: 1024 * 1024 } },
    1,
  )
  expect(message).toContain('5 files in outputs/ (1 starred in the gallery)')
})
