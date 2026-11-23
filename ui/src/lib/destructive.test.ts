import { expect, it, vi } from 'vitest'
import {
  destroyConfirm,
  modelDeleteConfirm,
  serverConfirm,
  undoToast,
  workspaceConfirm,
} from './destructive'

const notify = vi.hoisted(() => ({ undo: vi.fn(), failure: vi.fn() }))
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
      'This permanently removes 12 outputs (2 starred), 4 assets and 3 ' +
      'workflows (10.0 MB). The prompt library and the shared asset ' +
      'library are kept.',
    confirmLabel: 'Delete workspace',
    danger: true,
  })
  expect(
    workspaceConfirm('studio', { outputs: { files: 1, bytes: MB } }, 0).message,
  ).toBe(
    'This permanently removes 1 output (1.0 MB). The prompt library and ' +
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
