import { expect, it } from 'vitest'
import { destroyConfirm, workspaceConfirm } from './destructive'

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
