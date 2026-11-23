import { beforeEach, expect, it, vi } from 'vitest'
import { askDeletion, deletionChoices } from './deletion'
import { chooseDialog, confirmDialog } from '../confirm.svelte'
import type { GalleryFile } from '../types'

vi.mock('../confirm.svelte', () => ({
  chooseDialog: vi.fn(),
  confirmDialog: vi.fn(),
}))

beforeEach(() => vi.mocked(chooseDialog).mockReset())

const f = (name: string, starred = false) => ({ name, starred }) as GalleryFile

it('a selection with no starred files is one plain delete', () => {
  expect(deletionChoices([f('a'), f('b')], ['a', 'b'])).toEqual({
    title: 'Delete 2 files?',
    message: 'This removes them on disk.',
    confirmLabel: 'Delete',
    danger: true,
    starred: [],
  })
  const one = deletionChoices([f('a')], ['a'])
  expect([one.title, one.message, one.danger]).toEqual([
    'Delete 1 file?',
    'This removes it on disk.',
    true,
  ])
})

it('counts starred files among the selection only', () => {
  const r = deletionChoices([f('a', true), f('b'), f('c', true)], ['a', 'b'])
  expect(r.starred).toEqual(['a'])
  expect([r.title, r.message, r.danger]).toEqual([
    'Delete 2 files?',
    'This removes them on disk. 1 of them is starred.',
    true,
  ])
  expect(
    deletionChoices([f('a', true), f('b', true)], ['a', 'b']).message,
  ).toBe('This removes them on disk. 2 of them are starred.')
})

const labels = () =>
  vi.mocked(chooseDialog).mock.calls[0][1].map((choice) => choice.label)

it('offers to spare the starred files when some are unstarred', async () => {
  vi.mocked(chooseDialog).mockResolvedValueOnce('unstarred')
  expect(await askDeletion([f('a', true), f('b')], ['a', 'b'])).toEqual(['b'])
  expect(labels()).toEqual(['Delete all', 'Delete the 1 unstarred', 'Cancel'])
  // The question is destroyConfirm's: its title, and Delete all in danger
  const [message, choices, options] = vi.mocked(chooseDialog).mock.calls[0]
  expect(message).toBe('This removes them on disk. 1 of them is starred.')
  expect(options).toEqual({ title: 'Delete 2 files?' })
  expect(choices[0].danger).toBe(true)
})

it('a selection that is all starred offers no "Delete the 0 unstarred"', async () => {
  vi.mocked(chooseDialog).mockResolvedValueOnce('all')
  const files = [f('a', true), f('b', true)]
  expect(await askDeletion(files, ['a', 'b'])).toEqual(['a', 'b'])
  expect(labels()).toEqual(['Delete all', 'Cancel'])
})

it('a selection with nothing starred asks once, in danger', async () => {
  vi.mocked(confirmDialog).mockResolvedValueOnce(true)
  expect(await askDeletion([f('a'), f('b')], ['a', 'b'])).toEqual(['a', 'b'])
  expect(vi.mocked(confirmDialog).mock.calls[0]).toEqual([
    'This removes them on disk.',
    expect.objectContaining({
      title: 'Delete 2 files?',
      confirmLabel: 'Delete',
      danger: true,
    }),
  ])
})
