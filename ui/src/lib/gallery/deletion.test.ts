import { beforeEach, expect, it, vi } from 'vitest'
import { askDeletion, deletionChoices } from './deletion'
import { chooseDialog } from '../confirm.svelte'
import type { GalleryFile } from '../types'

vi.mock('../confirm.svelte', () => ({
  chooseDialog: vi.fn(),
  confirmDialog: vi.fn(),
}))

beforeEach(() => vi.mocked(chooseDialog).mockReset())

const f = (name: string, starred = false) => ({ name, starred }) as GalleryFile

it('a selection with no starred files is one plain delete', () => {
  expect(deletionChoices([f('a'), f('b')], ['a', 'b'])).toEqual({
    message: 'Delete 2 files? This removes them on disk.',
    starred: [],
  })
  expect(deletionChoices([f('a')], ['a']).message).toBe(
    'Delete 1 file? This removes them on disk.',
  )
})

it('counts starred files among the selection only', () => {
  const r = deletionChoices([f('a', true), f('b'), f('c', true)], ['a', 'b'])
  expect(r.starred).toEqual(['a'])
  expect(r.message).toBe(
    'Delete 2 files? 1 of them is starred. This removes them on disk.',
  )
  expect(
    deletionChoices([f('a', true), f('b', true)], ['a', 'b']).message,
  ).toBe('Delete 2 files? 2 of them are starred. This removes them on disk.')
})

const labels = () =>
  vi.mocked(chooseDialog).mock.calls[0][1].map((choice) => choice.label)

it('offers to spare the starred files when some are unstarred', async () => {
  vi.mocked(chooseDialog).mockResolvedValueOnce('unstarred')
  expect(await askDeletion([f('a', true), f('b')], ['a', 'b'])).toEqual(['b'])
  expect(labels()).toEqual(['Delete all', 'Delete the 1 unstarred', 'Cancel'])
})

it('a selection that is all starred offers no "Delete the 0 unstarred"', async () => {
  vi.mocked(chooseDialog).mockResolvedValueOnce('all')
  const files = [f('a', true), f('b', true)]
  expect(await askDeletion(files, ['a', 'b'])).toEqual(['a', 'b'])
  expect(labels()).toEqual(['Delete all', 'Cancel'])
})
