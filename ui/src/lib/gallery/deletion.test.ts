import { expect, it } from 'vitest'
import { deletionChoices } from './deletion'
import type { GalleryFile } from '../types'

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
