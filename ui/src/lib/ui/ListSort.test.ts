import { expect, it } from 'vitest'
import { newestFirst } from './ListSort.svelte'

it('newest first breaks a tie by name, and puts an entry with no time last, by name', () => {
  const at: Record<string, number | undefined> = { b: 5, a: 5, c: 9 }
  expect(newestFirst(['z', 'b', 'y', 'c', 'a'], (name) => at[name])).toEqual([
    'c',
    'a',
    'b',
    'y',
    'z',
  ])
})
