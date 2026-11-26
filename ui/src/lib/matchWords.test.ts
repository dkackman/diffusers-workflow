import { expect, it } from 'vitest'
import { matchWords } from './matchWords'

it('wants every word, in any field and any order, case aside', () => {
  expect(matchWords('fox RED', 'a red', 'Fox')).toBe(true)
  expect(matchWords('red whale', 'a red', 'fox')).toBe(false)
})

it('matches part of a word, but never across two fields', () => {
  expect(matchWords('ortrai', 'portrait')).toBe(true)
  expect(matchWords('redfox', 'red', 'fox')).toBe(false)
})

it('matches everything on a blank query, and skips absent fields', () => {
  expect(matchWords('  ')).toBe(true)
  expect(matchWords('fox', undefined, null, 'fox')).toBe(true)
  expect(matchWords('null', null)).toBe(false)
})
