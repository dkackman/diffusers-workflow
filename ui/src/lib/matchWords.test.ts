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

it('takes the words in either order and ignores extra spaces', () => {
  expect(matchWords('red car', 'a car, red')).toBe(true)
  expect(matchWords('car red', 'a red car')).toBe(true)
  expect(matchWords('red car', 'a blue car')).toBe(false)
  expect(matchWords('  RED  ', 'red')).toBe(true)
  expect(matchWords('red \t  car', 'red car')).toBe(true)
})

it('matches everything on a blank query, and skips absent fields', () => {
  expect(matchWords('', 'fox')).toBe(true)
  expect(matchWords('  ')).toBe(true)
  expect(matchWords('fox', undefined, null, 'fox')).toBe(true)
  expect(matchWords('null', null)).toBe(false)
})
