import { expect, it } from 'vitest'
import { formatRequestFilter, parseRequestFilter } from './requestFilter'

it('parses a request id with and without an item', () => {
  expect(parseRequestFilter('r1')).toEqual({ request: 'r1', item: '' })
  expect(parseRequestFilter('r1/shot-2')).toEqual({
    request: 'r1',
    item: 'shot-2',
  })
  expect(parseRequestFilter(undefined)).toEqual({ request: '', item: '' })
})

it('formats what it parses, and nothing without a request', () => {
  expect(formatRequestFilter({ request: 'r1', item: 'a' })).toBe('r1/a')
  expect(formatRequestFilter({ request: 'r1', item: '' })).toBe('r1')
  expect(formatRequestFilter({ request: '', item: 'a' })).toBe('')
})
