import { expect, it } from 'vitest'
import {
  formatRequestFilter,
  requestQuery,
  parseRequestFilter,
} from './requestFilter'

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

it('writes the hash query for a request, and for one item of it', () => {
  expect(requestQuery('r1')).toBe('?req=r1')
  expect(requestQuery('r1', 'shot-2')).toBe('?req=r1%2Fshot-2')
})
