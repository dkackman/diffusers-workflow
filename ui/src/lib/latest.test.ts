import { expect, it } from 'vitest'
import { latestOnly } from './latest'

it('only the newest call of a kind is current', () => {
  const begin = latestOnly()
  const first = begin()
  expect(first()).toBe(true)
  const second = begin()
  expect(first()).toBe(false)
  expect(second()).toBe(true)
})

it('keeps each kind of call apart', () => {
  const listing = latestOnly()
  const detail = latestOnly()
  const listed = listing()
  detail()
  expect(listed()).toBe(true)
})
