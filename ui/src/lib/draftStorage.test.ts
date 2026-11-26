import { expect, it } from 'vitest'
import { draftAtRisk } from './draftStorage'

it('a draft with text that differs from the saved note is at risk', () => {
  expect(draftAtRisk('needs more red', null)).toBe(true)
  expect(draftAtRisk('needs more red', 'needs red')).toBe(true)
})

it('an empty, blank or unchanged draft is not', () => {
  expect(draftAtRisk('', null)).toBe(false)
  expect(draftAtRisk('  \n', 'kept')).toBe(false)
  expect(draftAtRisk('kept', 'kept')).toBe(false)
})
