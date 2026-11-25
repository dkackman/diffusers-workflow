import { afterEach, expect, it } from 'vitest'
import { collapsedState } from './collapsed.svelte'

afterEach(() => localStorage.clear())

it('remembers what was shut under its key', () => {
  const first = collapsedState('collapsed-test')
  expect(first.has('a')).toBe(false)
  first.toggle('a')
  expect(first.has('a')).toBe(true)
  expect(collapsedState('collapsed-test').has('a')).toBe(true)
  expect(collapsedState('collapsed-other').has('a')).toBe(false)
  first.toggle('a')
  expect(collapsedState('collapsed-test').has('a')).toBe(false)
})

it('a getter key is followed, each key keeping its own', () => {
  let key = 'collapsed-one'
  const state = collapsedState(() => key)
  state.toggle('a')
  key = 'collapsed-two'
  expect(state.has('a')).toBe(false)
  key = 'collapsed-one'
  expect(state.has('a')).toBe(true)
  expect(collapsedState('collapsed-one').has('a')).toBe(true)
})
