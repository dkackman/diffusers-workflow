import { afterEach, expect, it } from 'vitest'
import { focusPageFilter } from './pageFrame.svelte'

afterEach(() => {
  document.body.innerHTML = ''
})

it('focuses the input inside a page header filter', () => {
  document.body.innerHTML =
    '<main><div class="pagehead"><label class="filter"><input id="a"></label></div></main>'
  expect(focusPageFilter()).toBe(true)
  expect(document.activeElement?.id).toBe('a')
})

it('focuses an input that is itself the filter', () => {
  document.body.innerHTML =
    '<main><div class="pagehead"><input class="filter" id="b"></div></main>'
  expect(focusPageFilter()).toBe(true)
  expect(document.activeElement?.id).toBe('b')
})

it('says false when the page has no search', () => {
  document.body.innerHTML =
    '<main><div class="pagehead"></div></main><input class="filter">'
  expect(focusPageFilter()).toBe(false)
})
