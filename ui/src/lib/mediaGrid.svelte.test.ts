import { flushSync } from 'svelte'
import { afterEach, expect, it, vi } from 'vitest'
import { clearOnEscape, revealOnce } from './mediaGrid.svelte'
import { Picks } from './picks.svelte'
import { holdLayer } from './ui/layers.svelte'

// What a page holds as the helper reads it: the open name and where each
// listed name sits
const view = $state({
  open: null as string | null,
  places: {} as Record<string, string>,
})
let teardown = () => {}

function track(reveal: (name: string, place: string) => void) {
  teardown = $effect.root(() => {
    revealOnce(
      () => view.open,
      (name) => view.places[name],
      reveal,
    )
  })
  flushSync()
}

afterEach(() => {
  teardown()
  view.open = null
  view.places = {}
})

it('reveals an open file once its place is known, and only once', () => {
  const reveal = vi.fn()
  view.open = 'a.png'
  track(reveal)
  // not listed yet: nothing to open
  expect(reveal).not.toHaveBeenCalled()
  view.places = { 'a.png': 'folder1' }
  flushSync()
  expect(reveal).toHaveBeenCalledExactlyOnceWith('a.png', 'folder1')
  // a reload of the same listing does not open it again
  view.places = { 'a.png': 'folder1', 'b.png': 'folder2' }
  flushSync()
  expect(reveal).toHaveBeenCalledTimes(1)
})

it('reveals each newly opened file, and nothing with none open', () => {
  const reveal = vi.fn()
  view.places = { 'a.png': 'f1', 'b.png': 'f2' }
  track(reveal)
  expect(reveal).not.toHaveBeenCalled()
  view.open = 'a.png'
  flushSync()
  view.open = 'b.png'
  flushSync()
  expect(reveal.mock.calls).toEqual([
    ['a.png', 'f1'],
    ['b.png', 'f2'],
  ])
})

const escape = new KeyboardEvent('keydown', { key: 'Escape' })

function picked(names: string[], shown: string[]) {
  const order = { shown }
  const picks = new Picks(() => order.shown)
  for (const name of names) picks.toggle(name, false)
  return { picks, order }
}

it('Escape clears the picks; another key does not', () => {
  const { picks } = picked(['a', 'b'], ['a', 'b'])
  const onKey = clearOnEscape(picks)
  onKey(new KeyboardEvent('keydown', { key: 'Enter' }))
  expect(picks.size).toBe(2)
  onKey(escape)
  expect(picks.size).toBe(0)
})

it('Escape under an open overlay leaves the picks', () => {
  const { picks } = picked(['a'], ['a'])
  const release = holdLayer()
  try {
    clearOnEscape(picks)(escape)
    clearOnEscape(picks, { keepHidden: true })(escape)
  } finally {
    release()
  }
  expect(picks.has('a')).toBe(true)
})

it('keepHidden leaves the ticks a filter hides when none is shown', () => {
  const { picks, order } = picked(['a'], ['a'])
  order.shown = ['b']
  clearOnEscape(picks, { keepHidden: true })(escape)
  order.shown = ['a']
  expect(picks.has('a')).toBe(true)
  // without it, the hidden tick goes too
  order.shown = ['b']
  clearOnEscape(picks)(escape)
  order.shown = ['a']
  expect(picks.has('a')).toBe(false)
})
