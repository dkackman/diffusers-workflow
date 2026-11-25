import { expect, it, vi } from 'vitest'
import { Picks, actOnEach } from './picks.svelte'

const picksOver = (...names: string[]) => new Picks(() => names)

it('toggles a name on and off', () => {
  const picks = picksOver('a', 'b')

  picks.toggle('a', false)
  expect(picks.has('a')).toBe(true)
  expect(picks.size).toBe(1)

  picks.toggle('a', false)
  expect(picks.has('a')).toBe(false)
})

it('spans a shift-range from the last toggle, in grid order', () => {
  const picks = picksOver('a', 'b', 'c', 'd', 'e')

  picks.toggle('b', false)
  picks.toggle('d', true)

  expect(picks.names).toEqual(['b', 'c', 'd'])
})

it('spans a shift-range backwards too', () => {
  const picks = picksOver('a', 'b', 'c', 'd')

  picks.toggle('d', false)
  picks.toggle('b', true)

  expect(picks.names).toEqual(['b', 'c', 'd'])
})

// Extending a selection is what shift is for; toggling each cell would make
// the result depend on whatever the range happened to already contain
it('only ever adds over a shift-range', () => {
  const picks = picksOver('a', 'b', 'c')
  picks.toggle('a', false)
  picks.toggle('c', false)

  picks.toggle('a', true)

  expect(picks.names).toEqual(['a', 'b', 'c'])
})

it('shift with no anchor is a plain toggle', () => {
  const picks = picksOver('a', 'b')

  picks.toggle('b', true)

  expect(picks.names).toEqual(['b'])
})

it('reports the selection in grid order, not the order it was ticked', () => {
  const picks = picksOver('a', 'b', 'c')

  picks.toggle('c', false)
  picks.toggle('a', false)

  expect(picks.names).toEqual(['a', 'c'])
})

// A bulk action must never touch what the user cannot see, so the count is
// the visible selection - a tick the filter is hiding is inert until the
// filter lifts it back into view
it('size counts only what the order includes', () => {
  let shown = ['a', 'b', 'c']
  const picks = new Picks(() => shown)
  picks.toggle('a', false)
  picks.toggle('b', false)

  shown = ['a']

  expect(picks.size).toBe(1)

  shown = ['a', 'b', 'c']

  expect(picks.size).toBe(2)
})

it('forgets names that left the listing', () => {
  const picks = picksOver('a', 'b', 'c')
  picks.selectAll()

  picks.keepOnly(['a', 'c'])

  expect(picks.size).toBe(2)
  expect(picks.has('b')).toBe(false)
})

it('keeps only what failed, so a retry needs no re-ticking', () => {
  const picks = picksOver('a', 'b', 'c')
  picks.selectAll()

  picks.keepFailed(['a', 'b', 'c'], ['b'])

  expect(picks.names).toEqual(['b'])
})

it('keepFailed keeps ticks the action never attempted', () => {
  let shown = ['a', 'b', 'c']
  const picks = new Picks(() => shown)
  picks.selectAll()

  // 'c' is hidden by the filter, so the action never attempted it - it must
  // survive a keepFailed call scoped to what it did attempt
  shown = ['a', 'b']
  picks.keepFailed(['a', 'b'], ['a'])

  shown = ['a', 'b', 'c']
  expect(picks.names).toEqual(['a', 'c'])
})

it('clears the anchor with the selection', () => {
  const picks = picksOver('a', 'b', 'c')
  picks.toggle('a', false)
  picks.clear()

  picks.toggle('c', true)

  expect(picks.names).toEqual(['c'])
})

it('runs one act per name in order and collects the failures', async () => {
  const seen: string[] = []
  const failed = await actOnEach(['a', 'b', 'c'], (name) => {
    seen.push(name)
    return name === 'b' ? Promise.reject(new Error('no')) : Promise.resolve()
  })

  expect(seen).toEqual(['a', 'b', 'c'])
  expect(failed).toEqual([{ name: 'b', error: new Error('no') }])
  expect(failed[0].error).toBeInstanceOf(Error)
})

it('does not hold the grid order it was built with', () => {
  const order = vi.fn(() => ['a'])
  const picks = new Picks(order)

  picks.selectAll()

  // Read through the getter every time, so a filter changing under the
  // selection is seen rather than snapshotted at construction
  expect(order).toHaveBeenCalled()
  expect(picks.names).toEqual(['a'])
})

it('stars every pick once, and Undo puts each back as it was', async () => {
  const picks = picksOver('a', 'b', 'c')
  const held: Record<string, boolean> = { a: true, b: false, c: false }
  const set = vi.fn((name: string, next: boolean) => {
    held[name] = next
    return Promise.resolve()
  })
  const undo = vi.fn()
  const notify = (await import('./toast')).notify
  vi.spyOn(notify, 'undo').mockImplementation((_m, run) => undo(_m, run))
  picks.selectAll()
  const star = picks.star(set, 'file', { was: (n) => held[n] })

  star.run(true)
  star.run(true)
  await vi.waitFor(() => expect(undo).toHaveBeenCalledTimes(1))

  expect(set.mock.calls).toEqual([
    ['a', true],
    ['b', true],
    ['c', true],
  ])
  expect(undo.mock.calls[0][0]).toBe('Starred 3 files')
  undo.mock.calls[0][1]()
  await vi.waitFor(() => expect(set).toHaveBeenCalledTimes(6))
  expect(held).toEqual({ a: true, b: false, c: false })
})

it('a star that fails for some names them and undoes only the rest', async () => {
  const picks = picksOver('a', 'b')
  const set = vi.fn((name: string) =>
    name === 'b' ? Promise.reject(new Error('gone')) : Promise.resolve(),
  )
  const notifyMod = (await import('./toast')).notify
  const failure = vi.spyOn(notifyMod, 'failure').mockImplementation(() => {})
  const undo = vi.spyOn(notifyMod, 'undo').mockImplementation(() => {})
  picks.selectAll()
  picks.star(set, 'file').run(true)

  await vi.waitFor(() => expect(failure).toHaveBeenCalledTimes(1))
  expect(failure.mock.calls[0][3]).toContain('b:')
  await vi.waitFor(() =>
    expect(undo).toHaveBeenCalledWith('Starred 1 file', expect.any(Function)),
  )
})

it("a star Undo leaves another writer's star", async () => {
  const picks = picksOver('a', 'b')
  const held: Record<string, boolean> = { a: false, b: false }
  const set = vi.fn((name: string, next: boolean) => {
    held[name] = next
    return Promise.resolve()
  })
  const notifyMod = (await import('./toast')).notify
  const error = vi.spyOn(notifyMod, 'error').mockImplementation(() => {})
  const undo = vi.spyOn(notifyMod, 'undo').mockImplementation(() => {})
  picks.selectAll()
  picks.star(set, 'file', { was: (n) => held[n] }).run(true)
  await vi.waitFor(() => expect(undo).toHaveBeenCalledTimes(1))
  // Someone else unstarred b since: the Undo puts back only a
  held.b = false
  ;(undo.mock.calls[0][1] as () => void)()
  await vi.waitFor(() => expect(held.a).toBe(false))
  await new Promise((r) => setTimeout(r, 0))
  expect(set.mock.calls.slice(2)).toEqual([['a', false]])
  expect(error).not.toHaveBeenCalled()
  // Both changed since: nothing is written, and the toast says why
  undo.mockClear()
  picks.star(set, 'file', { was: (n) => held[n] }).run(true)
  await vi.waitFor(() => expect(undo).toHaveBeenCalledTimes(1))
  held.a = false
  held.b = false
  ;(undo.mock.calls[0][1] as () => void)()
  await vi.waitFor(() =>
    expect(error).toHaveBeenCalledWith(
      'Nothing undone: the star was changed since.',
    ),
  )
  expect(set).toHaveBeenCalledTimes(5)
})
