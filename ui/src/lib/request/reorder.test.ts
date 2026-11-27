// The reorder gestures: what is handed up for a key or a drag, and when the
// order is written. reorderMath.test.ts holds the arithmetic.
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import type { Move } from './reorderMath'
import { COMMIT_MS, handleId, Reorder } from './reorder.svelte'

let list: HTMLElement
const onmove = vi.fn<(key: string, move: Move) => void>()
const oncommit = vi.fn()
let reorder: Reorder
let keys = ['a', 'b', 'c']

beforeEach(() => {
  vi.useFakeTimers()
  keys = ['a', 'b', 'c']
  onmove.mockReset()
  oncommit.mockReset()
  list = document.createElement('ul')
  for (const [index, key] of ['a', 'b', 'c'].entries()) {
    const row = document.createElement('li')
    row.dataset.key = key
    row.innerHTML = `<button id="${handleId(key)}"></button>`
    // jsdom lays nothing out: rows 20px tall, one under another
    row.getBoundingClientRect = () =>
      ({ top: index * 20, height: 20 }) as DOMRect
    list.appendChild(row)
  }
  document.body.appendChild(list)
  reorder = new Reorder({
    keys: () => keys,
    list: () => list,
    onmove,
    oncommit,
  })
})
afterEach(() => {
  vi.useRealTimers()
  document.body.innerHTML = ''
})

const key = (name: string, extra: object = {}) =>
  new KeyboardEvent('keydown', { key: name, cancelable: true, ...extra })
const pointer = (clientY: number, extra: object = {}) =>
  ({
    button: 0,
    pointerId: 1,
    clientY,
    currentTarget: list.querySelector('button'),
    preventDefault: () => {},
    ...extra,
  }) as unknown as PointerEvent

it('an arrow moves, keeps focus on the handle and writes once, later', async () => {
  const event = key('ArrowDown')
  await reorder.keydown(event, 'a')
  expect(event.defaultPrevented).toBe(true)
  expect(onmove).toHaveBeenCalledTimes(1)
  expect(onmove.mock.calls[0][0]).toBe('a')
  expect(document.activeElement?.id).toBe(handleId('a'))
  vi.advanceTimersByTime(COMMIT_MS - 1)
  await reorder.keydown(key('ArrowDown'), 'a')
  vi.advanceTimersByTime(COMMIT_MS - 1)
  expect(oncommit).not.toHaveBeenCalled()
  vi.advanceTimersByTime(1)
  expect(oncommit).toHaveBeenCalledTimes(1)
})

it('leaves a modified arrow, another key and a move past the end', async () => {
  await reorder.keydown(key('ArrowDown', { altKey: true }), 'a')
  await reorder.keydown(key('Enter'), 'a')
  await reorder.keydown(key('ArrowUp'), 'a')
  expect(onmove).not.toHaveBeenCalled()
})

it('losing focus writes at once, and only a waiting move', async () => {
  reorder.blur()
  expect(oncommit).not.toHaveBeenCalled()
  await reorder.keydown(key('End'), 'a')
  reorder.blur()
  expect(oncommit).toHaveBeenCalledTimes(1)
  vi.advanceTimersByTime(COMMIT_MS)
  expect(oncommit).toHaveBeenCalledTimes(1)
})

it('a drag marks the landing row and moves on release', async () => {
  const [a, , c] = [...list.children] as HTMLElement[]
  reorder.pointerdown(pointer(10), 'a')
  reorder.pointermove(pointer(55))
  expect(reorder.dragging).toBe('a')
  expect(c.classList.contains('insert-after')).toBe(true)
  reorder.pointerup(pointer(55))
  await vi.advanceTimersByTimeAsync(0)
  expect(onmove).toHaveBeenCalledTimes(1)
  expect(reorder.dragging).toBeNull()
  expect(a.style.transform).toBe('')
  expect(c.classList.contains('insert-after')).toBe(false)
})

it('a drop after the rows changed under the drag moves nothing', async () => {
  // The drop place was measured against the rows at pointer-down: a read
  // that adds, drops or reorders one mid-drag would land it a place off
  reorder.pointerdown(pointer(10), 'a')
  reorder.pointermove(pointer(55))
  keys = ['x', 'a', 'b', 'c']
  reorder.pointerup(pointer(55))
  await vi.advanceTimersByTimeAsync(0)
  expect(onmove).not.toHaveBeenCalled()
  expect(reorder.dragging).toBeNull()
})

it('a press that barely travels, or a cancelled drag, moves nothing', () => {
  reorder.pointerdown(pointer(10), 'a')
  reorder.pointermove(pointer(12))
  reorder.pointerup(pointer(12))
  reorder.pointerdown(pointer(10), 'a')
  reorder.pointermove(pointer(55))
  reorder.cancel()
  reorder.pointerup(pointer(55))
  expect(onmove).not.toHaveBeenCalled()
  expect(reorder.dragging).toBeNull()
})

it('writes what is waiting when the page goes', async () => {
  await reorder.keydown(key('ArrowDown'), 'a')
  reorder.destroy()
  expect(oncommit).toHaveBeenCalledTimes(1)
})
