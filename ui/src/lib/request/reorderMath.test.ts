import { expect, it } from 'vitest'
import {
  anchorAt,
  dropIndex,
  indexAfter,
  moveTarget,
  moveTo,
  placeBefore,
} from './reorderMath'

const KEYS = ['a', 'b', 'c', 'd']

it('moves one place, or to an end, and stops at the ends', () => {
  expect(indexAfter(4, 1, 'up')).toBe(0)
  expect(indexAfter(4, 1, 'down')).toBe(2)
  expect(indexAfter(4, 0, 'up')).toBe(0)
  expect(indexAfter(4, 3, 'down')).toBe(3)
  expect(indexAfter(4, 2, 'top')).toBe(0)
  expect(indexAfter(4, 1, 'bottom')).toBe(3)
})

it('names a move by the item it lands before, null for the end', () => {
  expect(moveTarget(KEYS, 'c', 'up')).toEqual({
    before: 'b',
    position: 2,
    total: 4,
  })
  expect(moveTarget(KEYS, 'b', 'down')).toEqual({
    before: 'd',
    position: 3,
    total: 4,
  })
  expect(moveTarget(KEYS, 'c', 'down')).toEqual({
    before: null,
    position: 4,
    total: 4,
  })
  expect(moveTarget(KEYS, 'c', 'top')?.before).toBe('a')
  expect(moveTarget(KEYS, 'a', 'bottom')?.before).toBeNull()
})

it('is no move at all when the item is already there, or is not listed', () => {
  expect(moveTarget(KEYS, 'a', 'up')).toBeNull()
  expect(moveTarget(KEYS, 'a', 'top')).toBeNull()
  expect(moveTarget(KEYS, 'd', 'down')).toBeNull()
  expect(moveTarget(KEYS, 'd', 'bottom')).toBeNull()
  expect(moveTarget(KEYS, 'zz', 'up')).toBeNull()
  expect(moveTo(KEYS, 'b', 1)).toBeNull()
  expect(moveTo(KEYS, 'b', 9)).toBeNull()
})

it('placing before the anchor gives the order the move promised', () => {
  for (const key of KEYS)
    for (let to = 0; to < KEYS.length; to++) {
      const placed = placeBefore(KEYS, key, anchorAt(KEYS, key, to))
      expect(placed.indexOf(key)).toBe(to)
      expect([...placed].sort()).toEqual(KEYS)
    }
})

it('an anchor still means the same place after items are added around it', () => {
  // "c before b", read against a plan the agent has since added x and y to
  expect(placeBefore(['a', 'x', 'b', 'y', 'c', 'd'], 'c', 'b')).toEqual([
    'a',
    'x',
    'c',
    'b',
    'y',
    'd',
  ])
  // an anchor that is gone puts it last rather than somewhere arbitrary
  expect(placeBefore(KEYS, 'a', 'gone')).toEqual(['b', 'c', 'd', 'a'])
})

it('a drop lands where the pointer is among the other rows', () => {
  const rows = [0, 40, 80, 120].map((top) => ({ top, height: 40 }))
  // dragging row 0: past the middle of row 1 only
  expect(dropIndex(rows, 0, 65)).toBe(1)
  expect(dropIndex(rows, 0, 10)).toBe(0)
  expect(dropIndex(rows, 0, 500)).toBe(3)
  // dragging row 3 up above everything
  expect(dropIndex(rows, 3, -5)).toBe(0)
  expect(dropIndex(rows, 3, 70)).toBe(2)
  expect(moveTo(KEYS, 'a', dropIndex(rows, 0, 65))).toEqual({
    before: 'c',
    position: 2,
    total: 4,
  })
})
