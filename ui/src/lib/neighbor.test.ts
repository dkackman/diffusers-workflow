import { describe, expect, it } from 'vitest'
import { neighbor } from './neighbor'

describe('neighbor', () => {
  const items = ['a', 'b', 'c']
  it('steps forward and back', () => {
    expect(neighbor(items, 'b', 1)).toBe('c')
    expect(neighbor(items, 'b', -1)).toBe('a')
  })
  it('stops at the ends rather than wrapping', () => {
    expect(neighbor(items, 'c', 1)).toBeNull()
    expect(neighbor(items, 'a', -1)).toBeNull()
  })
  it('is null for a name no longer listed', () => {
    expect(neighbor(items, 'z', 1)).toBeNull()
  })
  it('goes on from where an unlisted name last stood', () => {
    // 'x' stood at 1, between 'a' and 'b'; 'b' has taken its place
    expect(neighbor(items, 'x', 1, 1)).toBe('b')
    expect(neighbor(items, 'x', -1, 1)).toBe('a')
    expect(neighbor(items, 'x', -1, 0)).toBeNull()
    expect(neighbor(items, 'x', 1, 3)).toBeNull()
  })
})
