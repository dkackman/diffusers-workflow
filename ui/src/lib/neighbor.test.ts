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
})
