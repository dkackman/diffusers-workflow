import { describe, expect, it } from 'vitest'
import { isNameSegment } from './names'

describe('isNameSegment', () => {
  it.each(['ep4', 'ep4.v2', 'café', '_lead', 'cast-2'])('accepts %s', (n) =>
    expect(isNameSegment(n)).toBe(true),
  )
  it.each(['', '.hidden', '-lead', 'a/b', 'a b'])('refuses %s', (n) =>
    expect(isNameSegment(n)).toBe(false),
  )
  it('counts code points, as the engine does', () => {
    expect(isNameSegment('𝐀'.repeat(100))).toBe(true)
    expect(isNameSegment('𝐀'.repeat(101))).toBe(false)
  })
})
