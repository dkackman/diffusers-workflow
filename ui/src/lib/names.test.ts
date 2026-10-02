import { describe, expect, it } from 'vitest'
import { isNameSegment, leafName } from './names'

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

describe('leafName', () => {
  it('is the last segment of a slash path', () => {
    expect(leafName('portraits/2024/face.png')).toBe('face.png')
  })

  it('is the name itself when there is no folder', () => {
    expect(leafName('face.png')).toBe('face.png')
  })
})
