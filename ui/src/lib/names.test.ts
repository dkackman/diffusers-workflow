import { describe, expect, it } from 'vitest'
import { distinctLeaves, isNameSegment, leafName } from './names'

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

describe('distinctLeaves', () => {
  it('two same-leaf workflows are told apart', () => {
    const shown = distinctLeaves([
      'people/portrait.json',
      'pets/portrait.json',
      'landscape.json',
    ])
    expect(shown.get('people/portrait.json')).toBe('people/portrait.json')
    expect(shown.get('pets/portrait.json')).toBe('pets/portrait.json')
    expect(shown.get('landscape.json')).toBe('landscape.json')
  })
  it('is the shortest tail that no other path ends with', () => {
    const shown = distinctLeaves([
      'a/x/run1/a.png',
      'b/x/run1/a.png',
      'a/run2/a.png',
      'c/b.png',
    ])
    expect([...shown.values()]).toEqual([
      'a/x/run1/a.png',
      'b/x/run1/a.png',
      'run2/a.png',
      'b.png',
    ])
  })
  it('a path given twice is its leaf', () => {
    expect(distinctLeaves(['d/a.png', 'd/a.png']).get('d/a.png')).toBe('a.png')
  })
})
