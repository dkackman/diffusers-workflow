import { describe, expect, it } from 'vitest'
import { PROMPT, VARIABLE, reference, referenceName } from './references'

describe('reference and referenceName', () => {
  it('build and read back a name, folders included', () => {
    expect(reference(PROMPT, 'cast/priya')).toBe('prompt:cast/priya')
    expect(referenceName('prompt:cast/priya', PROMPT)).toBe('cast/priya')
  })
  it('trim the name, as the engine does', () => {
    expect(referenceName('prompt: scenic ', PROMPT)).toBe('scenic')
  })
  it('answer null for another prefix or a non-string', () => {
    expect(referenceName('variable:x', PROMPT)).toBeNull()
    expect(referenceName(42, VARIABLE)).toBeNull()
  })
})
