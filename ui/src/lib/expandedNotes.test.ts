import { beforeEach, describe, expect, it } from 'vitest'
import { isRemembered, remember } from './expandedNotes'

describe('expandedNotes', () => {
  beforeEach(() => localStorage.clear())

  it('remembers and forgets', () => {
    remember('workspace:a', true)
    expect(isRemembered('workspace:a')).toBe(true)
    remember('workspace:a', false)
    expect(isRemembered('workspace:a')).toBe(false)
  })
  it('keeps only the newest 200', () => {
    for (let i = 0; i < 205; i++) remember(`k${i}`, true)
    expect(isRemembered('k0')).toBe(false)
    expect(isRemembered('k204')).toBe(true)
  })
})
