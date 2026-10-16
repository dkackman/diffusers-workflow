import { describe, expect, it } from 'vitest'
import { isEditable } from './editable'

describe('isEditable', () => {
  it('is true for text entry and media, false for plain elements', () => {
    expect(isEditable(document.createElement('input'))).toBe(true)
    expect(isEditable(document.createElement('textarea'))).toBe(true)
    expect(isEditable(document.createElement('select'))).toBe(true)
    // a focused <video> seeks on the arrow keys itself
    expect(isEditable(document.createElement('video'))).toBe(true)
    expect(isEditable(document.createElement('div'))).toBe(false)
    expect(isEditable(null)).toBe(false)
  })
})
