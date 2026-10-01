import { describe, expect, it } from 'vitest'
import { writableRoot } from './libraries'

describe('writableRoot', () => {
  it('names the writable workspace root, whatever its place on the path', () => {
    expect(
      writableRoot([
        { origin: 'examples', root: '/ex/workflows', writable: false },
        { origin: 'workspace', root: '/ws/workflows', writable: true },
      ]),
    ).toBe('/ws/workflows')
  })

  it('skips a writable shared root, which is not where a save lands', () => {
    expect(
      writableRoot([
        { origin: 'common', root: '/root/common/assets', writable: true },
        { origin: 'workspace', root: '/ws/assets', writable: true },
      ]),
    ).toBe('/ws/assets')
  })

  it('is empty when there is no workspace root', () => {
    expect(writableRoot([])).toBe('')
    expect(writableRoot(undefined)).toBe('')
    expect(
      writableRoot([{ origin: 'examples', root: '/ex', writable: false }]),
    ).toBe('')
  })
})
