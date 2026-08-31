import { afterEach, describe, expect, it, vi } from 'vitest'
import { storageGet, storageSet } from './storage'

function stubStorage(backing: Record<string, string>) {
  vi.stubGlobal('localStorage', {
    getItem: (k: string) => (k in backing ? backing[k] : null),
    setItem: (k: string, v: string) => {
      backing[k] = v
    },
  })
}

afterEach(() => vi.unstubAllGlobals())

describe('storage', () => {
  it('round-trips JSON values under the dw- prefix', () => {
    const backing: Record<string, string> = {}
    stubStorage(backing)
    storageSet('step-modes:ZImage', { generate: 'compact' })
    expect(backing['dw-step-modes:ZImage']).toBe('{"generate":"compact"}')
    expect(storageGet('step-modes:ZImage', {})).toEqual({ generate: 'compact' })
  })

  it('returns the fallback for missing keys and corrupt JSON', () => {
    stubStorage({ 'dw-broken': '{not json' })
    expect(storageGet('missing', 'fb')).toBe('fb')
    expect(storageGet('broken', 'fb')).toBe('fb')
  })

  it('survives storage that throws (private mode)', () => {
    vi.stubGlobal('localStorage', {
      getItem: () => {
        throw new Error('denied')
      },
      setItem: () => {
        throw new Error('denied')
      },
    })
    expect(storageGet('anything', 42)).toBe(42)
    expect(() => storageSet('anything', 1)).not.toThrow()
  })
})
