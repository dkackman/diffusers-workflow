import { afterEach, describe, expect, it, vi } from 'vitest'
import { alertMode } from './finishAlert'

describe('alertMode', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('uses desktop notifications on a secure origin', () => {
    vi.stubGlobal('isSecureContext', true)
    vi.stubGlobal('Notification', function () {})
    expect(alertMode()).toBe('notification')
  })
  it('falls back to a chime over plain http from another machine', () => {
    vi.stubGlobal('isSecureContext', false)
    vi.stubGlobal('Notification', function () {})
    expect(alertMode()).toBe('chime')
  })
  it('falls back to a chime where the API does not exist', () => {
    vi.stubGlobal('isSecureContext', true)
    vi.stubGlobal('Notification', undefined)
    expect(alertMode()).toBe('chime')
  })
})
