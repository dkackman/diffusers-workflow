import { expect, it } from 'vitest'
import { ACTIVE_MS, IDLE_MS, newerOf, refreshInterval } from './refresh'
import type { LedgerRequest } from '../types'

const request = (
  state: LedgerRequest['state'],
  statuses: string[],
  revision = 1,
): LedgerRequest =>
  ({
    id: 'r1',
    state,
    revision,
    items: statuses.map((status, i) => ({ key: `k${i}`, status })),
  }) as unknown as LedgerRequest

it('follows a job in flight closely, whatever the request state', () => {
  expect(refreshInterval(request('open', ['done', 'running']))).toBe(ACTIVE_MS)
  expect(refreshInterval(request('open', ['queued']))).toBe(ACTIVE_MS)
  expect(refreshInterval(request('done', ['running']))).toBe(ACTIVE_MS)
})

it('keeps an idle open request fresh at a slow interval', () => {
  expect(refreshInterval(request('open', ['planned', 'done']))).toBe(IDLE_MS)
  expect(refreshInterval(request('open', []))).toBe(IDLE_MS)
  expect(IDLE_MS).toBeGreaterThan(ACTIVE_MS)
})

it('leaves a closed request with nothing in flight, or none, alone', () => {
  expect(refreshInterval(request('done', ['done']))).toBeNull()
  expect(refreshInterval(request('abandoned', ['planned']))).toBeNull()
  expect(refreshInterval(null)).toBeNull()
})

it('a refresh at an older revision does not replace the one held', () => {
  const held = request('open', ['planned'], 4)
  expect(newerOf(held, request('open', ['planned'], 3))).toBe(held)
})

it('a refresh at the same or a later revision replaces it', () => {
  const held = request('open', ['planned'], 4)
  // The same revision still carries new machine state: an item's status
  const same = request('open', ['queued'], 4)
  expect(newerOf(held, same)).toBe(same)
  const later = request('open', ['planned'], 5)
  expect(newerOf(held, later)).toBe(later)
  expect(newerOf(null, later)).toBe(later)
})
