// Whether the server is there and lets this browser in, fed by every API
// response (apiRequest.ts `send`) and the status poll (statusPoll.ts), with
// the status poll's last memory and health readings. `epoch` rises when an
// outage or a lock ends - a reconnect, or a token the server accepted - so
// what a page read while it could not get through is read again
// (`onReconnect`). Imports nothing from api.ts: api.ts reports here.
import type { HealthInfo, MemoryInfo } from './types'

export type ConnectionState = 'ok' | 'offline' | 'locked' | 'unknown'

export const connection = $state({
  state: 'unknown' as ConnectionState,
  epoch: 0,
  health: null as HealthInfo | null,
  memory: null as MemoryInfo | null,
})

/** The header's word with nothing running: why the server is silent */
export const quietWord = (state: ConnectionState): string =>
  state === 'offline' || state === 'locked' ? state : 'idle'

// Not state: nothing renders from it
let listeners: Array<() => void> = []

// How far the server's clock is ahead of this browser's, in seconds, from
// the `Date` header of its responses. Not state: whoever shows an elapsed
// time reads it on its own tick
let skew = 0
/** `Date` carries whole seconds, so a difference this small is noise */
const SKEW_NOISE_S = 2

/** The `Date` of a response as it arrives. One a cache may hold carries the
 * date it was made, so it is left out */
export function reportServerDate(headers: Headers | undefined): void {
  if (/max-age=[1-9]/.test(headers?.get('cache-control') ?? '')) return
  const at = Date.parse(headers?.get('date') ?? '')
  if (Number.isNaN(at)) return
  // the header is truncated: the middle of its second is the best guess
  const seen = at / 1000 + 0.5 - Date.now() / 1000
  const next = Math.abs(seen) < SKEW_NOISE_S ? 0 : seen
  // keep the last reading unless this one moves it by more than the header's
  // own resolution, so a running figure does not jitter
  if (next === 0 || Math.abs(next - skew) > 1) skew = next
}

/** Now on the server's clock, in unix seconds */
export const serverNow = (): number => Date.now() / 1000 + skew

/** A response came back: a 401 is locked, anything else is through. Leaving
 * offline or locked raises the epoch; leaving the first `unknown` does not,
 * since nothing was read while it held. */
export function reportResponse(status: number): void {
  if (status === 401) {
    connection.state = 'locked'
    return
  }
  const back = connection.state === 'offline' || connection.state === 'locked'
  connection.state = 'ok'
  if (!back) return
  connection.epoch++
  for (const fn of listeners) fn()
}

/** The request never reached the server (fetch's TypeError) */
export function reportUnreachable(): void {
  connection.state = 'offline'
}

/** `fn` runs on every epoch rise, until the returned function stops it */
export function onReconnect(fn: () => void): () => void {
  listeners = [...listeners, fn]
  return () => {
    listeners = listeners.filter((other) => other !== fn)
  }
}
