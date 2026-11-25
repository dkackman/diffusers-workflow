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
