import { api } from '../api'
import type { JobEvent } from '../types'

/** Events asked for per request, and the most one fetch gathers. A live
 * job's in-memory log can run longer than the cap; Refresh pages on. */
export const EVENT_LOG_PAGE = 500
export const EVENT_LOG_CAP = 5000

export type EventLogFetch = {
  events: JobEvent[]
  /** The seq to page on from: the last event kept, or `after` if none. */
  lastSeq: number
  note: string | null
  /** The cap stopped the paging while the server still had more. */
  capped: boolean
}

/** A job's event log from `after` on, paged at EVENT_LOG_PAGE while the
 * server says a page was cut short, and stopped at `cap` events. Pure over
 * `api.eventLog`: it holds nothing between calls. */
export async function fetchEventLog(
  id: string,
  after = -1,
  cap = EVENT_LOG_CAP,
): Promise<EventLogFetch> {
  const events: JobEvent[] = []
  let cursor = after
  let note: string | null = null
  let more = true
  while (more && events.length < cap) {
    const page = await api.eventLog(id, {
      after: cursor,
      limit: Math.min(EVENT_LOG_PAGE, cap - events.length),
    })
    note = page.note ?? note
    events.push(...page.events)
    more = page.truncated && page.events.length > 0
    cursor = page.last_seq
  }
  return {
    events,
    lastSeq: events.length ? events[events.length - 1].seq : after,
    note,
    capped: more,
  }
}

// The fields a row already shows in its own columns
const ROW_FIELDS = new Set(['seq', 'event', 'at', 'kind'])

/** What a row says beyond its seq, time and kind: the event's message, or
 * the rest of its payload as compact JSON. */
export function eventSummary(event: JobEvent): string {
  if (typeof event.message === 'string') return event.message
  const rest = Object.fromEntries(
    Object.entries(event).filter(([key]) => !ROW_FIELDS.has(key)),
  )
  return Object.keys(rest).length ? JSON.stringify(rest) : ''
}

/** The kind a row is filed under: the event name, with a warning's own
 * `kind` after it, so a stall and a clip read apart. */
export function eventKind(event: JobEvent): string {
  return typeof event.kind === 'string'
    ? `${event.event} · ${event.kind}`
    : event.event
}
