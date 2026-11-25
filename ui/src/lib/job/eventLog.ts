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

/** A row that is its own message; with none, the event's name */
const said = (name: string) => (e: JobEvent) =>
  typeof e.message === 'string' && e.message ? e.message : name

type Sentence = { text: (event: JobEvent) => string; used: string[] }

const word = (value: unknown) => String(value ?? '')

/** The events a row words in plain sentences, and the payload fields each
 * sentence already says (so they are not repeated in the folded data). */
const SENTENCES: Record<string, Sentence> = {
  step_start: { text: (e) => `Started ${word(e.step)}`, used: ['step'] },
  step_end: { text: (e) => `Finished ${word(e.step)}`, used: ['step'] },
  pipeline_step: {
    text: (e) =>
      e.total_steps == null
        ? `Denoise ${word(e.step)}`
        : `Denoise ${word(e.step)} of ${word(e.total_steps)}`,
    used: ['step', 'total_steps'],
  },
  workflow_start: { text: () => 'Started the workflow', used: [] },
  job_status: { text: (e) => `Status: ${word(e.status)}`, used: ['status'] },
  queue_wait: { text: () => 'Waiting for a card', used: [] },
  warning: { text: said('warning'), used: ['message'] },
  log: { text: said('log'), used: ['message'] },
}

/** What a row says beyond its seq, time and kind: a sentence for the events
 * the table names, else the event's own name, and the rest of the payload as
 * indented JSON (null when nothing is left) to fold behind Details. */
export function eventSummary(event: JobEvent): {
  text: string
  data: string | null
} {
  const sentence = SENTENCES[event.event]
  const text = sentence ? sentence.text(event) : event.event
  const used = new Set(sentence?.used)
  const rest = Object.fromEntries(
    Object.entries(event).filter(
      ([key]) => !ROW_FIELDS.has(key) && !used.has(key),
    ),
  )
  return {
    text,
    data: Object.keys(rest).length ? JSON.stringify(rest, null, 2) : null,
  }
}

/** The kind a row is filed under: the event name, with a warning's own
 * `kind` after it, so a stall and a clip read apart. */
export function eventKind(event: JobEvent): string {
  return typeof event.kind === 'string'
    ? `${event.event} · ${event.kind}`
    : event.event
}
