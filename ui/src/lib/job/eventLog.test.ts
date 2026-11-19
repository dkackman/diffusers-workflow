import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api'
import type { EventLogReply, JobEvent } from '../types'
import {
  EVENT_LOG_CAP,
  eventKind,
  eventSummary,
  fetchEventLog,
} from './eventLog'

vi.mock('../api', () => ({ api: { eventLog: vi.fn() } }))

/** A server holding events 0..total-1, paged the way the route pages:
 * after is exclusive, truncated when more lie past the page. */
function serve(total: number, note: string | null = null) {
  vi.mocked(api.eventLog).mockImplementation(
    async (id, { after = -1, limit = 200 } = {}) => {
      const pending: JobEvent[] = []
      for (let seq = after + 1; seq < total; seq++)
        pending.push({ seq, event: 'log', message: `line ${seq}` })
      const page = pending.slice(0, limit)
      return {
        id,
        status: 'succeeded',
        events: page,
        last_seq: page.length ? page[page.length - 1].seq : Math.max(after, -1),
        truncated: pending.length > page.length,
        note,
      } satisfies EventLogReply
    },
  )
}

afterEach(() => vi.mocked(api.eventLog).mockReset())

describe('fetchEventLog', () => {
  it('reads one page when the server says it is whole', async () => {
    serve(3, 'only the tail was kept')
    const got = await fetchEventLog('j1')
    expect(got.events.map((e) => e.seq)).toEqual([0, 1, 2])
    expect(got.note).toBe('only the tail was kept')
    expect(got.capped).toBe(false)
    expect(got.lastSeq).toBe(2)
    expect(vi.mocked(api.eventLog).mock.calls).toEqual([
      ['j1', { after: -1, limit: 500 }],
    ])
  })

  it('follows truncated, paging on from each last_seq', async () => {
    serve(1200)
    const got = await fetchEventLog('j1')
    expect(got.events.length).toBe(1200)
    expect(got.events.map((e) => e.seq)).toEqual([...Array(1200).keys()])
    expect(got.capped).toBe(false)
    expect(
      vi.mocked(api.eventLog).mock.calls.map(([, query]) => query?.after),
    ).toEqual([-1, 499, 999])
  })

  it('stops at the cap and says the server holds more', async () => {
    serve(7000)
    const got = await fetchEventLog('j1')
    expect(got.events.length).toBe(EVENT_LOG_CAP)
    expect(EVENT_LOG_CAP).toBe(5000)
    expect(got.capped).toBe(true)
    expect(got.lastSeq).toBe(4999)
    expect(vi.mocked(api.eventLog).mock.calls.length).toBe(10)
  })

  it('is not capped when the log is exactly the cap', async () => {
    serve(5000)
    const got = await fetchEventLog('j1')
    expect(got.events.length).toBe(5000)
    expect(got.capped).toBe(false)
  })

  it('pages on from a given seq, and keeps it when nothing is new', async () => {
    serve(10)
    const got = await fetchEventLog('j1', 9)
    expect(got.events).toEqual([])
    expect(got.lastSeq).toBe(9)
    expect(vi.mocked(api.eventLog).mock.calls[0][1]?.after).toBe(9)
  })
})

describe('row text', () => {
  it("shows a message as is, or the payload's rest as compact JSON", () => {
    expect(eventSummary({ seq: 1, event: 'log', at: 2, message: 'hi' })).toBe(
      'hi',
    )
    expect(
      eventSummary({ seq: 2, event: 'step_start', at: 3, step: 'gen' }),
    ).toBe('{"step":"gen"}')
    expect(eventSummary({ seq: 3, event: 'job_done' })).toBe('')
  })

  it('files a warning under its own kind beside the event name', () => {
    expect(eventKind({ seq: 1, event: 'warning', kind: 'phase_stall' })).toBe(
      'warning · phase_stall',
    )
    expect(eventKind({ seq: 1, event: 'log' })).toBe('log')
  })
})
