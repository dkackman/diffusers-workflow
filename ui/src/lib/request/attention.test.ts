import { expect, it } from 'vitest'
import {
  attention,
  doneFigure,
  liveItems,
  needsYou,
  orderRequests,
  summaryBands,
} from './attention'
import { summary } from './fixtures'

const line = (status_counts: Record<string, number>, over = {}) =>
  attention(summary({ status_counts, ...over }))

it('says the first thing that holds: failed, running, queued, idle, done', () => {
  expect(line({ failed: 1, running: 1, planned: 2 })).toEqual({
    kind: 'failed',
    text: '1 failed',
    live: true,
  })
  expect(line({ running: 1, queued: 2, done: 3 })).toEqual({
    kind: 'running',
    text: 'running',
    live: true,
  })
  expect(line({ queued: 2, planned: 1 }).text).toBe('queued')
  expect(line({ planned: 3, done: 1 })).toEqual({
    kind: 'idle',
    text: 'idle: 3 planned, nothing running',
    live: false,
  })
  expect(line({ planned: 3 }, { ready: 2 }).text).toBe(
    'idle: 3 planned (2 ready), nothing running',
  )
  expect(line({ done: 2, dropped: 1 }).text).toBe('all done, still open')
  expect(line({ dropped: 1 }).text).toBe('nothing planned yet')
})

it('leaves a closed request to its state tag', () => {
  expect(line({ failed: 1 }, { state: 'abandoned' })).toEqual({
    kind: 'closed',
    text: '',
    live: false,
  })
})

it('counts done of live, dropped items left out', () => {
  const row = summary({
    items: 6,
    status_counts: { done: 2, planned: 3, dropped: 1 },
  })
  expect(liveItems(row)).toBe(5)
  expect(doneFigure(row)).toBe('2/5')
  expect(summaryBands(row).map((b) => [b.tone, b.count])).toEqual([
    ['done', 2],
    ['rest', 3],
  ])
})

it('orders what needs the user first, then what is in flight, then the newest', () => {
  const rows = [
    summary({
      id: 'old',
      status_counts: { done: 1 },
      updated_at: '2026-01-01T00:00:00+00:00',
    }),
    summary({
      id: 'new',
      status_counts: { done: 1 },
      updated_at: '2026-02-01T00:00:00+00:00',
    }),
    summary({ id: 'flying', status_counts: { queued: 1 } }),
    summary({ id: 'ready', status_counts: { planned: 1 }, ready: 1 }),
    summary({
      id: 'failed',
      status_counts: { failed: 1 },
      updated_at: '2026-03-01T00:00:00+00:00',
    }),
    summary({
      id: 'shut',
      state: 'done',
      status_counts: { failed: 1 },
      updated_at: '2026-04-01T00:00:00+00:00',
    }),
  ]
  expect(orderRequests(rows).map((r) => r.id)).toEqual([
    'failed',
    'ready',
    'flying',
    'shut',
    'new',
    'old',
  ])
  expect(rows.map(needsYou)).toEqual([false, false, false, true, true, false])
})

it('counts revised items: idle, and needing the user', () => {
  expect(line({ revised: 1, done: 2 }, { ready: 1 })).toEqual({
    kind: 'idle',
    text: 'idle: 1 revised (1 ready), nothing running',
    live: false,
  })
  expect(line({ revised: 1, planned: 2 }).text).toBe(
    'idle: 1 revised, 2 planned, nothing running',
  )
  expect(needsYou(summary({ status_counts: { revised: 1, done: 1 } }))).toBe(
    true,
  )
  expect(
    needsYou(
      summary({ state: 'done', status_counts: { revised: 1, done: 1 } }),
    ),
  ).toBe(false)
})
