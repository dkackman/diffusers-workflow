import { describe, expect, it } from 'vitest'
import type { HealthInfo, JobDetail } from './types'
import { pageFrame } from './pageFrame.svelte'
import {
  BASE_TITLE,
  endedSince,
  endedTitle,
  nextWatch,
  percentDone,
  runningIds,
  tabTitle,
} from './jobWatch'

const health = (ids: (string | null)[], current: string | null = null) =>
  ({
    current_job: current ?? ids.find(Boolean) ?? null,
    workers: ids.map((id) => ({ current_job: id })),
  }) as unknown as HealthInfo

const progress = (p: Partial<NonNullable<JobDetail['progress']>>) =>
  p as JobDetail['progress']

describe('runningIds', () => {
  it('names every card’s job once', () => {
    expect(runningIds(health(['a', null, 'b']))).toEqual(['a', 'b'])
  })
  it('falls back to current_job when workers is absent', () => {
    expect(runningIds({ current_job: 'x' } as unknown as HealthInfo)).toEqual([
      'x',
    ])
  })
  it('is empty for no health', () => {
    expect(runningIds(null)).toEqual([])
  })
})

describe('endedSince', () => {
  it('is what was running and is not now', () => {
    expect(endedSince(['a', 'b'], ['b', 'c'])).toEqual(['a'])
  })
})

describe('nextWatch', () => {
  it('first poll learns what is running and ends nothing', () => {
    expect(nextWatch(null, health(['a', 'b']))).toEqual({
      watched: ['a', 'b'],
      ended: [],
      running: ['a', 'b'],
    })
  })
  it('a failed poll ends nothing and keeps what was watched', () => {
    expect(nextWatch(['a', 'b'], null)).toEqual({
      watched: ['a', 'b'],
      ended: [],
      running: ['a', 'b'],
    })
    expect(nextWatch(null, null).watched).toBeNull()
  })
  it('a later real poll ends only the job that vanished', () => {
    const next = nextWatch(['a', 'b'], health(['b', 'c']))
    expect(next.ended).toEqual(['a'])
    expect(next.watched).toEqual(['b', 'c'])
  })
  it('a real poll after a failed one still ends a vanished job', () => {
    const kept = nextWatch(['a'], null).watched
    expect(nextWatch(kept, health([])).ended).toEqual(['a'])
  })
})

describe('adopted jobs', () => {
  it('an adopted job that finished before the next poll ends once', () => {
    const adopted = new Set(['x'])
    const next = nextWatch(['a'], health(['a']), adopted)
    expect(next.ended).toEqual(['x'])
    expect(nextWatch(null, health([]), adopted).ended).toEqual(['x'])
    expect(nextWatch(['x'], health([]), adopted).ended).toEqual(['x'])
  })
  it('an adopted job is held while anything is queued', () => {
    const queued = { ...health([]), queued: 1 } as HealthInfo
    expect(nextWatch(null, queued, new Set(['x'])).ended).toEqual([])
  })
})

describe('percentDone', () => {
  it('counts finished steps plus the denoise fraction of the current one', () => {
    expect(
      percentDone(
        progress({
          step_index: 1,
          total_steps: 4,
          denoise_step: 5,
          denoise_total_steps: 10,
        }),
      ),
    ).toBe(37)
  })
  it('never claims 100 while running', () => {
    expect(
      percentDone(
        progress({
          step_index: 3,
          total_steps: 4,
          denoise_step: 10,
          denoise_total_steps: 10,
        }),
      ),
    ).toBe(99)
  })
  it('is null with no step count', () => {
    expect(percentDone(null)).toBeNull()
    expect(percentDone(progress({ total_steps: 0 }))).toBeNull()
  })
})

describe('tabTitle', () => {
  it('is the app name when idle', () => {
    expect(tabTitle(null)).toBe(BASE_TITLE)
  })
  it('shows percent, the workflow leaf name and other running jobs', () => {
    const job = {
      workflow: 'minimax/ref-to-video',
      workflow_name: 'templates/minimax/ref-to-video',
      progress: progress({ step_index: 0, total_steps: 2 }),
    }
    expect(tabTitle(job, 1)).toBe(`▶ 0% · ref-to-video +1 — ${BASE_TITLE}`)
  })
  it("the running title keeps the page's name", () => {
    const job = {
      workflow: 'minimax/ref-to-video',
      workflow_name: 'templates/minimax/ref-to-video',
      progress: progress({ step_index: 0, total_steps: 2 }),
    }
    pageFrame.title = 'Gallery'
    try {
      expect(tabTitle(job, 1)).toBe(
        '▶ 0% · ref-to-video +1 — Gallery · diffusers-workflow',
      )
      expect(endedTitle({ ...job, status: 'succeeded' })).toBe(
        '✓ ref-to-video — Gallery · diffusers-workflow',
      )
    } finally {
      pageFrame.title = ''
    }
  })
  it('leaves the percent out when there is none', () => {
    expect(
      tabTitle({ workflow: 'wf', workflow_name: null, progress: null }),
    ).toBe(`▶ wf — ${BASE_TITLE}`)
  })
})

describe('endedTitle', () => {
  it('marks the outcome', () => {
    const base = { workflow: 'wf', workflow_name: null, progress: null }
    expect(endedTitle({ ...base, status: 'succeeded' })).toBe(
      `✓ wf — ${BASE_TITLE}`,
    )
    expect(endedTitle({ ...base, status: 'failed' })).toBe(
      `✗ wf — ${BASE_TITLE}`,
    )
    expect(endedTitle({ ...base, status: 'cancelled' })).toBe(
      `⊘ wf — ${BASE_TITLE}`,
    )
  })
})
