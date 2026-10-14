import { describe, expect, it } from 'vitest'
import { observedText, observedTitle } from './observed'

describe('observedText', () => {
  it('prefers the warm figure - what the next run costs with the model resident', () => {
    expect(
      observedText({ warm_minutes: 12.9, cold_minutes: 14.5, runs: 3 }),
    ).toBe('~12.9 min here')
  })
  it('falls back to the cold figure', () => {
    expect(observedText({ cold_minutes: 0.4, runs: 1 })).toBe('<1 min here')
  })
  it('is null without a figure', () => {
    expect(observedText(null)).toBeNull()
    expect(observedText({ runs: 2 })).toBeNull()
  })
})

describe('observedTitle', () => {
  it('names both figures, the run count and the card', () => {
    expect(
      observedTitle({
        warm_minutes: 12.9,
        warm_runs: 2,
        cold_minutes: 14.5,
        cold_runs: 1,
        runs: 3,
        name: 'RTX 4090',
        since: '2026-10-01T00:00:00Z',
      }),
    ).toBe(
      'measured on this server (RTX 4090): ~12.9 min with the model loaded (2 runs), ~14.5 min from cold (1 run); 3 runs since 2026-10-01',
    )
  })
})
