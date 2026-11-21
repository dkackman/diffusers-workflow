import { describe, expect, it } from 'vitest'
import {
  attemptsText,
  chainContext,
  chainRows,
  isChained,
  measureText,
  thumbSrc,
} from './chainRows'
import type { ManifestEntry } from './types'

const measure = (
  value: number,
  ceiling: number | null,
  extra: Record<string, unknown> = {},
) => ({
  value,
  ceiling,
  bound: 'max',
  warn_only: false,
  failed: false,
  ...extra,
})

const entry = (shots: any[], files = ['run/chain.mp4']): ManifestEntry =>
  ({ step: 'chain', files, shots }) as unknown as ManifestEntry

const shot = (i: number, extra: Record<string, unknown> = {}) => ({
  name: `segment ${i}`,
  start_frame: (i - 1) * 97,
  num_frames: 97,
  planned_segments: 4,
  ...extra,
})

describe('chainRows', () => {
  it('is null for a step that is not a chain', () => {
    expect(
      chainRows(entry([{ name: 'a', start_frame: 0, num_frames: 10 }])),
    ).toBeNull()
    expect(isChained(entry([]))).toBe(false)
    expect(chainRows(null)).toBeNull()
  })

  it('marks pass, warn and fail from the engine verdict', () => {
    const table = chainRows(
      entry([
        shot(1, { attempts: 1, measures: {} }),
        shot(2, {
          attempts: 1,
          measures: {
            jump_ratio: measure(0.42, 0.6),
            texture_ratio: measure(1.6, 1.5, { warn_only: true, failed: true }),
          },
        }),
        shot(3, {
          attempts: 3,
          seed: 2000049,
          measures: { jump_ratio: measure(0.9, 0.6, { failed: true }) },
        }),
      ]),
      { retries: 2 },
    )!
    const [one, two, three] = table.rows
    expect(one.measureState).toBe('recorded')
    expect(one.measures).toEqual({})
    expect(two.measures.jump_ratio.verdict).toBe('pass')
    expect(two.measures.texture_ratio.verdict).toBe('warn')
    expect(three.measures.jump_ratio.verdict).toBe('fail')
    expect(table.measureNames).toEqual(['jump_ratio', 'texture_ratio'])
    expect(attemptsText(three)).toBe('3 of 3')
    expect(three.keptSeed).toBe(2000049)
    expect(two.keptSeed).toBeNull()
  })

  it('takes the verdict from failed, not value against ceiling', () => {
    // texture_ratio is bounded both ways: below 1/ceiling fails too
    const table = chainRows(
      entry([
        shot(1, {
          attempts: 1,
          measures: {
            texture_ratio: measure(0.5, 1.5, { warn_only: true, failed: true }),
          },
        }),
      ]),
    )!
    expect(table.rows[0].measures.texture_ratio.verdict).toBe('warn')
  })

  it('reads a min bound and a max bound the right way round', () => {
    const table = chainRows(
      entry([
        shot(1, {
          attempts: 1,
          measures: {
            audio_dbfs: measure(-18.234, -70, { bound: 'min' }),
            jump_ratio: measure(0.4213, 0.6),
            cut_ratio: measure(2.5, null),
          },
        }),
      ]),
    )!
    const m = table.rows[0].measures
    expect(measureText(m.audio_dbfs)).toBe('-18.2 ≥ -70')
    expect(measureText(m.jump_ratio)).toBe('0.421 ≤ 0.6')
    expect(measureText(m.cut_ratio)).toBe('2.5 (no ceiling)')
    // engine order, not record order
    expect(table.measureNames).toEqual([
      'jump_ratio',
      'cut_ratio',
      'audio_dbfs',
    ])
  })

  it('shows a measure it does not know by its own name, after the known ones', () => {
    const table = chainRows(
      entry([
        shot(1, {
          attempts: 1,
          measures: {
            zeta_score: measure(3, 5),
            alpha_drift: measure(1, 2, { failed: true }),
            look_ratio: measure(0.2, 0.5),
          },
        }),
      ]),
    )!
    expect(table.measureNames).toEqual([
      'look_ratio',
      'alpha_drift',
      'zeta_score',
    ])
    expect(table.rows[0].measures.alpha_drift.verdict).toBe('fail')
  })

  it('says not recorded for a checked shot with no measures', () => {
    // A job from before measures were recorded: attempts but no measures
    const table = chainRows(
      entry([shot(1, { attempts: 1 }), shot(2, { attempts: 2 })]),
    )!
    expect(table.rows.map((r) => r.measureState)).toEqual([
      'not-recorded',
      'not-recorded',
    ])
    expect(table.measureNames).toEqual([])
  })

  it('records no measure cells for a chain run with the check off', () => {
    const table = chainRows(entry([shot(1), shot(2)]))!
    expect(table.rows.map((r) => r.measureState)).toEqual([
      'unchecked',
      'unchecked',
    ])
    expect(attemptsText(table.rows[0])).toBe('-')
  })

  it('marks a resumed and a repaired shot', () => {
    const table = chainRows(
      entry([
        shot(1, { resumed: true }),
        shot(2, { attempts: 1, measures: { jump_ratio: measure(0.3, 0.6) } }),
        { name: 'segment 3 (repaired)', start_frame: 194, num_frames: 97 },
      ]),
    )!
    const [one, two, three] = table.rows
    expect(one.resumed).toBe(true)
    expect(one.measureState).toBe('resumed')
    expect(two.resumed).toBe(false)
    expect(three.repaired).toBe(true)
    expect(three.measureState).toBe('not-recorded')
    expect(attemptsText(three)).toBe('-')
  })

  it('counts rows against the plan when a take stops short', () => {
    const table = chainRows(entry([shot(1), shot(2)]))!
    expect(table.rows).toHaveLength(2)
    expect(table.planned).toBe(4)
  })

  it('falls back to a bare attempt count when the budget is unknown', () => {
    const table = chainRows(
      entry([shot(1, { attempts: 1 }), shot(2, { attempts: 2 })]),
    )!
    expect(attemptsText(table.rows[0])).toBe('1 attempt')
    expect(attemptsText(table.rows[1])).toBe('2 attempts')
  })

  it('thumbs the segment file when kept, else the take at the start frame', () => {
    const kept = chainRows(
      entry(
        [shot(1, { segment_file: 'chain.segment-001.mp4' }), shot(2)],
        ['run/chain.mp4', 'run/chain.segment-001.mp4'],
      ),
      { fps: 24 },
    )!
    expect(kept.rows[0].thumb).toEqual({
      file: 'run/chain.segment-001.mp4',
      seconds: 0,
      fraction: 0,
    })
    expect(kept.rows[1].thumb?.file).toBe('run/chain.mp4')
    expect(kept.rows[1].thumb?.seconds).toBeCloseTo(97 / 24)
    expect(thumbSrc('/o/chain.mp4', kept.rows[1].thumb!)).toBe(
      '/o/chain.mp4#t=4.042',
    )
    expect(thumbSrc('/o/s.mp4', kept.rows[0].thumb!)).toBe('/o/s.mp4#t=0.001')

    // A segment file the entry does not list sits beside the take
    const beside = chainRows(
      entry([shot(1, { segment_file: 'chain.segment-001.mp4' })]),
    )!
    expect(beside.rows[0].thumb?.file).toBe('run/chain.segment-001.mp4')

    // No frame rate on record: seek by share of the take
    const noFps = chainRows(entry([shot(1), shot(2)]))!
    expect(noFps.rows[1].thumb).toEqual({
      file: 'run/chain.mp4',
      seconds: null,
      fraction: 0.5,
    })
    expect(thumbSrc('/o/chain.mp4', noFps.rows[1].thumb!)).toBe('/o/chain.mp4')
  })
})

describe('chainContext', () => {
  const definition = {
    variables: { frame_rate: 24, retries: 1 },
    steps: [
      {
        name: 'chain',
        pipeline: {
          chain: { check: true, retries: 'variable:retries' },
          arguments: { frame_rate: 'variable:frame_rate' },
        },
      },
      {
        name: 'plain',
        pipeline: { chain: { check: true }, arguments: { fps: 30 } },
      },
    ],
  }

  it('resolves the frame rate and retries through variables, arguments first', () => {
    expect(chainContext(definition, 'chain', { retries: 3 })).toEqual({
      fps: 24,
      retries: 3,
    })
    expect(chainContext(definition, 'chain', {})).toEqual({
      fps: 24,
      retries: 1,
    })
  })

  it('defaults retries to 0 and finds a for_each member by its base name', () => {
    expect(chainContext(definition, 'plain@a')).toEqual({ fps: 30, retries: 0 })
  })

  it('knows neither for a step the definition does not hold', () => {
    expect(chainContext(definition, 'missing')).toEqual({
      fps: null,
      retries: null,
    })
    expect(chainContext(null, 'chain')).toEqual({ fps: null, retries: null })
  })
})
