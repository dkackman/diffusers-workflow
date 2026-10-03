import { describe, expect, it } from 'vitest'
import {
  estimateEta,
  nextStepTimes,
  phaseLabel,
  stepProgress,
} from './progress'
import type { JobEvent } from './types'

let seq = 0
const event = (name: string, fields: Record<string, unknown> = {}) =>
  ({ seq: seq++, event: name, ...fields }) as unknown as JobEvent

describe('stepProgress', () => {
  it('reports the latest phase of the running step', () => {
    const progress = stepProgress([
      event('step_start', { step: 'gen' }),
      event('phase', { phase: 'loading', detail: 'acme/model' }),
      event('phase', { phase: 'generating', detail: 'acme/model' }),
    ])
    expect(progress.phase).toBe('generating')
    expect(progress.label).toBe('generating acme/model')
  })

  it('drops the previous step’s denoise counter at a new step', () => {
    const progress = stepProgress([
      event('step_start', { step: 'one' }),
      event('pipeline_step', { step: 25, total_steps: 25 }),
      event('step_end', { step: 'one' }),
      event('step_start', { step: 'two' }),
      event('phase', { phase: 'loading', detail: 'other/model' }),
    ])
    // The bar would otherwise render 25/25 over a step that is still loading
    expect(progress.denoise).toBeNull()
    expect(progress.label).toBe('loading other/model')
  })

  it('keeps the counter of the step that is generating', () => {
    const progress = stepProgress([
      event('step_start', { step: 'gen' }),
      event('phase', { phase: 'generating' }),
      event('pipeline_step', { step: 3, total_steps: 25 }),
      event('pipeline_step', { step: 4, total_steps: 25 }),
    ])
    expect(progress.denoise).toEqual({ step: 4, total_steps: 25 })
  })

  it('reports nothing before the first step starts', () => {
    const progress = stepProgress([event('workflow_start', { steps: ['gen'] })])
    expect(progress).toEqual({ phase: null, label: null, denoise: null })
  })
})

describe('phaseLabel', () => {
  it('renders a detailless phase on its own', () => {
    expect(phaseLabel(event('phase', { phase: 'decoding' }))).toBe('decoding')
  })

  it('passes an unknown phase through rather than dropping it', () => {
    expect(phaseLabel(event('phase', { phase: 'uploading' }))).toBe('uploading')
  })
})

describe('estimateEta', () => {
  it('projects the remaining steps at the recent per-step pace', () => {
    // 1s per step, 6 of 10 done: 4 left
    expect(
      estimateEta([0, 1000, 2000, 3000], { step: 6, total_steps: 10 }),
    ).toBe(4)
  })

  it('reads only the last six arrivals, so a slow start fades out', () => {
    expect(
      estimateEta([0, 50000, 51000, 52000, 53000, 54000, 55000], {
        step: 7,
        total_steps: 9,
      }),
    ).toBe(2)
  })

  it('says nothing on fewer than three arrivals', () => {
    expect(estimateEta([0, 1000], { step: 2, total_steps: 10 })).toBeNull()
  })

  it('says nothing without a total, or with nothing left', () => {
    expect(estimateEta([0, 1, 2], { step: 3, total_steps: null })).toBeNull()
    expect(estimateEta([0, 1, 2], null)).toBeNull()
    expect(estimateEta([0, 1, 2], { step: 10, total_steps: 10 })).toBeNull()
  })
})

describe('nextStepTimes', () => {
  it('records a denoise step, keeping a window of seven', () => {
    const times = [1, 2, 3, 4, 5, 6, 7]
    expect(nextStepTimes(times, event('pipeline_step'), 8)).toEqual([
      2, 3, 4, 5, 6, 7, 8,
    ])
  })

  it('starts over at a new step, iteration or status change', () => {
    for (const name of ['step_start', 'iteration_start', 'job_status'])
      expect(nextStepTimes([1, 2, 3], event(name), 9)).toEqual([])
  })

  it('leaves the clocks alone on any other event', () => {
    const times = [1, 2, 3]
    expect(nextStepTimes(times, event('log'), 9)).toBe(times)
  })
})
