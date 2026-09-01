import { describe, expect, it } from 'vitest'
import { stepProgress, phaseLabel } from './progress'
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
