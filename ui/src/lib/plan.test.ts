import { describe, expect, it } from 'vitest'
import { describePlan } from './plan'
import type { Plan } from './types'

const base: Plan = {
  fingerprint: 'sha256:abc',
  steps: 8,
  list_entries: { shots: 5 },
  cached_steps: 0,
  downloads_required: [{ repo: 'org/model', gb: 41.2 }],
  estimate: {
    minutes: 42,
    basis: 'catalog',
    device: 'cuda',
    measured_on: 'RTX 3090',
    partial: false,
  },
}

describe('describePlan', () => {
  it('reads the work, the figure with its basis, the cache and the downloads', () => {
    const lines = describePlan(base)
    expect(lines.map((line) => line.text)).toEqual([
      '8 steps (shots: 5)',
      '~42 min on cuda - measured on RTX 3090',
      '0 of 8 steps cached',
      'needs download: org/model (41.2 GB)',
    ])
    expect(lines.map((line) => line.tone)).toEqual([
      'plain',
      'plain',
      'plain',
      'warn',
    ])
  })

  it('re-priced for the list passed says so', () => {
    const [, figure] = describePlan({
      ...base,
      estimate: { ...base.estimate, basis: 'per_entry' },
    })
    expect(figure.text).toBe(
      '~42 min on cuda - re-priced for 5 shots, measured on RTX 3090',
    )
  })

  it('a figure from another accelerator is a warning, not a quote', () => {
    const [, figure] = describePlan({
      ...base,
      estimate: { ...base.estimate, basis: 'other_device', device: 'mps' },
    })
    expect(figure.text).toBe(
      'no figure for mps - 42 min was measured on RTX 3090',
    )
    expect(figure.tone).toBe('warn')
  })

  it('no cost block is said plainly', () => {
    const [, figure] = describePlan({
      ...base,
      estimate: {
        minutes: null,
        basis: 'unknown',
        device: 'cuda',
        measured_on: null,
        partial: false,
      },
    })
    expect(figure.text).toBe('no measured cost')
    expect(figure.tone).toBe('warn')
  })

  it('a partial figure names what it is missing', () => {
    const [, figure] = describePlan({
      ...base,
      estimate: { ...base.estimate, partial: true },
    })
    expect(figure.text).toContain('plus a composed workflow with no cost')
  })

  it('an unknown cache answer and a single step read naturally', () => {
    const lines = describePlan({
      ...base,
      steps: 1,
      list_entries: {},
      cached_steps: null,
      downloads_required: [],
    })
    expect(lines.map((line) => line.text)).toEqual([
      '1 step',
      '~42 min on cuda - measured on RTX 3090',
    ])
  })

  it('every cached step means a rerun generates nothing', () => {
    const lines = describePlan({ ...base, cached_steps: 8, downloads_required: [] })
    expect(lines[2].text).toBe('all 8 steps cached - a run generates nothing')
  })

  it('a download without a size, and a checkpoint URL, are still named', () => {
    const lines = describePlan({
      ...base,
      downloads_required: [
        { repo: 'org/a', gb: null },
        { repo: null, url: 'https://x.test/ckpt.safetensors', gb: null },
      ],
    })
    expect(lines[3].text).toBe(
      'needs download: org/a, https://x.test/ckpt.safetensors',
    )
  })
})
