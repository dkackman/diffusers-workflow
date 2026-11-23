import { describe, expect, it } from 'vitest'
import { describeEstimate, describeEstimateTotal, describePlan } from './plan'
import type { Plan } from './types'

const base: Plan = {
  fingerprint: 'sha256:abc',
  steps: 8,
  list_entries: { shots: 5 },
  cached_steps: 0,
  elided_steps: [],
  downloads_required: [
    { repo: 'org/model', gb: 41.2, gated: false, access_blocked: false },
  ],
  estimate: {
    minutes: 42,
    basis: 'catalog',
    device: 'cuda',
    measured_on: 'RTX 3090',
    partial: false,
    unpriced: [],
    runs: null,
    cached_minutes: 42,
  },
  workspace: 'default',
}

describe('describePlan', () => {
  it('reads the work, the figure with its basis, the cache and the downloads', () => {
    const lines = describePlan(base)
    expect(lines.map((line) => line.text)).toEqual([
      '8 steps (shots: 5)',
      'about 42 min on cuda - measured on RTX 3090',
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
      'about 42 min on cuda - re-priced for 5 shots, measured on RTX 3090',
    )
  })

  it('an extrapolated figure says it is an estimate, not a measurement', () => {
    const [, figure] = describePlan({
      ...base,
      estimate: { ...base.estimate, basis: 'derived' },
    })
    expect(figure.text).toBe(
      'about 42 min on cuda - estimated for 5 shots, from a figure measured on RTX 3090',
    )
    expect(figure.tone).toBe('warn')
  })

  it('a figure scaled along moved drivers says what moved', () => {
    const [, figure] = describePlan({
      ...base,
      list_entries: {},
      estimate: {
        ...base.estimate,
        minutes: 26.3,
        basis: 'derived',
        scaled: 'scaled from the curated 7.9 min: steps 30 vs 9',
      },
    })
    expect(figure.text).toBe(
      'about 26.3 min on cuda - estimated (scaled from the curated 7.9 min: steps 30 vs 9), from a figure measured on RTX 3090',
    )
    expect(figure.tone).toBe('warn')
  })

  it('a scaled figure from another accelerator says what moved too', () => {
    const [, figure] = describePlan({
      ...base,
      estimate: {
        ...base.estimate,
        minutes: 80,
        basis: 'other_device',
        device: 'mps',
        scaled: 'scaled from the curated 40 min: steps 18 vs 9',
      },
    })
    expect(figure.text).toBe(
      'no figure for mps - 80 min was measured on RTX 3090 (scaled from the curated 40 min: steps 18 vs 9)',
    )
  })

  it('a re-priced figure scaled along a child says what moved', () => {
    // dw/plan.py names `scaled` for every basis but observed, so the
    // plain branch must carry it as the warn branches do
    const [, figure] = describePlan({
      ...base,
      estimate: {
        ...base.estimate,
        basis: 'per_entry',
        scaled: 'scaled from the curated 7.9 min: steps 30 vs 9',
      },
    })
    expect(figure.text).toBe(
      'about 42 min on cuda - re-priced for 5 shots, measured on RTX 3090 (scaled from the curated 7.9 min: steps 30 vs 9)',
    )
    expect(figure.tone).toBe('plain')
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
        unpriced: [],
        runs: null,
        cached_minutes: null,
      },
    })
    expect(figure.text).toBe('no measured cost')
    expect(figure.tone).toBe('warn')
  })

  it('a partial figure names what it is missing', () => {
    const [, figure] = describePlan({
      ...base,
      estimate: { ...base.estimate, partial: true, unpriced: ['child.json'] },
    })
    expect(figure.text).toContain('plus unpriced: child.json')
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
      'about 42 min on cuda - measured on RTX 3090',
    ])
  })

  it('every cached step means a rerun generates nothing', () => {
    const lines = describePlan({
      ...base,
      cached_steps: 8,
      downloads_required: [],
    })
    expect(lines[2].text).toBe('all 8 steps cached - a job generates nothing')
  })

  it('a download without a size, and a checkpoint URL, are still named', () => {
    const lines = describePlan({
      ...base,
      downloads_required: [
        { repo: 'org/a', gb: null, gated: null, access_blocked: null },
        {
          repo: null,
          url: 'https://x.test/ckpt.safetensors',
          gb: null,
          gated: null,
          access_blocked: null,
        },
      ],
    })
    expect(lines[3].text).toBe(
      'needs download: org/a, https://x.test/ckpt.safetensors',
    )
  })

  it('host RAM reads its figure and basis, plain', () => {
    const cases: [Plan['host_memory'], string][] = [
      [
        { basis: 'observed', projected_mb: 23552, runs: 4 },
        'host RAM about 23 GB (observed)',
      ],
      [
        { basis: 'fit', projected_mb: 30000, runs: 3 },
        'host RAM about 29.3 GB (fit to 3 runs)',
      ],
      [
        { basis: 'fit', projected_mb: 1024, runs: 1 },
        'host RAM about 1 GB (fit to 1 run)',
      ],
      [
        { basis: 'declared', projected_mb: 65536, runs: 0 },
        'host RAM about 64 GB (declared)',
      ],
      [
        { basis: 'inherited', projected_mb: 20480, runs: 2 },
        'host RAM about 20 GB (inherited from an example)',
      ],
    ]
    for (const [memory, text] of cases) {
      const lines = describePlan({ ...base, host_memory: memory })
      expect(lines[2]).toEqual({ text, tone: 'plain' })
    }
  })

  it('an inherited figure with no source says it came from an example', () => {
    const [, figure] = describePlan({
      ...base,
      estimate: {
        ...base.estimate,
        basis: 'inherited',
        inherited_from: undefined,
      },
    })
    expect(figure.text).toBe('about 42 min, inherited from an example')
  })

  it('a low-confidence figure says so and is a warning', () => {
    const [, figure] = describePlan({
      ...base,
      estimate: { ...base.estimate, basis: 'observed', low_confidence: true },
    })
    expect(figure).toEqual({
      text: 'about 42 min on cuda - measured on RTX 3090, low confidence',
      tone: 'warn',
    })
  })

  it('a tempered figure says what it was blended toward', () => {
    const [, figure] = describePlan({
      ...base,
      estimate: {
        ...base.estimate,
        basis: 'observed',
        minutes: 30,
        runs: 2,
        tempered: true,
        observed_minutes: 20,
        curated_minutes: 40,
      },
    })
    expect(figure).toEqual({
      text: 'about 30 min on cuda - measured on RTX 3090, blended toward the curated 40 min from 2 runs at 20 min',
      tone: 'plain',
    })
  })

  it('names the card the run dispatches to when it differs from the device', () => {
    const [, figure] = describePlan({
      ...base,
      estimate: {
        ...base.estimate,
        priced_for: 'cuda:1 NVIDIA GeForce RTX 3090',
      },
    })
    expect(figure.text).toBe(
      'about 42 min on cuda:1 NVIDIA GeForce RTX 3090 - measured on RTX 3090',
    )
    const [, same] = describePlan({
      ...base,
      estimate: { ...base.estimate, priced_for: 'cuda' },
    })
    expect(same.text).toBe('about 42 min on cuda - measured on RTX 3090')
  })

  it('a withheld figure says so and names its source', () => {
    const [, figure] = describePlan({
      ...base,
      estimate: {
        ...base.estimate,
        minutes: null,
        basis: 'unknown',
        measured_on: null,
        cached_minutes: null,
        withheld: 'its frame count cannot be compared to the template',
        inherited_from: 'templates/text-to-video.json',
      },
    })
    expect(figure).toEqual({
      text: 'no figure - withheld (not priced from templates/text-to-video.json)',
      tone: 'warn',
    })
  })

  it('an inherited figure names its template, a warning while low confidence', () => {
    const inherited = {
      ...base.estimate,
      basis: 'inherited' as const,
      inherited_from: 'templates/text-to-image.json',
    }
    const [, firm] = describePlan({ ...base, estimate: inherited })
    expect(firm).toEqual({
      text: 'about 42 min, inherited from templates/text-to-image.json',
      tone: 'plain',
    })
    const [, thin] = describePlan({
      ...base,
      estimate: { ...inherited, low_confidence: true },
    })
    expect(thin).toEqual({
      text: 'about 42 min, inherited from templates/text-to-image.json, low confidence',
      tone: 'warn',
    })
  })

  it('an inherited or unpriced figure names the card last when it differs from the device', () => {
    const card = 'mps Apple M5 Pro (MPS)'
    const inherited = {
      ...base.estimate,
      device: 'mps',
      basis: 'inherited' as const,
      inherited_from: 'templates/text-to-image',
      priced_for: card,
    }
    const [, firm] = describePlan({ ...base, estimate: inherited })
    expect(firm).toEqual({
      text: `about 42 min, inherited from templates/text-to-image - for ${card}`,
      tone: 'plain',
    })
    const [, thin] = describePlan({
      ...base,
      estimate: { ...inherited, low_confidence: true },
    })
    expect(thin).toEqual({
      text: `about 42 min, inherited from templates/text-to-image, low confidence - for ${card}`,
      tone: 'warn',
    })
    const unknown = {
      ...inherited,
      minutes: null,
      basis: 'unknown' as const,
      measured_on: null,
      cached_minutes: null,
    }
    const [, withheld] = describePlan({
      ...base,
      estimate: { ...unknown, withheld: 'frames differ' },
    })
    expect(withheld).toEqual({
      text: `no figure - withheld (not priced from templates/text-to-image) - for ${card}`,
      tone: 'warn',
    })
    const [, reason] = describePlan({
      ...base,
      estimate: {
        ...unknown,
        inherited_from: undefined,
        unpriced_reason: 'no run here',
      },
    })
    expect(reason.text).toBe(`no figure - no run here - for ${card}`)
    const [, bare] = describePlan({
      ...base,
      estimate: { ...unknown, inherited_from: undefined },
    })
    expect(bare.text).toBe(`no measured cost - for ${card}`)
    const [, same] = describePlan({
      ...base,
      estimate: { ...inherited, priced_for: 'mps' },
    })
    expect(same.text).toBe('about 42 min, inherited from templates/text-to-image')
  })

  it('an unpriced reason replaces the bare no measured cost', () => {
    const [, figure] = describePlan({
      ...base,
      estimate: {
        ...base.estimate,
        minutes: null,
        basis: 'unknown',
        measured_on: null,
        cached_minutes: null,
        unpriced_reason: 'task steps only: its length follows its input media',
      },
    })
    expect(figure).toEqual({
      text: 'no figure - task steps only: its length follows its input media',
      tone: 'warn',
    })
  })

  it('a gated and a missing repo are each named in the download line', () => {
    const lines = describePlan({
      ...base,
      downloads_required: [
        { repo: 'org/gated', gb: 12.5, gated: 'manual', access_blocked: true },
        {
          repo: 'org/missing',
          gb: null,
          gated: null,
          access_blocked: null,
          not_found: true,
        },
      ],
    })
    expect(lines[3]).toEqual({
      text: "needs download: org/gated (12.5 GB, gated: this box's token can't reach it), org/missing (not found)",
      tone: 'warn',
    })
  })
})

describe('describeEstimate', () => {
  it('reads one estimate as the plan figure does', () => {
    expect(describeEstimate(base.estimate)).toEqual({
      text: 'about 42 min on cuda - measured on RTX 3090',
      tone: 'plain',
    })
  })

  it('says an absent estimate is not a free one', () => {
    expect(describeEstimate(null)).toEqual({
      text: 'no estimate',
      tone: 'warn',
    })
  })

  it('an unpriced estimate keeps its reason', () => {
    expect(
      describeEstimate({
        ...base.estimate,
        minutes: null,
        basis: 'unknown',
        unpriced_reason: 'no runs yet',
      }).text,
    ).toBe('no figure - no runs yet')
  })
})

describe('describeEstimateTotal', () => {
  it('sums the priced minutes', () => {
    expect(
      describeEstimateTotal([
        base.estimate,
        { ...base.estimate, minutes: 0.25 },
      ]),
    ).toEqual({ text: 'total about 42.3 min', tone: 'plain' })
  })

  it('says how many had no figure, so the total does not read as complete', () => {
    expect(
      describeEstimateTotal([
        base.estimate,
        null,
        { ...base.estimate, minutes: null, basis: 'unknown' },
      ]),
    ).toEqual({ text: 'total about 42 min, plus 2 with no figure', tone: 'warn' })
  })

  it('with nothing priced has no figure', () => {
    expect(describeEstimateTotal([null]).text).toBe('total: no figure')
    expect(describeEstimateTotal([]).text).toBe('total: nothing to run')
  })
})
