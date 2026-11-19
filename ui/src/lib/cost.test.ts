import { describe, expect, it } from 'vitest'
import { costBasisWords, costMeasuredAt, costMeasuredOn } from './cost'

describe('costMeasuredAt', () => {
  it('prints a scalar as itself, in map order', () => {
    expect(costMeasuredAt({ steps: 30, width: 1024, fast: true })).toBe(
      'steps=30, width=1024, fast=true',
    )
    expect(costMeasuredAt({ weights_dtype: 'bf16' })).toBe('weights_dtype=bf16')
  })

  it('prints a list as its entry count', () => {
    expect(costMeasuredAt({ shots: ['a', 'b', 'c'] })).toBe('shots=3 entries')
    expect(costMeasuredAt({ shots: [] })).toBe('shots=0 entries')
  })

  it('prints null as unset', () => {
    expect(costMeasuredAt({ image: null })).toBe('image=unset')
  })

  it('prints an object as {…}', () => {
    expect(costMeasuredAt({ cache: { kind: 'fbc' } })).toBe('cache={…}')
  })

  it('cuts a value over 40 characters', () => {
    const long = 'x'.repeat(41)
    expect(costMeasuredAt({ p: long })).toBe(`p=${'x'.repeat(40)}…`)
    expect(costMeasuredAt({ p: 'y'.repeat(40) })).toBe(`p=${'y'.repeat(40)}`)
  })

  it('is empty for an empty or missing map', () => {
    expect(costMeasuredAt({})).toBe('')
    expect(costMeasuredAt(undefined)).toBe('')
    expect(costMeasuredAt(null)).toBe('')
  })
})

describe('costBasisWords', () => {
  it("says curated as a maintainer's figure", () => {
    expect(costBasisWords('curated')).toBe("a maintainer's figure")
  })
  it('prints a basis it does not know raw', () => {
    expect(costBasisWords('measured-in-ci')).toBe('measured-in-ci')
  })
})

describe('costMeasuredOn', () => {
  const cost = [{ device: 'cuda', name: 'RTX 4090', vram_gb: 22, minutes: 3 }]
  it('names the first entry and its settings', () => {
    expect(costMeasuredOn(cost, { steps: 30 })).toBe(
      'measured on RTX 4090 at steps=30',
    )
  })
  it('drops the at clause without drivers and falls back to the device', () => {
    expect(
      costMeasuredOn([{ device: 'mps', vram_gb: 1, minutes: 1 }], {}),
    ).toBe('measured on mps')
  })
  it('is empty with no cost', () => {
    expect(costMeasuredOn(null, { steps: 30 })).toBe('')
  })
})
