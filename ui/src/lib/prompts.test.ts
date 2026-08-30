import { describe, expect, it } from 'vitest'
import {
  emptyPrompt,
  knownIntendedModels,
  manifestTextFile,
  parseTags,
  presetForIntendedModel,
} from './prompts'
import type { EnhancerPreset } from './types'

const preset = (overrides: Partial<EnhancerPreset>): EnhancerPreset => ({
  key: 'x',
  label: 'X',
  default_model: 'org/model',
  models: [],
  intended_models: [],
  placeholder: '',
  ...overrides,
})

describe('presetForIntendedModel', () => {
  const presets = [
    preset({ key: 't2i' }),
    preset({ key: 'h3', intended_models: ['minimax-h3'] }),
  ]

  it('matches case-insensitively in either direction', () => {
    expect(presetForIntendedModel(presets, 'MiniMax-H3')?.key).toBe('h3')
    expect(presetForIntendedModel(presets, 'minimax-h3-base')?.key).toBe('h3')
    expect(presetForIntendedModel(presets, 'h3')?.key).toBe('h3')
  })

  it('falls back to the first preset', () => {
    expect(presetForIntendedModel(presets, 'z-image')?.key).toBe('t2i')
    expect(presetForIntendedModel(presets, undefined)?.key).toBe('t2i')
    expect(presetForIntendedModel([], 'anything')).toBeUndefined()
  })
})

describe('manifestTextFile', () => {
  it('finds the first text file across steps', () => {
    const manifest = [
      { step: 'a', files: ['/out/x.png'] },
      { step: 'b', files: ['/out/enhance.0-0.TXT', '/out/other.txt'] },
    ]
    expect(manifestTextFile(manifest)).toBe('/out/enhance.0-0.TXT')
    expect(manifestTextFile([{ step: 'a', files: ['/out/x.png'] }])).toBe(
      undefined,
    )
    expect(manifestTextFile(undefined)).toBe(undefined)
  })
})

describe('parseTags', () => {
  it('trims, drops empties and deduplicates', () => {
    expect(parseTags(' a, b ,, a , c')).toEqual(['a', 'b', 'c'])
    expect(parseTags('')).toEqual([])
  })
})

describe('knownIntendedModels', () => {
  it('unions and sorts across presets', () => {
    const models = knownIntendedModels([
      preset({ intended_models: ['b', 'a'] }),
      preset({ intended_models: ['a', 'c'] }),
    ])
    expect(models).toEqual(['a', 'b', 'c'])
  })
})

describe('emptyPrompt', () => {
  it('starts with only the required field', () => {
    expect(emptyPrompt()).toEqual({ text: '' })
  })
})
