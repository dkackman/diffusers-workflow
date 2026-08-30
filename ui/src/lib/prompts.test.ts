import { describe, expect, it } from 'vitest'
import {
  emptyPrompt,
  knownIntendedModels,
  manifestTextFile,
  parseTags,
  presetForIntendedModel,
  promptListId,
  promptTooltip,
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

describe('promptTooltip', () => {
  const texts = { scenic: 'a scenic landscape', 'minimax/fox': 'a red fox' }

  it('returns the stored text for a prompt reference', () => {
    expect(promptTooltip('prompt:scenic', texts)).toBe('a scenic landscape')
    expect(promptTooltip('prompt:minimax/fox', texts)).toBe('a red fox')
  })

  it('is undefined for anything else, so no title attribute renders', () => {
    expect(promptTooltip('variable:prompt', texts)).toBeUndefined()
    expect(promptTooltip('plain text', texts)).toBeUndefined()
    expect(promptTooltip('prompt:unknown', texts)).toBeUndefined()
    expect(promptTooltip(42, texts)).toBeUndefined()
    expect(promptTooltip(undefined, texts)).toBeUndefined()
  })
})

describe('promptListId', () => {
  it('names the datalist once the value commits to prompt:', () => {
    expect(promptListId('prompt:')).toBe('prompt-references')
    expect(promptListId('prompt:scenic')).toBe('prompt-references')
  })

  it('is undefined before that, so no dropdown pops over ordinary text', () => {
    expect(promptListId('prom')).toBeUndefined()
    expect(promptListId('a cat wearing a hat')).toBeUndefined()
    expect(promptListId('variable:prompt')).toBeUndefined()
    expect(promptListId(42)).toBeUndefined()
    expect(promptListId(undefined)).toBeUndefined()
  })
})

describe('emptyPrompt', () => {
  it('starts with only the required field', () => {
    expect(emptyPrompt()).toEqual({ text: '' })
  })
})
