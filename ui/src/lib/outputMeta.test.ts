import { describe, expect, it } from 'vitest'
import { describeOutputMeta, promptText } from './outputMeta'

describe('promptText', () => {
  it('shows a string, joins a list of strings, and is empty otherwise', () => {
    expect(promptText('a cat')).toBe('a cat')
    expect(promptText(['a cat', 7, 'a dog'])).toBe('a cat\na dog')
    expect(promptText(undefined)).toBe('')
  })
})

describe('describeOutputMeta', () => {
  it('reads the model, seed and prompts embedded in an output', () => {
    expect(
      describeOutputMeta({
        model_name: 'org/model',
        seed: 7,
        arguments: { prompt: 'a cat', negative_prompt: ['blur'] },
      }),
    ).toEqual({
      model: 'org/model',
      seed: 7,
      prompt: 'a cat',
      negativePrompt: 'blur',
    })
  })

  it('falls back to the argument seed, and to nothing', () => {
    expect(describeOutputMeta({ arguments: { seed: 3 } }).seed).toBe(3)
    expect(describeOutputMeta(null)).toEqual({
      model: '',
      seed: undefined,
      prompt: '',
      negativePrompt: '',
    })
  })
})
