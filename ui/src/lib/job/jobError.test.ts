import { describe, expect, it } from 'vitest'
import { explainJobError } from './jobError'

describe('explainJobError', () => {
  it.each([
    'CUDA out of memory. Tried to allocate 2 GiB',
    'torch.OutOfMemoryError: boom',
    'MPS backend out of memory (MPS allocated: 9 GB)',
  ])('out of memory: %s', (text) => {
    expect(explainJobError(text)).toEqual({
      title: 'The card ran out of memory.',
      remedy: 'Run it again with a smaller size or fewer frames.',
      action: 'rerun',
    })
  })

  it('a size not divisible by 8', () => {
    expect(explainJobError('Height and width must be divisible by 8')).toEqual({
      title: 'The width and height must be multiples of 8.',
      remedy: 'Run it again with a size divisible by 8.',
      action: 'rerun',
    })
  })

  it('a gated model', () => {
    for (const text of [
      'GatedRepoError: 403',
      '403 Client Error: Forbidden for url: https://huggingface.co/x',
    ])
      expect(explainJobError(text)).toEqual({
        title: 'This model needs Hugging Face access.',
        remedy:
          'Accept its terms on Hugging Face, then sign in on the server with hf auth login.',
        action: 'models',
      })
  })

  it('an unknown error has no remedy or action', () => {
    expect(explainJobError('KeyError: x')).toEqual({
      title: 'The job stopped with an error.',
      remedy: null,
      action: null,
    })
  })
})
