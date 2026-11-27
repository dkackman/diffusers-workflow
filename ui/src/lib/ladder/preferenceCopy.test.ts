import { describe, expect, it } from 'vitest'
import { downloadNote, keyLabel, reasonText } from './preferenceCopy'

describe('keyLabel', () => {
  it('reads a key as a sentence', () => {
    expect(keyLabel('text_to_image')).toBe('Text to image')
    expect(keyLabel('image-edit')).toBe('Image edit')
  })
})

describe('reasonText', () => {
  it('names a device-only reason in words', () => {
    expect(reasonText('cuda-only')).toBe('needs CUDA')
    expect(reasonText('mps-only')).toBe('needs Apple MPS')
    expect(reasonText('cpu-only')).toBe('CPU only')
    expect(reasonText('cuda/mps-only')).toBe('needs CUDA or Apple MPS')
  })
  it('leaves any other reason as it came', () => {
    const r = 'needs 32 GB VRAM on cuda, the smallest card has 24 GB'
    expect(reasonText(r)).toBe(r)
    expect(reasonText('gated, no access (last full check)')).toBe(
      'gated, no access (last full check)',
    )
  })
})

describe('downloadNote', () => {
  it('turns a sized download into first-use copy', () => {
    expect(downloadNote(['will download 41.2 GB'])).toEqual({
      text: 'downloads 41.2 GB on first use',
    })
  })
  it('says weights when there is no size', () => {
    const weights = { text: 'downloads its weights on first use' }
    expect(downloadNote(['will download its weights'])).toEqual(weights)
    expect(downloadNote(['not downloaded'])).toEqual(weights)
  })
  it('is none without a download note', () => {
    expect(downloadNote([])).toBeNull()
    expect(downloadNote(['something else'])).toBeNull()
  })
})
