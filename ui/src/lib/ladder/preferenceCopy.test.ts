import { describe, expect, it } from 'vitest'
import {
  downloadNote,
  keyLabel,
  reasonText,
  refusalLead,
} from './preferenceCopy'

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
    expect(reasonText('cpu-only')).toBe('needs the CPU')
    expect(reasonText('cuda/mps-only')).toBe('needs CUDA or Apple MPS')
  })
  it('names the devices the same in any order', () => {
    expect(reasonText('mps/cuda-only')).toBe('needs CUDA or Apple MPS')
    expect(reasonText('cpu/cuda-only')).toBe('needs CUDA or the CPU')
    expect(reasonText('cuda/cpu-only')).toBe('needs CUDA or the CPU')
    expect(reasonText('cpu/mps/cuda-only')).toBe(
      'needs CUDA, Apple MPS or the CPU',
    )
  })
  it('leaves a device it does not know as given', () => {
    expect(reasonText('xpu-only')).toBe('xpu-only')
    expect(reasonText('no-device-only')).toBe('no-device-only')
  })
  it('words the known-blocked skip', () => {
    expect(reasonText('gated, no access (last full check)')).toBe(
      'gated: no access (as of the last full check)',
    )
  })
  it('leaves any other reason as it came', () => {
    const r = 'needs 32 GB VRAM on cuda, the smallest card has 24 GB'
    expect(reasonText(r)).toBe(r)
  })
})

describe('downloadNote', () => {
  it('turns a sized download into before-first-job copy', () => {
    expect(downloadNote(['will download 41.2 GB'])).toEqual({
      text: 'downloads 41.2 GB before its first job',
    })
  })
  it('says weights when there is no size', () => {
    const weights = { text: 'downloads its weights before its first job' }
    expect(downloadNote(['will download its weights'])).toEqual(weights)
    expect(downloadNote(['not downloaded'])).toEqual(weights)
  })
  it('is none without a download note', () => {
    expect(downloadNote([])).toBeNull()
    expect(downloadNote(['something else'])).toBeNull()
  })
})

describe('refusalLead', () => {
  it('says nothing fits when the refusal passed models over', () => {
    expect(
      refusalLead('tier best: no target fits this host - a: cuda-only', [
        { workflow: 'a', reason: 'cuda-only' },
      ]),
    ).toBe('Nothing fits this machine')
  })
  it('gives any other refusal a neutral lead and its message', () => {
    expect(refusalLead("names a target not in the catalog: 'x'", [])).toBe(
      "This tier is refused: names a target not in the catalog: 'x'",
    )
  })
})
