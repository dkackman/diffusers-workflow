import { describe, expect, it } from 'vitest'
import {
  mediaKindFor,
  mediaLocation,
  withMediaLocation,
} from './mediaArguments'
import type { PipelineParameter } from './types'

const param = (overrides: Partial<PipelineParameter>): PipelineParameter => ({
  name: 'x',
  required: false,
  default: null,
  annotation: null,
  ...overrides,
})

describe('mediaKindFor', () => {
  it('reads image/video off the signature annotation when introspection has one', () => {
    expect(mediaKindFor(param({ annotation: 'PipelineImageInput' }), 'x')).toBe(
      'image',
    )
    expect(
      mediaKindFor(param({ doc_type: '`PIL.Image.Image`, *optional*' }), 'x'),
    ).toBe('image')
    expect(mediaKindFor(param({ annotation: 'VideoInput' }), 'x')).toBe('video')
    expect(mediaKindFor(param({ annotation: 'int' }), 'x')).toBeNull()
  })

  it('falls back to the naming convention when introspection carries no type at all', () => {
    // dw/introspection.py's describe_task hands back annotation: None for
    // every task argument (e.g. depth_estimator's "image")
    const untyped = param({ annotation: null })
    expect(mediaKindFor(untyped, 'image')).toBe('image')
    expect(mediaKindFor(untyped, 'mask_image')).toBe('image')
    expect(mediaKindFor(untyped, 'control_image')).toBe('image')
    expect(mediaKindFor(untyped, 'video')).toBe('video')
    expect(mediaKindFor(undefined, 'image')).toBe('image')
    expect(mediaKindFor(untyped, 'prompt')).toBeNull()
    expect(mediaKindFor(untyped, 'num_inference_steps')).toBeNull()
  })

  it('trusts a real, non-media annotation over the name heuristic', () => {
    // the name-fallback only fires when introspection gave no type at all -
    // an "image" parameter that is genuinely typed as something else (e.g.
    // a scale factor) must not be swept into the media widget
    expect(mediaKindFor(param({ annotation: 'float' }), 'image')).toBeNull()
  })
})

describe('mediaLocation / withMediaLocation', () => {
  it('reads the location out of either value shape the engine accepts', () => {
    expect(mediaLocation('photo.png')).toBe('photo.png')
    expect(mediaLocation({ location: 'photo.png' })).toBe('photo.png')
    expect(mediaLocation({ media_type: 'image', location: 'photo.png' })).toBe(
      'photo.png',
    )
    expect(mediaLocation(undefined)).toBe('')
    expect(mediaLocation({})).toBe('')
  })

  it('writes a new location back preserving the value shape it started as', () => {
    expect(withMediaLocation('old.png', 'new.png')).toBe('new.png')
    expect(withMediaLocation('', 'new.png')).toBe('new.png')
    expect(
      withMediaLocation(
        { media_type: 'image', location: 'old.png' },
        'new.png',
      ),
    ).toEqual({ media_type: 'image', location: 'new.png' })
  })
})
