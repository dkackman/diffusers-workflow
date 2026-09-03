import { describe, expect, it } from 'vitest'
import {
  coerce,
  danglingReferences,
  isLongText,
  isReference,
  mediaKindFor,
  mediaLocation,
  referenceSuggestions,
  widgetFor,
  withMediaLocation,
  QUANT_PRESETS,
} from './editor'
import type { PipelineParameter } from './types'

const param = (overrides: Partial<PipelineParameter>): PipelineParameter => ({
  name: 'x',
  required: false,
  default: null,
  annotation: null,
  ...overrides,
})

describe('widgetFor', () => {
  it('follows the existing value type first', () => {
    expect(widgetFor(param({}), 3)).toBe('number')
    expect(widgetFor(param({}), true)).toBe('boolean')
    expect(widgetFor(param({}), { a: 1 })).toBe('json')
    expect(widgetFor(param({}), [1, 2])).toBe('json')
  })

  it('engine references are always text, whatever the annotation says', () => {
    expect(widgetFor(param({ annotation: 'int' }), 'variable:steps')).toBe(
      'text',
    )
    expect(
      widgetFor(param({ annotation: 'bool' }), 'previous_result:gen'),
    ).toBe('text')
  })

  it('falls back to the annotation for empty values', () => {
    expect(widgetFor(param({ annotation: 'int | None' }), '')).toBe('number')
    expect(widgetFor(param({ annotation: 'float' }), '')).toBe('number')
    expect(widgetFor(param({ annotation: 'bool' }), '')).toBe('boolean')
  })
})

describe('coerce', () => {
  it('numbers become numbers, but references and non-numbers stay strings', () => {
    expect(coerce('number', '25')).toBe(25)
    expect(coerce('number', '0.5')).toBe(0.5)
    expect(coerce('number', 'variable:steps')).toBe('variable:steps')
    expect(coerce('number', 'not-a-number')).toBe('not-a-number')
  })

  it('json fields parse when valid and stay raw when not', () => {
    expect(coerce('json', '{"a": 1}')).toEqual({ a: 1 })
    expect(coerce('json', 'broken{')).toBe('broken{')
  })
})

describe('isReference', () => {
  it('recognizes every engine reference prefix', () => {
    for (const prefix of [
      'variable:',
      'previous_result:',
      'constant:',
      'prompt:',
    ]) {
      expect(isReference(prefix + 'x')).toBe(true)
    }
    expect(isReference('plain text')).toBe(false)
    expect(isReference(42)).toBe(false)
  })
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

describe('quantization presets', () => {
  it('every preset carries a config_type and constructor arguments', () => {
    for (const [name, preset] of Object.entries(QUANT_PRESETS)) {
      if (name === 'none') {
        expect(preset).toBeNull()
      } else {
        expect(preset!.config_type).toBeTruthy()
        expect(Object.keys(preset!.arguments).length).toBeGreaterThan(0)
      }
    }
  })
})

describe('referenceSuggestions', () => {
  const workflow = {
    variables: { prompt: 'x', steps: 9 },
    steps: [
      { name: 'gen', result: { content_type: 'video/mp4' } },
      { name: 'refine' },
      { name: 'last' },
    ],
  }

  it('offers variables plus only earlier steps', () => {
    const forLast = referenceSuggestions(workflow, 2)
    expect(forLast).toContain('variable:prompt')
    expect(forLast).toContain('previous_result:gen')
    expect(forLast).toContain('previous_result:refine')
    expect(forLast).not.toContain('previous_result:last')

    const forFirst = referenceSuggestions(workflow, 0)
    expect(forFirst.filter((s) => s.startsWith('previous_result'))).toEqual([])
  })

  it('adds media suffixes for video-producing steps', () => {
    const forSecond = referenceSuggestions(workflow, 1)
    expect(forSecond).toContain('previous_result:gen.frames')
    expect(forSecond).toContain('previous_result:gen.audio')
    expect(forSecond).not.toContain('previous_result:refine.frames')
  })

  it('offers every stored prompt when the library is supplied', () => {
    const suggestions = referenceSuggestions(workflow, 0, [
      'scenic',
      'minimax/fox',
    ])
    expect(suggestions).toContain('prompt:scenic')
    expect(suggestions).toContain('prompt:minimax/fox')
    expect(referenceSuggestions(workflow, 0)).not.toContain('prompt:scenic')
  })
})

describe('danglingReferences', () => {
  it('flags undeclared variables and unknown or later steps', () => {
    const workflow = {
      variables: { prompt: 'x' },
      steps: [
        {
          name: 'gen',
          pipeline: {
            arguments: {
              prompt: 'variable:prompt',
              image: 'previous_result:later',
            },
          },
        },
        {
          name: 'later',
          pipeline: { arguments: { size: 'variable:missing' } },
        },
      ],
    }
    const problems = danglingReferences(workflow)
    expect(problems).toHaveLength(2)
    expect(problems[0]).toContain('previous_result:later')
    expect(problems[1]).toContain('variable:missing')
  })

  it('accepts valid references, property paths and nested values', () => {
    const workflow = {
      variables: { prompt: 'x' },
      steps: [
        { name: 'gen', result: { content_type: 'video/mp4' } },
        {
          name: 'use',
          task: {
            arguments: {
              frames: 'previous_result:gen.frames',
              nested: [{ deep: 'variable:prompt' }],
            },
          },
        },
      ],
    }
    expect(danglingReferences(workflow)).toEqual([])
  })

  it('checks prompt references only against a supplied library', () => {
    const workflow = {
      steps: [
        { name: 'gen', pipeline: { arguments: { prompt: 'prompt:scenic' } } },
      ],
    }
    // no listing - the server resolves these at run time, nothing to flag
    expect(danglingReferences(workflow)).toEqual([])
    expect(danglingReferences(workflow, ['scenic'])).toEqual([])
    const problems = danglingReferences(workflow, ['other'])
    expect(problems).toHaveLength(1)
    expect(problems[0]).toContain('prompt:scenic')
  })
})

describe('isLongText', () => {
  it('long strings and multi-line strings are long', () => {
    expect(isLongText('x'.repeat(61))).toBe(true)
    expect(isLongText('one\ntwo')).toBe(true)
  })

  it('short single-line strings and non-strings are not', () => {
    expect(isLongText('a cat')).toBe(false)
    expect(isLongText('')).toBe(false)
    expect(isLongText(25)).toBe(false)
    expect(isLongText(null)).toBe(false)
    expect(isLongText({ a: 1 })).toBe(false)
  })
})
