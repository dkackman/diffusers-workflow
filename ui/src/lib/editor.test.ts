import { describe, expect, it } from 'vitest'
import {
  coerce,
  danglingReferences,
  isReference,
  referenceSuggestions,
  widgetFor,
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
