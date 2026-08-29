import { describe, expect, it } from 'vitest'
import { coerce, isReference, widgetFor, QUANT_PRESETS } from './editor'
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
    expect(widgetFor(param({ annotation: 'int' }), 'variable:steps')).toBe('text')
    expect(widgetFor(param({ annotation: 'bool' }), 'previous_result:gen')).toBe('text')
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
    for (const prefix of ['variable:', 'previous_result:', 'constant:']) {
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
