import { describe, expect, it } from 'vitest'
import { argumentOf, differingArguments, valueText } from './compare'

const meta = (args: Record<string, unknown>, extra = {}) => ({
  arguments: args,
  ...extra,
})

describe('differingArguments', () => {
  it('lists only the keys whose values differ, model and seed first', () => {
    expect(
      differingArguments([
        meta(
          { prompt: 'a', steps: 12, width: 1344 },
          { model_name: 'h3', seed: 1 },
        ),
        meta(
          { prompt: 'a', steps: 30, width: 1344 },
          { model_name: 'h3', seed: 2 },
        ),
      ]),
    ).toEqual(['seed', 'steps'])
  })
  it('a key one side lacks is a difference', () => {
    expect(differingArguments([meta({ lora: 'x' }), meta({})])).toEqual([
      'lora',
    ])
  })
  it('compares structured values by content', () => {
    expect(
      differingArguments([meta({ q: { a: 1 } }), meta({ q: { a: 1 } })]),
    ).toEqual([])
  })
  it('missing metadata counts as unknown, not as different', () => {
    expect(differingArguments([meta({ steps: 1 }), null])).toEqual([])
  })
})

describe('argumentOf / valueText', () => {
  it('reads model and seed from the top level, the rest from arguments', () => {
    const m = meta({ steps: 12 }, { model_name: 'h3', seed: 9 })
    expect(argumentOf(m, 'model')).toBe('h3')
    expect(argumentOf(m, 'seed')).toBe(9)
    expect(argumentOf(m, 'steps')).toBe(12)
  })
  it('prints strings bare, the rest as JSON, cut long', () => {
    expect(valueText('a cat')).toBe('a cat')
    expect(valueText({ a: 1 })).toBe('{"a":1}')
    expect(valueText(undefined)).toBe('—')
    expect(valueText('x'.repeat(300))).toHaveLength(201)
  })
})
