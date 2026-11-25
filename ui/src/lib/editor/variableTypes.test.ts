import { describe, expect, it } from 'vitest'
import {
  asType,
  inferValue,
  typeOf,
  variableNameProblem,
} from './variableTypes'

const RULE =
  'A variable name starts with a letter or _ and holds letters, numbers, _ and -.'

describe('variable names', () => {
  it("follow the engine's rule", () => {
    for (const ok of ['seed', '_x', 'a-b_9', 'Prompt2']) {
      expect(variableNameProblem(ok, [])).toBeNull()
    }
    for (const bad of ['', '9lives', '-x', 'has space', 'a.b', 'a@b', 'é']) {
      expect(variableNameProblem(bad, [])).toBe(RULE)
    }
  })

  it('a taken name is refused', () => {
    expect(variableNameProblem('seed', ['seed'])).toBe(
      'There is already a variable called seed.',
    )
  })
})

describe('inferValue', () => {
  it('infers numbers, booleans, null and JSON', () => {
    expect(inferValue('30')).toBe(30)
    expect(inferValue('-1.5')).toBe(-1.5)
    expect(inferValue('true')).toBe(true)
    expect(inferValue('false')).toBe(false)
    expect(inferValue('null')).toBeNull()
    expect(inferValue('{"a": 1}')).toEqual({ a: 1 })
    expect(inferValue('[1, 2]')).toEqual([1, 2])
  })

  it('keeps other text as text', () => {
    expect(inferValue('a cat')).toBe('a cat')
    expect(inferValue('')).toBe('')
    expect(inferValue('NaN')).toBe('NaN')
    expect(inferValue('{broken')).toBe('{broken')
  })

  it('a reference stays text', () => {
    expect(inferValue('variable:size')).toBe('variable:size')
    expect(inferValue('constant:30')).toBe('constant:30')
  })
})

describe('types', () => {
  it('names the type of a value', () => {
    expect(typeOf(null)).toBe('none')
    expect(typeOf({})).toBe('json')
    expect(typeOf([1])).toBe('json')
    expect(typeOf(2)).toBe('number')
    expect(typeOf(false)).toBe('boolean')
    expect(typeOf('x')).toBe('text')
  })

  it('asType turns typed text into the type', () => {
    expect(asType('30', 'number')).toBe(30)
    expect(asType('variable:n', 'number')).toBe('variable:n')
    expect(asType('x', 'number')).toBe('x')
    expect(asType('true', 'boolean')).toBe(true)
    expect(asType('False', 'boolean')).toBe(false)
    expect(asType('{"a":1}', 'json')).toEqual({ a: 1 })
    expect(asType('30', 'text')).toBe('30')
    expect(asType('anything', 'none')).toBeNull()
  })

  it('a boolean row keeps text that is not true or false as text', () => {
    expect(asType('yes', 'boolean')).toBe('yes')
    expect(asType('1', 'boolean')).toBe('1')
    expect(asType('ture', 'boolean')).toBe('ture')
  })
})
