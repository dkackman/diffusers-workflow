import { expect, it } from 'vitest'
import { placeFindings } from './findings'
import type { ValidationResult } from './types'

const result = (over: Partial<ValidationResult>) =>
  ({ valid: true, warnings: [], errors: [], ...over }) as ValidationResult

it('a warning naming a shown variable goes under it', () => {
  const placed = placeFindings(
    result({ warnings: ['arguments.seed: fixed seed'] }),
    ['seed'],
  )
  expect(placed.byVariable.seed).toEqual([{ tone: 'warn', text: 'fixed seed' }])
  expect(placed.author).toEqual([])
  expect(placed.general).toEqual([])
})

it("a warning naming nothing is the author's", () => {
  const placed = placeFindings(
    result({
      warnings: ['a step has no output', 'arguments.gone: not shown'],
    }),
    ['seed'],
  )
  expect(placed.author.map((f) => f.text)).toEqual([
    'a step has no output',
    'arguments.gone: not shown',
  ])
  expect(placed.general).toEqual([])
})

it("errors are never the author's", () => {
  const placed = placeFindings(
    result({
      valid: false,
      errors: [{ path: 'steps[0]', message: 'no pipeline' }],
    }),
    [],
  )
  expect(placed.general).toEqual([
    { tone: 'bad', text: 'steps[0]: no pipeline' },
  ])
  expect(placed.author).toEqual([])
  const bare = placeFindings(result({ valid: false, error: 'refused' }), [])
  expect(bare.general).toEqual([{ tone: 'bad', text: 'refused' }])
  expect(bare.author).toEqual([])
})
