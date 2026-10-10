import { expect, it } from 'vitest'
import { wsColor } from './wsColor'

it('is stable for a name and differs between names', () => {
  expect(wsColor('acorn-wars')).toBe(wsColor('acorn-wars'))
  expect(wsColor('acorn-wars')).not.toBe(wsColor('acorn-war'))
  expect(wsColor('default')).toMatch(/^oklch\(0\.7 0\.14 \d+\)$/)
})
