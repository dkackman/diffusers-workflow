import { expect, it } from 'vitest'
import { promotionTitle, promotionsOf } from './promotions'

const presets = {
  resolution: {
    '480p': { width: 832, height: 480 },
    '768p': { width: 1344, height: 768, steps: 40 },
  },
  undeclared: { a: { width: 1 } },
  broken: null as unknown as Record<string, unknown>,
}
const declared = { resolution: '480p', width: 832, steps: 30, broken: 'x' }

it('offers every other bundle of a selector the run declared', () => {
  expect(promotionsOf(presets, declared, {}, declared)).toEqual([
    { selector: 'resolution', value: '768p', kept: [] },
  ])
})

it('leaves out the bundle the run used, by the value it ran with', () => {
  const args = { resolution: '768p' }
  expect(
    promotionsOf(presets, declared, args, { ...declared, ...args }),
  ).toEqual([{ selector: 'resolution', value: '480p', kept: [] }])
})

it('names the bundle keys the run set itself, which stay as set', () => {
  const args = { steps: 12, resolution: '480p' }
  const [promotion] = promotionsOf(presets, declared, args, {
    ...declared,
    ...args,
  })
  expect(promotion).toEqual({
    selector: 'resolution',
    value: '768p',
    kept: ['steps=12'],
  })
  expect(promotionTitle(promotion)).toContain('; keeps steps=12 set explicitly')
})

it('offers nothing without presets or a definition on file', () => {
  expect(promotionsOf(null, declared, {}, declared)).toEqual([])
  expect(promotionsOf(presets, null, {}, {})).toEqual([])
})
