import { expect, it } from 'vitest'
import { typedArguments } from './typedArguments'

it('a number default sends a number', () => {
  expect(
    typedArguments(
      { steps: '30', on: 'true', size: '{"w":1}', name: 'x' },
      { steps: 20, on: false, size: {}, name: 'y' },
    ),
  ).toEqual({ steps: 30, on: true, size: { w: 1 }, name: 'x' })
})

it('a reference stays text', () => {
  expect(
    typedArguments(
      { steps: 'variable:n', image: 'asset:cast/a.png' },
      { steps: 20, image: 'asset:cast/b.png' },
    ),
  ).toEqual({ steps: 'variable:n', image: 'asset:cast/a.png' })
})

it('a blank is dropped', () => {
  expect(typedArguments({ a: '', b: '  ', c: '1' }, { c: 0 })).toEqual({ c: 1 })
})

it('a boolean field typed yes is sent as text, not false', () => {
  const declared = { keep: false, snap: true, look: false, typo: true }
  expect(
    typedArguments(
      { keep: 'yes', snap: 'True', look: 'FALSE', typo: 'ture' },
      declared,
    ),
  ).toEqual({ keep: 'yes', snap: true, look: false, typo: 'ture' })
  // A reference fed to a boolean stays the reference
  expect(typedArguments({ keep: 'asset:on.txt' }, declared)).toEqual({
    keep: 'asset:on.txt',
  })
})
