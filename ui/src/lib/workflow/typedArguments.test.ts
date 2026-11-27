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

it('a null default takes a typed list or object as JSON, and text as text', () => {
  // a required variable the workflow declares null: a script's `lines`, a
  // cut's `shots`. Sent as a string, the task refuses it
  const declared = { lines: null, shots: null, note: null, count: null }
  expect(
    typedArguments(
      {
        lines: ' ["hello there"] ',
        shots: '{"a":1}',
        note: 'hello',
        count: '5',
      },
      declared,
    ),
  ).toEqual({
    lines: ['hello there'],
    shots: { a: 1 },
    note: 'hello',
    count: '5',
  })
  // a bracket that is not JSON is still what was typed
  expect(typedArguments({ lines: '[draft' }, declared)).toEqual({
    lines: '[draft',
  })
})

it('a list default takes a bare scalar as text, as the engine splits it', () => {
  const declared = { sizes: [1] }
  expect(typedArguments({ sizes: '5' }, declared)).toEqual({ sizes: '5' })
  expect(typedArguments({ sizes: 'true' }, declared)).toEqual({ sizes: 'true' })
  expect(typedArguments({ sizes: '[5]' }, declared)).toEqual({ sizes: [5] })
  expect(typedArguments({ sizes: 'a, b' }, declared)).toEqual({ sizes: 'a, b' })
})
