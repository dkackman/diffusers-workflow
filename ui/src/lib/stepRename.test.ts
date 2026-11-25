import { expect, it } from 'vitest'
import { renameStep, stepNameProblem } from './stepRename'

// Every shape that names step `a`, and a step `ab` whose name only starts
// with it, referenced the same ways
const workflow = () => ({
  id: 'w',
  variables: {
    clip: 'previous_result:a.frames',
    other: 'previous_result:ab',
    plain: 'a',
  },
  steps: [
    { name: 'a', task: { command: 'x', arguments: {} } },
    { name: 'ab', task: { command: 'x', arguments: {} } },
    {
      name: 'use',
      for_each: [{ name: 'cold_open' }],
      task: {
        command: 'y',
        arguments: {
          image: 'previous_result:a',
          frames: 'previous_result:a.frames',
          member: 'previous_result:a@cold_open',
          deep: [{ refs: [{ from_previous_result: 'a' }] }],
          prop: { from_previous_result: 'a.mask' },
          prefixed: { from_previous_result: 'previous_result:a' },
          all: 'gather:a',
          gathered: 'gather:a.videos',
          near: 'previous_result:ab',
          nearer: 'previous_result:ab.frames',
          nearest: { from_previous_result: 'ab' },
          gatherNear: 'gather:ab',
          prose: 'shot: previous_result:a',
          literal: 'a',
        },
      },
    },
  ],
})

it('rewrites every shape of reference and nothing that only starts with the name', () => {
  const before = workflow()
  const { workflow: after } = renameStep(before, 'a', 'b')
  expect(after.steps![0].name).toBe('b')
  expect(after.steps![1].name).toBe('ab')
  expect(after.variables).toEqual({
    clip: 'previous_result:b.frames',
    other: 'previous_result:ab',
    plain: 'a',
  })
  expect(after.steps![2].task!.arguments).toEqual({
    image: 'previous_result:b',
    frames: 'previous_result:b.frames',
    member: 'previous_result:b@cold_open',
    deep: [{ refs: [{ from_previous_result: 'b' }] }],
    prop: { from_previous_result: 'b.mask' },
    prefixed: { from_previous_result: 'previous_result:b' },
    all: 'gather:b',
    gathered: 'gather:b.videos',
    near: 'previous_result:ab',
    nearer: 'previous_result:ab.frames',
    nearest: { from_previous_result: 'ab' },
    gatherNear: 'gather:ab',
    prose: 'shot: previous_result:a',
    literal: 'a',
  })
  // A new document: the one passed in is left as it was
  expect(before).toEqual(workflow())
})

it('counts what it changed', () => {
  expect(renameStep(workflow(), 'a', 'b').count).toBe(9)
  expect(renameStep(workflow(), 'use', 'shots').count).toBe(0)
  expect(renameStep(workflow(), 'ab', 'abc').count).toBe(5)
})

it('refuses an empty name, a dot, an @ and a space', () => {
  const copy = "A step name can't be empty or hold a space, . or @."
  for (const name of ['', 'a.b', 'a@b', 'a b', ' a'])
    expect(stepNameProblem(name, [])).toBe(copy)
  expect(stepNameProblem('a-b_2', [])).toBeNull()
})

it('refuses a taken name', () => {
  expect(stepNameProblem('gen', ['gen', 'up'])).toBe(
    'Another step is already called gen.',
  )
  expect(stepNameProblem('gen2', ['gen', 'up'])).toBeNull()
})
