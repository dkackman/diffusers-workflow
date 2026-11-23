import { afterEach, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { tick } from 'svelte'
import StepList from './StepList.svelte'
import { StepModes } from '../stepModes.svelte'

const notify = vi.hoisted(() => ({
  undo: vi.fn(),
  failure: vi.fn(),
  success: vi.fn(),
}))
vi.mock('../toast', () => ({ notify }))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

const taskStep = (name: string) => ({
  name,
  task: { command: 'upscale', arguments: { scale: name.length } },
})

/** The step names down the rail, in order */
const names = () =>
  Array.from(document.querySelectorAll('.steprow')).map((row) =>
    row.id.replace(/^step-/, ''),
  )

it('removing a step offers Undo, and the step comes back where it was after another was added', async () => {
  const modes = new StepModes(() => 'test-step-modes')
  render(StepList, {
    steps: [taskStep('a'), taskStep('b'), taskStep('c')],
    modes,
    folder: '',
    flow: [],
    problemsByStep: new Map(),
    stepReferences: [],
    hovered: null,
  })
  expect(names()).toEqual(['a', 'b', 'c'])
  const removes = screen.getAllByRole('button', { name: 'remove step' })
  await fireEvent.click(removes[1])
  expect(names()).toEqual(['a', 'c'])
  expect(notify.undo).toHaveBeenCalledTimes(1)
  expect(notify.undo.mock.calls[0][0]).toBe('Removed step b')

  // Another step lands after the removal, before Undo is pressed
  await fireEvent.click(screen.getByRole('button', { name: /task step/ }))
  expect(names()).toEqual(['a', 'c', 'process'])

  notify.undo.mock.calls[0][1]()
  await tick()
  expect(names()).toEqual(['a', 'b', 'c', 'process'])
  // Nothing is lost: what the step held comes back with it, shown full
  expect(modes.of({ name: 'b' })).toBe('full')
  expect(notify.failure).not.toHaveBeenCalled()
})

it('an undone step whose name was taken meanwhile comes back renamed', async () => {
  const modes = new StepModes(() => 'test-step-modes')
  render(StepList, {
    steps: [taskStep('process'), taskStep('z')],
    modes,
    folder: '',
    flow: [],
    problemsByStep: new Map(),
    stepReferences: [],
    hovered: null,
  })
  await fireEvent.click(
    screen.getAllByRole('button', { name: 'remove step' })[0],
  )
  // A new task step takes the name the removed one had
  await fireEvent.click(screen.getByRole('button', { name: /task step/ }))
  expect(names()).toEqual(['z', 'process'])
  notify.undo.mock.calls[0][1]()
  await tick()
  expect(names()).toEqual(['process-2', 'z', 'process'])
})
