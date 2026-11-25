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

it('a double click on Undo restores the step once', async () => {
  render(StepList, {
    steps: [taskStep('a'), taskStep('b')],
    modes: new StepModes(() => 'test-step-modes'),
    folder: '',
    flow: [],
    problemsByStep: new Map(),
    stepReferences: [],
    hovered: null,
  })
  await fireEvent.click(
    screen.getAllByRole('button', { name: 'remove step' })[1],
  )
  const press = notify.undo.mock.calls[0][1]
  press()
  press()
  await tick()
  await new Promise((r) => setTimeout(r, 0))
  expect(names()).toEqual(['a', 'b'])
})

/** Type `to` into the name field of the step called `from`, as a change */
async function renameIn(from: string, to: string) {
  const field = document.querySelector(
    `#step-${from} input.name`,
  ) as HTMLInputElement
  field.value = to
  await fireEvent.change(field)
}

it("a rename keeps the step's mode and its references", async () => {
  const modes = new StepModes(() => 'test-step-modes')
  const steps = [
    taskStep('a'),
    {
      name: 'use',
      task: {
        command: 'y',
        arguments: { image: 'previous_result:a', near: 'previous_result:ab' },
      },
    },
  ]
  const workflow = { variables: { clip: 'gather:a' }, steps }
  render(StepList, {
    steps,
    workflow,
    modes,
    folder: '',
    flow: [],
    problemsByStep: new Map(),
    stepReferences: [],
    hovered: null,
  })
  modes.set({ name: 'a' }, 'full')
  await tick()
  await renameIn('a', 'base')
  expect(steps[0].name).toBe('base')
  expect(steps[1].task.arguments).toEqual({
    image: 'previous_result:base',
    near: 'previous_result:ab',
  })
  expect(workflow.variables).toEqual({ clip: 'gather:base' })
  expect(modes.of({ name: 'base' })).toBe('full')
  expect(notify.success).toHaveBeenCalledWith(
    'Renamed step a to base, and 2 references to it',
  )
})

it('a rename rewrites a reference inside a preset, and counts it', async () => {
  const steps = [taskStep('a'), taskStep('b')]
  const workflow = {
    presets: { quick: { image: 'previous_result:a' } },
    steps,
  }
  render(StepList, {
    steps,
    workflow,
    modes: new StepModes(() => 'test-step-modes'),
    folder: '',
    flow: [],
    problemsByStep: new Map(),
    stepReferences: [],
    hovered: null,
  })
  await renameIn('a', 'base')
  expect(workflow.presets).toEqual({ quick: { image: 'previous_result:base' } })
  expect(notify.success).toHaveBeenCalledWith(
    'Renamed step a to base, and 1 reference to it',
  )
})

it('a refused rename keeps the old name and says why', async () => {
  const steps = [taskStep('a'), taskStep('b')]
  render(StepList, {
    steps,
    workflow: { steps },
    modes: new StepModes(() => 'test-step-modes'),
    folder: '',
    flow: [],
    problemsByStep: new Map(),
    stepReferences: [],
    hovered: null,
  })
  await renameIn('a', 'b')
  expect(steps[0].name).toBe('a')
  expect(screen.getByRole('alert').textContent).toBe(
    'Another step is already called b.',
  )
  await renameIn('a', 'a.b')
  expect(screen.getByRole('alert').textContent).toBe(
    "A step name can't be empty or hold a space, . or @.",
  )
  expect(notify.success).not.toHaveBeenCalled()
})

it('undo puts a step back beside its old neighbour', async () => {
  render(StepList, {
    steps: [taskStep('a'), taskStep('b'), taskStep('c')],
    modes: new StepModes(() => 'test-step-modes'),
    folder: '',
    flow: [],
    problemsByStep: new Map(),
    stepReferences: [],
    hovered: null,
  })
  await fireEvent.click(
    screen.getAllByRole('button', { name: 'remove step' })[1],
  )
  // c, the step that followed b, moves to the top meanwhile
  await fireEvent.click(screen.getAllByRole('button', { name: 'move up' })[1])
  expect(names()).toEqual(['c', 'a'])
  notify.undo.mock.calls[0][1]()
  await tick()
  expect(names()).toEqual(['b', 'c', 'a'])
})

it('a step that comes back renamed says so', async () => {
  render(StepList, {
    steps: [taskStep('process'), taskStep('z')],
    modes: new StepModes(() => 'test-step-modes'),
    folder: '',
    flow: [],
    problemsByStep: new Map(),
    stepReferences: [],
    hovered: null,
  })
  await fireEvent.click(
    screen.getAllByRole('button', { name: 'remove step' })[0],
  )
  await fireEvent.click(screen.getByRole('button', { name: /task step/ }))
  notify.undo.mock.calls[0][1]()
  await tick()
  expect(notify.success).toHaveBeenCalledWith(
    'Step process came back as process-2',
  )
})
