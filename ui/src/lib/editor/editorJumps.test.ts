import { expect, it, vi } from 'vitest'
import type { DocumentEditor } from '../editorShell.svelte'
import type { StepModes } from '../stepModes.svelte'
import type { WorkflowDefinition, WorkflowStep } from '../types'
import { EditorJumps } from './editorJumps.svelte'

const jumps = (steps: WorkflowStep[]) => {
  const set = vi.fn()
  const setView = vi.fn()
  const ed = {
    doc: { steps },
    setView,
  } as unknown as DocumentEditor<WorkflowDefinition>
  return {
    go: new EditorJumps(ed, { set } as unknown as StepModes).goTo,
    set,
    setView,
  }
}

it("a finding's Go to opens its step", () => {
  const step = { name: 'a' } as WorkflowStep
  const { go, set, setView } = jumps([step])
  go({ kind: 'step', index: 0, name: 'a' })
  expect(set).toHaveBeenCalledWith(step, 'full')
  expect(setView).toHaveBeenCalledWith('form')
})

it('a finding whose step is gone does nothing', () => {
  // the step was removed and the check has not answered yet: the finding
  // still on screen names an index the document no longer has
  const { go, set, setView } = jumps([{ name: 'a' } as WorkflowStep])
  expect(() => go({ kind: 'step', index: 3, name: 'd' })).not.toThrow()
  expect(set).not.toHaveBeenCalled()
  expect(setView).not.toHaveBeenCalled()
})

it('a finding whose index now holds another step goes to the one it names', () => {
  const [a, c] = [{ name: 'a' }, { name: 'c' }] as WorkflowStep[]
  const { go, set } = jumps([a, c])
  go({ kind: 'step', index: 2, name: 'c' })
  expect(set).toHaveBeenCalledWith(c, 'full')
})
