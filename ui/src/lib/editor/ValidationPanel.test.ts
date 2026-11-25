import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import ValidationPanel from './ValidationPanel.svelte'
import type { Plan, ValidationResult } from '../types'

afterEach(cleanup)

const plan: Plan = {
  fingerprint: 'sha256:abc',
  steps: 1,
  list_entries: {},
  cached_steps: null,
  elided_steps: [],
  downloads_required: [
    {
      repo: 'org/missing',
      gb: null,
      gated: null,
      access_blocked: null,
      not_found: true,
    },
  ],
  estimate: {
    minutes: null,
    basis: 'unknown',
    device: 'mps',
    measured_on: null,
    partial: false,
    unpriced: [],
    runs: null,
    cached_minutes: null,
  },
  workspace: 'default',
}

/** The props a test that never dismisses or jumps passes */
const quiet = { doc: {}, ondismiss: () => {}, onjump: () => {} }

const result = (over: Partial<ValidationResult>): ValidationResult =>
  ({ valid: true, warnings: [], ...over }) as ValidationResult

describe('ValidationPanel', () => {
  it('an invalid answer that keeps its plan shows the error and the plan', () => {
    render(ValidationPanel, {
      validation: result({
        valid: false,
        errors: [{ path: 'model_name', message: 'org/missing was not found' }],
        plan,
      }),
      ...quiet,
    })
    expect(
      screen.getByText('model_name: org/missing was not found'),
    ).toBeTruthy()
    const box = screen.getByText('What a job will do').closest('details')!
    expect(box.textContent).toContain('needs download: org/missing (not found)')
    // the plan sits under the error list
    const error = screen.getByText('model_name: org/missing was not found')
    expect(
      error.compareDocumentPosition(box) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
  })

  it('an invalid answer lists its warnings beside its errors, literally', () => {
    render(ValidationPanel, {
      validation: result({
        valid: false,
        errors: [{ path: 'a', message: 'bad' }],
        warnings: ['WARN-PLAIN <b>x</b>'],
      }),
      ...quiet,
    })
    expect(screen.getByText('a: bad')).toBeTruthy()
    expect(screen.getByText(/WARN-PLAIN <b>x<\/b>/)).toBeTruthy()
  })

  it('a valid answer still shows its plan', () => {
    render(ValidationPanel, {
      validation: result({ plan: { ...plan, downloads_required: [] } }),
      ...quiet,
    })
    expect(
      screen.getByText('What a job will do').closest('details')!.textContent,
    ).toContain('1 step')
  })

  it('a valid answer with no warnings says Ready to run', () => {
    render(ValidationPanel, { validation: result({}), ...quiet })
    expect(screen.getByText('Ready to run.')).toBeTruthy()
  })

  it('a valid answer with an author warning is not green', () => {
    const { container } = render(ValidationPanel, {
      validation: result({ warnings: ['WARN-PLAIN something to read'] }),
      ...quiet,
    })
    const verdict = container.querySelector('.verdict')!
    expect(verdict.classList.contains('warn')).toBe(true)
    expect(verdict.classList.contains('good')).toBe(false)
    expect(screen.getByText(/WARN-PLAIN something to read/)).toBeTruthy()
  })

  it('a plan line that warns opens the plan, a plain plan stays shut', () => {
    render(ValidationPanel, { validation: result({ plan }), ...quiet })
    expect(
      screen.getByText('What a job will do').closest('details')!.open,
    ).toBe(true)
    cleanup()
    render(ValidationPanel, {
      validation: result({
        plan: {
          ...plan,
          downloads_required: [],
          estimate: { ...plan.estimate, minutes: 2, basis: 'observed' },
        },
      }),
      ...quiet,
    })
    expect(
      screen.getByText('What a job will do').closest('details')!.open,
    ).toBe(false)
  })

  it('an answer without a plan shows none', () => {
    render(ValidationPanel, {
      validation: result({
        valid: false,
        errors: [{ path: 'arguments.n', message: 'not an integer' }],
      }),
      ...quiet,
    })
    expect(screen.queryByText('What a job will do')).toBeNull()
  })

  it('an error with a place offers Go to', async () => {
    const onjump = vi.fn()
    render(ValidationPanel, {
      validation: result({
        valid: false,
        errors: [
          { path: 'steps[0].pipeline', message: 'no pipeline' },
          { path: 'variables.prompt', message: 'not a string' },
          { path: 'steps[5]', message: 'no such step' },
          { path: null, message: 'not a workflow' },
        ],
      }),
      doc: { variables: { prompt: 1 }, steps: [{ name: 'generate' }] },
      ondismiss: () => {},
      onjump,
    })
    expect(screen.getByText('steps[0].pipeline: no pipeline')).toBeTruthy()
    await fireEvent.click(
      screen.getByRole('button', { name: 'Go to step generate' }),
    )
    expect(onjump).toHaveBeenLastCalledWith({
      kind: 'step',
      index: 0,
      name: 'generate',
    })
    await fireEvent.click(
      screen.getByRole('button', { name: 'Go to variable prompt' }),
    )
    expect(onjump).toHaveBeenLastCalledWith({
      kind: 'variable',
      name: 'prompt',
    })
    // a step that is not there, and the root, have nowhere to go
    expect(screen.getAllByRole('button', { name: /^Go to/ })).toHaveLength(2)
    expect(screen.getByText('root: not a workflow')).toBeTruthy()
  })
})
