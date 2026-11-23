import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/svelte'
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
      ondismiss: () => {},
    })
    expect(
      screen.getByText('model_name: org/missing was not found'),
    ).toBeTruthy()
    const box = screen.getByRole('list', { name: 'what a job will do' })
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
      ondismiss: () => {},
    })
    expect(screen.getByText('a: bad')).toBeTruthy()
    expect(screen.getByText(/WARN-PLAIN <b>x<\/b>/)).toBeTruthy()
  })

  it('a valid answer still shows its plan', () => {
    render(ValidationPanel, {
      validation: result({ plan: { ...plan, downloads_required: [] } }),
      ondismiss: () => {},
    })
    expect(
      screen.getByRole('list', { name: 'what a job will do' }).textContent,
    ).toContain('1 step')
  })

  it('an answer without a plan shows none', () => {
    render(ValidationPanel, {
      validation: result({
        valid: false,
        errors: [{ path: 'arguments.n', message: 'not an integer' }],
      }),
      ondismiss: () => {},
    })
    expect(
      screen.queryByRole('list', { name: 'what a job will do' }),
    ).toBeNull()
  })
})
