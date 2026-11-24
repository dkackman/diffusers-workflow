import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import { ApiError } from '../apiErrors'
import RunCheckPanel from './RunCheckPanel.svelte'
import ValidationPanel from '../editor/ValidationPanel.svelte'
import { placeFindings } from '../findings'
import { verdictOf } from '../runCheck.svelte'
import type { ValidationResult } from '../types'

vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))
afterEach(cleanup)

it('a check that failed leads with plain words, the engine text folded behind Details', () => {
  const raw = 'Traceback (most recent call last): ValidationWorkerDied'
  const { container } = render(RunCheckPanel, {
    props: {
      verdict: {
        tone: 'bad',
        text: "Couldn't check this workflow.",
        issues: 0,
      },
      checking: false,
      plan: null,
      checkError: new ApiError(raw, 500),
      findings: [],
      author: [],
    },
  })
  const title = screen.getByRole('alert').querySelector('.title')
  expect(title?.textContent).toBe(
    "Couldn't check these values: The server hit an error",
  )
  expect(title?.textContent).not.toContain('Traceback')
  const details = container.querySelector('details') as HTMLDetailsElement
  expect(details.open).toBe(false)
  expect(details.querySelector('pre')?.textContent).toBe(raw)
})

it('no failed check, no findings list', () => {
  const { container } = render(RunCheckPanel, {
    props: {
      verdict: null,
      checking: false,
      plan: null,
      checkError: null,
      findings: [],
      author: [],
    },
  })
  expect(container.querySelector('.findings')).toBeNull()
})

it('both say the same verdict for the same answer', () => {
  const answer = {
    valid: false,
    errors: [{ path: 'steps[0]', message: 'no pipeline' }],
    warnings: ['a step has no output'],
  } as ValidationResult
  const placed = placeFindings(answer, [])
  const verdict = verdictOf(answer, null, placed)
  const run = render(RunCheckPanel, {
    props: {
      verdict,
      checking: false,
      plan: null,
      checkError: null,
      findings: placed.general,
      author: placed.author,
    },
  })
  expect(run.container.querySelector('.verdict')?.textContent).toContain(
    'A job would be refused: 1 issue to fix.',
  )
  expect(run.container.querySelector('#issue-0')?.textContent).toBe(
    'steps[0]: no pipeline',
  )
  expect(screen.getByText("1 note for the workflow's author")).toBeTruthy()
  cleanup()
  const editor = render(ValidationPanel, {
    props: { validation: answer, ondismiss: () => {} },
  })
  expect(editor.container.querySelector('.verdict')?.textContent).toContain(
    'A job would be refused: 1 issue to fix.',
  )
})
