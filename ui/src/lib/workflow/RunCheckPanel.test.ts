import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import { ApiError } from '../apiErrors'
import RunCheckPanel from './RunCheckPanel.svelte'

vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))
afterEach(cleanup)

it('a check that failed leads with plain words, the engine text folded behind Details', () => {
  const raw = 'Traceback (most recent call last): ValidationWorkerDied'
  const { container } = render(RunCheckPanel, {
    props: {
      verdict: { tone: 'bad', text: 'could not check this run' },
      checking: false,
      planLines: [],
      checkError: new ApiError(raw, 500),
      findings: [],
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
      planLines: [],
      checkError: null,
      findings: [],
    },
  })
  expect(container.querySelector('.findings')).toBeNull()
})
