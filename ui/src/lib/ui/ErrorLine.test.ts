import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import { ApiError } from '../apiErrors'
import ErrorLine from './ErrorLine.svelte'

vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))
afterEach(cleanup)

it('leads with the described title and keeps the engine text behind Details', () => {
  const error = new ApiError('A job is running.\nTraceback: boom', 409)
  const { container } = render(ErrorLine, {
    props: { error, doing: 'clear that' },
  })
  const alert = screen.getByRole('alert')
  expect(alert.textContent).toContain("Couldn't clear that: A job is running.")
  expect(screen.getByText('Engine detail')).toBeTruthy()
  const details = container.querySelector('details') as HTMLDetailsElement
  expect(details.open).toBe(false)
  expect(container.querySelector('pre')?.textContent).toContain('Traceback')
})

it('without doing, shows the title alone and no Details when it is all there is', () => {
  const { container } = render(ErrorLine, {
    props: { error: new ApiError('Nope.', 409) },
  })
  expect(screen.getByRole('alert').textContent).toContain('Nope.')
  expect(container.querySelector('details')).toBeNull()
})
