import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import Details from './Details.svelte'

const copyText = vi.hoisted(() => vi.fn(() => Promise.resolve(true)))
vi.mock('../clipboard', () => ({ copyText }))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

afterEach(() => {
  cleanup()
  copyText.mockClear()
})

it('shows the summary, hides the detail until opened, copies it', async () => {
  const { container } = render(Details, {
    props: { summary: 'Engine detail', detail: 'Traceback line 1\nline 2' },
  })
  expect(screen.getByText('Engine detail')).toBeTruthy()
  const el = container.querySelector('details') as HTMLDetailsElement
  expect(el.open).toBe(false)
  el.open = true
  expect(container.querySelector('pre')?.textContent).toBe(
    'Traceback line 1\nline 2',
  )
  await fireEvent.click(screen.getByRole('button'))
  expect(copyText).toHaveBeenCalledWith('Traceback line 1\nline 2')
})

it('renders nothing when there is no detail', () => {
  const { container } = render(Details, {
    props: { summary: 'Engine detail', detail: '' },
  })
  expect(container.querySelector('details')).toBeNull()
})
