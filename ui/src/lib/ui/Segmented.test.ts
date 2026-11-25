import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, describe, expect, it, vi } from 'vitest'
import Segmented from './Segmented.svelte'

const options = [
  { value: 'compact', label: 'compact', title: 'one line per area' },
  { value: 'full', label: 'full' },
]

describe('Segmented', () => {
  afterEach(cleanup)

  it('marks the chosen option pressed and reports a pick', async () => {
    const onselect = vi.fn()
    render(Segmented, {
      props: { options, value: 'compact', onselect, label: 'detail' },
    })
    expect(
      screen
        .getByRole('button', { name: 'compact' })
        .getAttribute('aria-pressed'),
    ).toBe('true')
    const full = screen.getByRole('button', { name: 'full' })
    expect(full.getAttribute('aria-pressed')).toBe('false')
    await fireEvent.click(full)
    expect(onselect).toHaveBeenCalledWith('full')
  })

  it('is a named group', () => {
    render(Segmented, {
      props: { options, value: 'full', onselect: () => {}, label: 'detail' },
    })
    expect(screen.getByRole('group', { name: 'detail' })).toBeTruthy()
  })
})
