import { Check } from '@lucide/svelte'
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

  it('an icon option', () => {
    render(Segmented, {
      props: {
        options: [{ value: 'grid', label: 'grid', icon: Check }],
        value: 'grid',
        onselect: () => {},
        label: 'view',
      },
    })
    const button = screen.getByRole('button', { name: 'grid' })
    const icon = button.querySelector('svg')
    expect(icon?.getAttribute('width')).toBe('14')
    // the icon goes before the label
    expect(button.firstElementChild).toBe(icon)
  })

  it('size sm', () => {
    const props = { options, value: 'full', onselect: () => {}, label: 'd' }
    render(Segmented, { props })
    expect(screen.getByRole('group').classList.contains('sm')).toBe(false)
    cleanup()
    render(Segmented, { props: { ...props, size: 'sm' as const } })
    expect(screen.getByRole('group').classList.contains('sm')).toBe(true)
  })

  it('is a named group', () => {
    render(Segmented, {
      props: { options, value: 'full', onselect: () => {}, label: 'detail' },
    })
    expect(screen.getByRole('group', { name: 'detail' })).toBeTruthy()
  })
})
