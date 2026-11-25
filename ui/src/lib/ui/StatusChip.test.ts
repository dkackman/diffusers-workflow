import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it } from 'vitest'
import StatusChip from './StatusChip.svelte'

afterEach(cleanup)

it('a cancelled job has one look everywhere', () => {
  const { container } = render(StatusChip, { props: { status: 'cancelled' } })
  const chip = container.querySelector('.chip.cancelled')
  expect(chip?.textContent).toBe('cancelled')
  expect(container.querySelector('.chip.idle')).toBeNull()
})

it('says its own word when given a label', () => {
  render(StatusChip, { props: { status: 'queued', label: 'waiting' } })
  expect(screen.getByText('waiting').classList.contains('queued')).toBe(true)
})
