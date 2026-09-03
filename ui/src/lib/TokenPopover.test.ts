import { render } from '@testing-library/svelte'
import { expect, it } from 'vitest'
import TokenPopover from './TokenPopover.svelte'

it('moves focus into the dialog when it opens', () => {
  const trigger = document.createElement('button')
  document.body.appendChild(trigger)
  trigger.focus()
  const { container } = render(TokenPopover, { open: true })
  const dialog = container.querySelector('[role="dialog"]')
  expect(dialog).not.toBeNull()
  expect(dialog!.getAttribute('aria-modal')).toBe('true')
  expect(dialog!.contains(document.activeElement)).toBe(true)
  trigger.remove()
})
