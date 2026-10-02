import { afterEach, expect, it } from 'vitest'
import { cleanup, fireEvent, render, waitFor } from '@testing-library/svelte'
import TokenPopover from './TokenPopover.svelte'
import { overlayOpen } from './ui/layers.svelte'

// jsdom has no layout: Bits never positions the popover, so it stays
// visibility:hidden, and an accessible-name query cannot see a hidden
// element. These find it by its content attribute and read its label;
// the e2e suite (a real browser) covers that it shows and is named, and
// that a press outside closes it (Bits' dismiss layer does not see a
// synthetic jsdom press as its own).
const popover = () =>
  document.querySelector<HTMLElement>('[data-popover-content]')

afterEach(() => {
  cleanup()
  document.body.innerHTML = ''
})

function trigger() {
  const button = document.createElement('button')
  button.textContent = 'API token'
  document.body.appendChild(button)
  button.focus()
  return button
}

it('is a non-modal dialog with focus inside, holding a layer', async () => {
  render(TokenPopover, { props: { open: true, anchor: trigger() } })
  await waitFor(() => expect(popover()).not.toBeNull())
  const pop = popover()!
  expect(pop.getAttribute('role')).toBe('dialog')
  expect(pop.getAttribute('aria-label')).toBe('API token')
  expect(pop.getAttribute('aria-modal')).not.toBe('true')
  await waitFor(() => expect(pop.contains(document.activeElement)).toBe(true))
  expect(overlayOpen()).toBe(true)
})

it('closes on Escape', async () => {
  render(TokenPopover, { props: { open: true, anchor: trigger() } })
  await waitFor(() => expect(popover()).not.toBeNull())
  await fireEvent.keyDown(popover()!, { key: 'Escape' })
  await waitFor(() => expect(popover()).toBeNull())
})

it('leaves a press on its own trigger to the trigger', async () => {
  // The trigger toggles; were the popover to close itself on that press
  // too, the trigger's toggle would reopen it
  const button = trigger()
  render(TokenPopover, { props: { open: true, anchor: button } })
  await waitFor(() => expect(popover()).not.toBeNull())
  await fireEvent.pointerDown(button)
  await fireEvent.pointerUp(button)
  await new Promise((r) => setTimeout(r, 50))
  expect(popover()).not.toBeNull()
})
