import { afterEach, expect, it } from 'vitest'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import KeyboardHelp from '../KeyboardHelp.svelte'
import { overlayOpen } from './layers.svelte'

afterEach(() => cleanup())

it('is a modal dialog with focus inside, holding a layer', async () => {
  render(KeyboardHelp, { open: true })
  const dialog = await screen.findByRole('dialog', {
    name: 'keyboard shortcuts',
  })
  expect(dialog.getAttribute('aria-modal')).toBe('true')
  await waitFor(() =>
    expect(dialog.contains(document.activeElement)).toBe(true),
  )
  expect(overlayOpen()).toBe(true)
})

it('closes on Escape and releases its layer', async () => {
  render(KeyboardHelp, { open: true })
  const dialog = await screen.findByRole('dialog', {
    name: 'keyboard shortcuts',
  })
  await fireEvent.keyDown(dialog, { key: 'Escape' })
  await waitFor(() =>
    expect(
      screen.queryByRole('dialog', { name: 'keyboard shortcuts' }),
    ).toBeNull(),
  )
  await waitFor(() => expect(overlayOpen()).toBe(false))
})
