import { afterEach, expect, it } from 'vitest'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import ConfirmDialog from './ConfirmDialog.svelte'
import { confirmDialog } from '../confirm.svelte'
import { overlayOpen } from './layers.svelte'

afterEach(() => cleanup())

it('asks, and answers true on the confirm button', async () => {
  render(ConfirmDialog)
  const answer = confirmDialog('Delete it?', { confirmLabel: 'Delete' })
  const dialog = await screen.findByRole('alertdialog')
  expect(dialog.textContent).toContain('Delete it?')
  expect(overlayOpen()).toBe(true)
  await fireEvent.click(screen.getByRole('button', { name: 'Delete' }))
  expect(await answer).toBe(true)
  await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
  await waitFor(() => expect(overlayOpen()).toBe(false))
})

it('answers false on Cancel and on Escape', async () => {
  render(ConfirmDialog)
  const cancelled = confirmDialog('Delete it?')
  await fireEvent.click(await screen.findByRole('button', { name: 'Cancel' }))
  expect(await cancelled).toBe(false)

  const escaped = confirmDialog('Delete it?')
  const dialog = await screen.findByRole('alertdialog')
  await fireEvent.keyDown(dialog, { key: 'Escape' })
  expect(await escaped).toBe(false)
})

it('moves focus into the dialog', async () => {
  render(ConfirmDialog)
  confirmDialog('Delete it?')
  const dialog = await screen.findByRole('alertdialog')
  await waitFor(() =>
    expect(dialog.contains(document.activeElement)).toBe(true),
  )
})
