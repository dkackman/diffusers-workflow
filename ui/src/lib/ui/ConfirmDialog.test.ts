import { afterEach, expect, it } from 'vitest'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import ConfirmDialog from './ConfirmDialog.svelte'
import {
  chooseDialog,
  confirmDialog,
  confirmState,
  resolveChoice,
} from '../confirm.svelte'
import { overlayOpen } from './layers.svelte'

afterEach(() => {
  cleanup()
  // Dialogs are queued in module state: drain any a test left unanswered
  while (confirmState.open) resolveChoice(null)
})

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

it('queues a second dialog until the first is answered', async () => {
  render(ConfirmDialog)
  const first = confirmDialog('First?', { confirmLabel: 'Yes one' })
  const second = confirmDialog('Second?')
  const dialog = await screen.findByRole('alertdialog')
  expect(dialog.textContent).toContain('First?')
  expect(dialog.textContent).not.toContain('Second?')
  await fireEvent.click(screen.getByRole('button', { name: 'Yes one' }))
  expect(await first).toBe(true)
  await waitFor(() =>
    expect(screen.getByRole('alertdialog').textContent).toContain('Second?'),
  )
  await fireEvent.keyDown(screen.getByRole('alertdialog'), { key: 'Escape' })
  expect(await second).toBe(false)
})

it('resolves a choice by value', async () => {
  render(ConfirmDialog)
  const answer = chooseDialog('x', [
    { label: 'Replace', value: 'replace', danger: true },
    { label: 'Cancel', value: 'cancel' },
  ])
  await fireEvent.click(await screen.findByRole('button', { name: 'Replace' }))
  expect(await answer).toBe('replace')
})

it('resolves null on Escape for a choice dialog', async () => {
  render(ConfirmDialog)
  const answer = chooseDialog('x', [
    { label: 'Keep', value: 'keep' },
    { label: 'Cancel', value: 'cancel' },
  ])
  await fireEvent.keyDown(await screen.findByRole('alertdialog'), {
    key: 'Escape',
  })
  expect(await answer).toBeNull()
})

it('marks a danger choice', async () => {
  render(ConfirmDialog)
  void chooseDialog('x', [
    { label: 'Replace', value: 'replace', danger: true },
    { label: 'Cancel', value: 'cancel' },
  ])
  const button = await screen.findByRole('button', { name: 'Replace' })
  expect(button.classList.contains('danger')).toBe(true)
})

it('is named by its title', async () => {
  render(ConfirmDialog)
  void confirmDialog('Really?', { title: 'Delete e2e-one.png?' })
  expect(
    await screen.findByRole('alertdialog', { name: 'Delete e2e-one.png?' }),
  ).toBeTruthy()
})

it('is named by the first sentence of the message by default', async () => {
  render(ConfirmDialog)
  void confirmDialog('Discard changes? They cannot be recovered.')
  expect(
    await screen.findByRole('alertdialog', { name: 'Discard changes?' }),
  ).toBeTruthy()
})
