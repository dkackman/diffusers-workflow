import { afterEach, expect, it } from 'vitest'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import ConfirmDialog from './ui/ConfirmDialog.svelte'
import {
  confirmDialog,
  confirmState,
  nameDialog,
  resolveChoice,
} from './confirm.svelte'

afterEach(() => {
  cleanup()
  while (confirmState.open) resolveChoice(null)
})

const noSpaces = (name: string) =>
  !name ? 'Type a name' : /\s/.test(name) ? 'No spaces in a name' : null

const ask = (initial = 'a.png') =>
  nameDialog('Later workflows can use the file under this name.', {
    title: 'Save as asset',
    label: 'Name in the asset library',
    initial,
    confirmLabel: 'Save',
    validate: noSpaces,
  })

const field = () =>
  screen.getByRole('textbox', {
    name: 'Name in the asset library',
  }) as HTMLInputElement
const save = () =>
  screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement

it('nameDialog answers the typed name', async () => {
  render(ConfirmDialog)
  const answer = ask()
  await screen.findByRole('alertdialog', { name: 'Save as asset' })
  expect(field().value).toBe('a.png')
  await fireEvent.input(field(), { target: { value: 'face.png' } })
  await fireEvent.click(save())
  expect(await answer).toBe('face.png')
})

it('an invalid name keeps Save disabled and says why', async () => {
  render(ConfirmDialog)
  const answer = ask()
  await screen.findByRole('alertdialog')
  expect(save().disabled).toBe(false)
  await fireEvent.input(field(), { target: { value: 'my face' } })
  expect(save().disabled).toBe(true)
  const why = screen.getByText('No spaces in a name')
  expect(field().getAttribute('aria-invalid')).toBe('true')
  expect(field().getAttribute('aria-describedby')).toBe(why.id)
  // Enter submits only a valid name
  await fireEvent.keyDown(field(), { key: 'Enter' })
  expect(confirmState.open).toBe(true)
  await fireEvent.input(field(), { target: { value: '' } })
  expect(screen.getByText('Type a name')).toBeTruthy()
  await fireEvent.input(field(), { target: { value: 'face' } })
  expect(save().disabled).toBe(false)
  expect(screen.queryByText('No spaces in a name')).toBeNull()
  await fireEvent.keyDown(field(), { key: 'Enter' })
  expect(await answer).toBe('face')
})

it('Escape answers null', async () => {
  render(ConfirmDialog)
  const escaped = ask()
  await fireEvent.keyDown(await screen.findByRole('alertdialog'), {
    key: 'Escape',
  })
  expect(await escaped).toBeNull()

  const cancelled = ask()
  await fireEvent.click(await screen.findByRole('button', { name: 'Cancel' }))
  expect(await cancelled).toBeNull()
})

it('a nameDialog opened while a confirm is up waits its turn', async () => {
  render(ConfirmDialog)
  const first = confirmDialog('Delete it?', { confirmLabel: 'Delete' })
  const second = ask('b.png')
  const dialog = await screen.findByRole('alertdialog')
  expect(dialog.textContent).toContain('Delete it?')
  expect(screen.queryByRole('textbox')).toBeNull()
  await fireEvent.click(screen.getByRole('button', { name: 'Delete' }))
  expect(await first).toBe(true)
  await waitFor(() => expect(field().value).toBe('b.png'))
  await fireEvent.click(save())
  expect(await second).toBe('b.png')
  // The field goes with its dialog: a confirm after it has none
  const third = confirmDialog('Again?')
  await waitFor(() =>
    expect(screen.getByRole('alertdialog').textContent).toContain('Again?'),
  )
  expect(screen.queryByRole('textbox')).toBeNull()
  resolveChoice('cancel')
  expect(await third).toBe(false)
})

it("a filename's dot does not end the title", async () => {
  const done = confirmDialog('Delete take.v2.png? This removes it on disk.')
  expect(confirmState.title).toBe('Delete take.v2.png?')
  resolveChoice(null)
  await done
})

it.each([
  ['Delete it? notes.txt stays.', 'Delete it?'],
  ['Replace a vs. b? Neither is kept.', 'Replace a vs. b?'],
  ['Remove drafts, e.g. old ones? They go.', 'Remove drafts, e.g. old ones?'],
])('titles "%s" with its first sentence', async (message, title) => {
  const done = confirmDialog(message)
  expect(confirmState.title).toBe(title)
  resolveChoice(null)
  await done
})
