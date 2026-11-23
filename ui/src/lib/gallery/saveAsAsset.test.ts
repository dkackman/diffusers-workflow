import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/svelte'
import ConfirmDialog from '../ui/ConfirmDialog.svelte'
import { confirmState, resolveChoice } from '../confirm.svelte'
import { action } from '../action.svelte'
import { ApiError } from '../apiErrors'
import type { GalleryFile } from '../types'
import { assetNameError, saveAsAsset } from './saveAsAsset'

const keepOutput = vi.hoisted(() => vi.fn())
const notifyMock = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  failure: vi.fn(),
  undo: vi.fn(),
}))
vi.mock('../api', async () => ({
  ApiError: (await import('../apiErrors')).ApiError,
  api: { keepOutput: (...a: unknown[]) => keepOutput(...a) },
}))
vi.mock('../toast', () => ({ notify: notifyMock }))

const file = {
  name: 'wf/run-1/final/face.png',
  kind: 'image',
} as GalleryFile

// keep_output's exists refusal carries no code, only this wording
// (dw/server/routes/assets.py)
const taken = () =>
  new ApiError(
    'asset:face.png already exists - pass overwrite=true to replace it',
    409,
  )
const kept = (name: string) => ({ reference: `asset:${name}` })

beforeEach(() => {
  keepOutput.mockReset()
  for (const fn of Object.values(notifyMock)) fn.mockReset()
  render(ConfirmDialog)
})
afterEach(() => {
  cleanup()
  while (confirmState.open) resolveChoice(null)
})

/** saveAsAsset as GalleryDetail runs it */
const save = (existing: string[] = []) =>
  action(saveAsAsset, {
    doing: 'save it as an asset',
    success: (r) => `Saved as asset ${r.name}`,
  }).run(file, existing)

async function name(value?: string) {
  const dialog = await screen.findByRole('alertdialog', {
    name: 'Save as asset',
  })
  const field = within(dialog).getByRole('textbox', {
    name: 'Name in the asset library',
  }) as HTMLInputElement
  expect(field.value).toBe('face.png')
  if (value !== undefined) await fireEvent.input(field, { target: { value } })
  await fireEvent.click(within(dialog).getByRole('button', { name: 'Save' }))
}

const conflict = () =>
  waitFor(() =>
    screen.getByRole('alertdialog', {
      name: '"face.png" is already in the asset library.',
    }),
  )

const closed = () =>
  waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())

it('saves under the typed name and says so', async () => {
  keepOutput.mockResolvedValueOnce(kept('hero.png'))
  const done = save()
  await name('hero.png')
  expect(await done).toEqual({ name: 'hero.png' })
  expect(keepOutput).toHaveBeenCalledWith(file.name, 'hero.png')
  expect(notifyMock.success).toHaveBeenCalledWith('Saved as asset hero.png')
})

it('an existing name asks Replace, Keep both or Cancel', async () => {
  keepOutput
    .mockRejectedValueOnce(taken())
    .mockResolvedValueOnce(kept('face.png'))
  const done = save(['face.png'])
  await name()
  const dialog = await conflict()
  const labels = within(dialog)
    .getAllByRole('button')
    .map((b) => b.textContent?.trim())
  expect(labels.sort()).toEqual(['Cancel', 'Keep both', 'Replace'])
  const replace = within(dialog).getByRole('button', { name: 'Replace' })
  expect(replace.classList.contains('danger')).toBe(true)
  expect(within(dialog).getByRole('button', { name: 'Keep both' }).title).toBe(
    'save it as face-2.png',
  )
  await fireEvent.click(replace)
  expect(await done).toEqual({ name: 'face.png' })
  expect(keepOutput).toHaveBeenLastCalledWith(file.name, 'face.png', true)
  expect(notifyMock.success).toHaveBeenCalledWith('Saved as asset face.png')
})

it('Keep both saves under the free name', async () => {
  keepOutput
    .mockRejectedValueOnce(taken())
    .mockResolvedValueOnce(kept('face-3.png'))
  const done = save(['face.png', 'face-2.png'])
  await name()
  await fireEvent.click(
    within(await conflict()).getByRole('button', { name: 'Keep both' }),
  )
  expect(await done).toEqual({ name: 'face-3.png' })
  expect(keepOutput).toHaveBeenLastCalledWith(file.name, 'face-3.png')
  expect(notifyMock.success).toHaveBeenCalledWith('Saved as asset face-3.png')
})

it('Keep both on a name typed without its extension frees the full name', async () => {
  keepOutput
    .mockRejectedValueOnce(taken())
    .mockResolvedValueOnce(kept('face-2.png'))
  const done = save(['face.png'])
  await name('face')
  await fireEvent.click(
    within(await conflict()).getByRole('button', { name: 'Keep both' }),
  )
  await done
  expect(keepOutput).toHaveBeenLastCalledWith(file.name, 'face-2.png')
})

it('Keep both tries the next name when the listing was behind', async () => {
  keepOutput
    .mockRejectedValueOnce(taken())
    .mockRejectedValueOnce(taken())
    .mockResolvedValueOnce(kept('face-3.png'))
  const done = save(['face.png'])
  await name()
  await fireEvent.click(
    within(await conflict()).getByRole('button', { name: 'Keep both' }),
  )
  expect(await done).toEqual({ name: 'face-3.png' })
  expect(keepOutput.mock.calls.map((c) => c[1])).toEqual([
    'face.png',
    'face-2.png',
    'face-3.png',
  ])
})

it('Cancel at either step makes no request and no toast', async () => {
  // The name dialog: Cancel, then Escape
  let done = save()
  await fireEvent.click(await screen.findByRole('button', { name: 'Cancel' }))
  expect(await done).toBeNull()
  await closed()
  done = save()
  await fireEvent.keyDown(await screen.findByRole('alertdialog'), {
    key: 'Escape',
  })
  expect(await done).toBeNull()
  expect(keepOutput).not.toHaveBeenCalled()
  await closed()

  // The exists question: Cancel, then Escape - only the first try was sent
  for (const answer of ['cancel', 'escape']) {
    keepOutput.mockReset()
    keepOutput.mockRejectedValueOnce(taken())
    done = save(['face.png'])
    await name()
    const dialog = await conflict()
    if (answer === 'cancel')
      await fireEvent.click(
        within(dialog).getByRole('button', { name: 'Cancel' }),
      )
    else await fireEvent.keyDown(dialog, { key: 'Escape' })
    expect(await done).toBeNull()
    expect(keepOutput).toHaveBeenCalledTimes(1)
    await closed()
  }
  for (const fn of Object.values(notifyMock)) expect(fn).not.toHaveBeenCalled()
})

it('a refusal for another reason is a failure, not a question', async () => {
  keepOutput.mockRejectedValueOnce(
    new ApiError('This workspace has no asset library', 409),
  )
  const done = save()
  await name()
  expect(await done).toBeUndefined()
  expect(notifyMock.failure).toHaveBeenCalledWith(
    'save it as an asset',
    expect.objectContaining({ message: 'This workspace has no asset library' }),
  )
  expect(keepOutput).toHaveBeenCalledTimes(1)
})

it('names a bad name and an empty one', () => {
  expect(assetNameError('face.png')).toBeNull()
  expect(assetNameError('')).toBe('Type a name for the asset')
  expect(assetNameError('my face.png')).toBe(
    'Start with a letter, digit or _; then letters, digits, _, - and .',
  )
  expect(assetNameError('x'.repeat(101))).toBe('Use at most 100 characters')
})
