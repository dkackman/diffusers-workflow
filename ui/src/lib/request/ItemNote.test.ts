import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import ItemNote from './ItemNote.svelte'
import { ApiError } from '../apiErrors'

const confirmDialog = vi.hoisted(() => vi.fn())
vi.mock('../confirm.svelte', () => ({ confirmDialog }))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

beforeEach(() => {
  confirmDialog.mockReset()
  sessionStorage.clear()
})
afterEach(cleanup)

const AT = '2026-01-02T03:04:05+00:00'
const props = (saved: string | null = null, over = {}) => ({
  comment: saved ? { text: saved, at: AT } : null,
  about: 'Shot 3',
  line: saved ? { text: 'Saved then.', answered: false } : null,
  onsave: vi.fn(),
  onreload: vi.fn(),
  ...over,
})
const box = () =>
  screen.getByLabelText('Note for the agent on Shot 3') as HTMLTextAreaElement
const noBox = () =>
  waitFor(() =>
    expect(screen.queryByLabelText('Note for the agent on Shot 3')).toBeNull(),
  )

async function draft(text: string, saved: string | null = null) {
  const given = props(saved)
  render(ItemNote, given)
  await fireEvent.click(
    screen.getByRole('button', {
      name: saved ? 'Edit note' : 'Leave a note for the agent',
    }),
  )
  await fireEvent.input(box(), { target: { value: text } })
  return given
}

it('says before a word is written that nothing is sent to the agent', async () => {
  await draft('')
  expect(screen.getByText(/Nothing is sent to the agent/)).toBeTruthy()
  expect(box().maxLength).toBe(4000)
  await waitFor(() => expect(document.activeElement).toBe(box()))
})

it('shows a saved note with the line about where it stands', () => {
  render(ItemNote, props('warmer, please'))
  expect(screen.getByText('warmer, please')).toBeTruthy()
  expect(screen.getByText('Saved then.')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Clear note' })).toBeNull()
})

it('saves the draft, and an empty one as no note', async () => {
  const given = await draft('needs more red')
  given.onsave.mockResolvedValue(undefined)
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  await noBox()
  expect(given.onsave).toHaveBeenCalledWith('needs more red')
  cleanup()
  const cleared = await draft('   ', 'old words')
  cleared.onsave.mockResolvedValue(undefined)
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  await waitFor(() => expect(cleared.onsave).toHaveBeenCalledWith(null))
})

it('Ctrl+Enter saves', async () => {
  const given = await draft('needs more red')
  given.onsave.mockResolvedValue(undefined)
  await fireEvent.keyDown(box(), { key: 'Enter', ctrlKey: true })
  await waitFor(() =>
    expect(given.onsave).toHaveBeenCalledWith('needs more red'),
  )
})

it('Esc closes the box and keeps the draft for the next time', async () => {
  await draft('half a thought')
  await fireEvent.keyDown(box(), { key: 'Escape' })
  await noBox()
  expect(confirmDialog).not.toHaveBeenCalled()
  expect(screen.getByText('You have an unsaved draft.')).toBeTruthy()
  await fireEvent.click(
    screen.getByRole('button', { name: 'Continue the draft' }),
  )
  expect(box().value).toBe('half a thought')
})

it('Cancel asks before discarding a draft, and keeps it when declined', async () => {
  await draft('needs more red')
  confirmDialog.mockResolvedValue(false)
  await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  await waitFor(() => expect(confirmDialog).toHaveBeenCalledTimes(1))
  expect(box().value).toBe('needs more red')
  confirmDialog.mockResolvedValue(true)
  await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  await noBox()
  // discarded: the next edit starts from what is saved
  await fireEvent.click(
    screen.getByRole('button', { name: 'Leave a note for the agent' }),
  )
  expect(box().value).toBe('')
})

it('Cancel does not ask over an empty or unchanged draft', async () => {
  await draft('as saved', 'as saved')
  await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  await noBox()
  expect(confirmDialog).not.toHaveBeenCalled()
  expect(screen.getByText('as saved')).toBeTruthy()
})

it('a refresh that hands down the note again leaves the draft alone', async () => {
  const view = render(ItemNote, props('first'))
  await fireEvent.click(screen.getByRole('button', { name: 'Edit note' }))
  await fireEvent.input(box(), { target: { value: 'first, and more' } })
  await view.rerender(props('first'))
  expect(box().value).toBe('first, and more')
})

it('says a stale revision in plain words, with a reload that keeps the draft', async () => {
  const given = await draft('my draft')
  given.onsave.mockRejectedValue(
    new ApiError(
      'at revision 5, not 3 - read it again and rewrite the plan against revision 5',
      409,
      'stale_revision',
    ),
  )
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  const alert = await screen.findByRole('alert')
  expect(alert.textContent).toContain('The plan changed while you were writing')
  expect(alert.textContent).not.toContain('rewrite the plan')
  await fireEvent.click(screen.getByRole('button', { name: 'Reload the plan' }))
  expect(given.onreload).toHaveBeenCalledTimes(1)
  await waitFor(() => expect(screen.queryByRole('alert')).toBeNull())
  expect(box().value).toBe('my draft')
})

it("shows another refusal as the server's message, with no reload", async () => {
  const given = await draft('my draft')
  given.onsave.mockRejectedValue(
    new ApiError(
      'this workspace has no root folder, so it cannot hold requests',
      409,
      'no_workspace_root',
    ),
  )
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  expect((await screen.findByRole('alert')).textContent).toContain(
    'no root folder',
  )
  expect(screen.queryByRole('button', { name: /Reload/ })).toBeNull()
  expect(box().value).toBe('my draft')
})

it('a failure that is not a refusal is a toast, and the draft stays', async () => {
  const { notify } = await import('../toast')
  const given = await draft('my draft')
  given.onsave.mockRejectedValue(new TypeError('Failed to fetch'))
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  await waitFor(() =>
    expect(notify.error).toHaveBeenCalledWith('Failed to fetch'),
  )
  expect(screen.queryByRole('alert')).toBeNull()
  expect(box().value).toBe('my draft')
})

it('counts the characters only near the limit', async () => {
  await draft('short')
  expect(screen.queryByText(/\/ 4000/)).toBeNull()
  await fireEvent.input(box(), { target: { value: 'x'.repeat(3700) } })
  expect(screen.getByText('3700 / 4000')).toBeTruthy()
})

it('offers to clear a note once the agent has written since', async () => {
  const given = props('warmer, please', {
    line: {
      text: 'The agent has updated the plan once since.',
      answered: true,
    },
  })
  given.onsave.mockResolvedValue(undefined)
  render(ItemNote, given)
  await fireEvent.click(screen.getByRole('button', { name: 'Clear note' }))
  await waitFor(() => expect(given.onsave).toHaveBeenCalledWith(null))
})

it('shows a note it cannot edit, and nothing where there is none to show', () => {
  render(ItemNote, props('too long', { editable: false }))
  expect(screen.getByText('too long')).toBeTruthy()
  expect(screen.queryByRole('button')).toBeNull()
  cleanup()
  const { container } = render(ItemNote, props(null, { editable: false }))
  expect(container.textContent?.trim()).toBe('')
})

it('keeps an unsaved draft for the tab, so a reload does not lose it', async () => {
  const draftKey = 'dw-request-note:studio/r1/s3'
  const given = props(null, { draftKey })
  const first = render(ItemNote, given)
  await fireEvent.click(
    screen.getByRole('button', { name: 'Leave a note for the agent' }),
  )
  await fireEvent.input(box(), { target: { value: 'half a thought' } })
  await waitFor(() =>
    expect(sessionStorage.getItem(draftKey)).toBe('half a thought'),
  )
  // the page goes and comes back
  first.unmount()
  render(ItemNote, given)
  expect(screen.getByText('You have an unsaved draft.')).toBeTruthy()
  await fireEvent.click(
    screen.getByRole('button', { name: 'Continue the draft' }),
  )
  expect(box().value).toBe('half a thought')
  // saved: nothing left behind
  given.onsave.mockResolvedValue(undefined)
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  await noBox()
  await waitFor(() => expect(sessionStorage.getItem(draftKey)).toBeNull())
})

it('a kept draft that is already what was saved is not offered again', () => {
  const draftKey = 'dw-request-note:studio/r1/s3'
  sessionStorage.setItem(draftKey, 'the same')
  render(ItemNote, props('the same', { draftKey }))
  expect(screen.queryByText('You have an unsaved draft.')).toBeNull()
})
