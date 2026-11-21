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
  note: saved ? { id: 'n1', text: saved, created: AT, at: AT } : null,
  about: 'Shot 3',
  line: saved ? { text: 'Saved then.', answered: false } : null,
  onsave: vi.fn(),
  ondelete: vi.fn(),
  onreload: vi.fn(),
  ...over,
})
const LABEL = /^(Edit your note|Note) for the agent on Shot 3$/
const box = () => screen.getByLabelText(LABEL) as HTMLTextAreaElement
const noBox = () =>
  waitFor(() => expect(screen.queryByLabelText(LABEL)).toBeNull())

async function draft(text: string, saved: string | null = null) {
  const given = props(saved)
  render(ItemNote, given)
  await fireEvent.click(
    screen.getByRole('button', {
      name: saved ? 'Edit note' : 'Add a note for the agent',
    }),
  )
  await fireEvent.input(box(), { target: { value: text } })
  return given
}

it('says before a word is written that no agent session is listening', async () => {
  await draft('')
  expect(screen.getByText(/No agent session is listening/)).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Save note' })).toBeTruthy()
  expect(box().maxLength).toBe(4000)
  await waitFor(() => expect(document.activeElement).toBe(box()))
})

it('says Send to agent while a session is listening, and saves either way', async () => {
  const given = props(null, { listening: true })
  given.onsave.mockResolvedValue(undefined)
  const view = render(ItemNote, given)
  await fireEvent.click(
    screen.getByRole('button', { name: 'Add a note for the agent' }),
  )
  expect(
    screen.getByText(
      'An agent session is listening; it will see this within about 10 seconds.',
    ),
  ).toBeTruthy()
  await fireEvent.input(box(), { target: { value: 'warmer' } })
  await fireEvent.click(screen.getByRole('button', { name: 'Send to agent' }))
  await noBox()
  expect(given.onsave).toHaveBeenCalledWith('warmer')
  // the session went away: the same box says so, and still saves
  await view.rerender({ ...given, listening: false })
  await fireEvent.click(
    screen.getByRole('button', { name: 'Add a note for the agent' }),
  )
  await fireEvent.input(box(), { target: { value: 'cooler' } })
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  await waitFor(() => expect(given.onsave).toHaveBeenCalledWith('cooler'))
})

it('a new note shows what it is about, and a reference comes off before it is saved', async () => {
  const { createRawSnippet } = await import('svelte')
  const onremoveref = vi.fn()
  const take = { item: 's3', job: 'j2', output: 'run/s3.mp4' }
  render(ItemNote, {
    ...props(null, { refs: [take, { item: 's5' }], onremoveref }),
    chip: createRawSnippet((ref: () => { item: string; job?: string }) => ({
      render: () => `<span class="chip">${ref().job ?? ref().item}</span>`,
    })),
  })
  expect(
    [...document.querySelectorAll('.chip')].map((c) => c.textContent),
  ).toEqual(['j2', 's5'])
  await fireEvent.click(
    screen.getByRole('button', { name: 'Remove run/s3.mp4' }),
  )
  expect(onremoveref).toHaveBeenCalledWith(take)
})

it('shows a saved note with the line about where it stands', () => {
  render(ItemNote, props('warmer, please'))
  expect(screen.getByText('warmer, please')).toBeTruthy()
  expect(screen.getByText('Saved then.')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Delete note' })).toBeTruthy()
})

it('saves the draft; a saved note emptied is deleted, an empty new one not sent', async () => {
  const given = await draft('needs more red')
  given.onsave.mockResolvedValue(undefined)
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  await noBox()
  expect(given.onsave).toHaveBeenCalledWith('needs more red')
  cleanup()
  const cleared = await draft('   ', 'old words')
  cleared.ondelete.mockResolvedValue(undefined)
  confirmDialog.mockResolvedValue(true)
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  await waitFor(() => expect(cleared.ondelete).toHaveBeenCalledTimes(1))
  expect(cleared.onsave).not.toHaveBeenCalled()
  cleanup()
  const empty = await draft('  ')
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  await noBox()
  expect(empty.onsave).not.toHaveBeenCalled()
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
    screen.getByRole('button', { name: 'Add a note for the agent' }),
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

it('deletes an answered note at once, and asks before one still waiting', async () => {
  const given = props('warmer, please', {
    line: {
      text: 'The agent has updated the plan once since.',
      answered: true,
    },
  })
  given.ondelete.mockResolvedValue(undefined)
  render(ItemNote, given)
  await fireEvent.click(screen.getByRole('button', { name: 'Delete note' }))
  await waitFor(() => expect(given.ondelete).toHaveBeenCalledTimes(1))
  expect(confirmDialog).not.toHaveBeenCalled()
  cleanup()
  const waiting = props('warmer, please')
  render(ItemNote, waiting)
  confirmDialog.mockResolvedValue(false)
  await fireEvent.click(screen.getByRole('button', { name: 'Delete note' }))
  await waitFor(() => expect(confirmDialog).toHaveBeenCalledTimes(1))
  expect(waiting.ondelete).not.toHaveBeenCalled()
  confirmDialog.mockResolvedValue(true)
  await fireEvent.click(screen.getByRole('button', { name: 'Delete note' }))
  await waitFor(() => expect(waiting.ondelete).toHaveBeenCalledTimes(1))
})

it('a waiting note emptied and saved asks first, and stays when declined', async () => {
  const given = await draft('', 'warmer, please')
  confirmDialog.mockResolvedValue(false)
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  await waitFor(() => expect(confirmDialog).toHaveBeenCalledTimes(1))
  expect(confirmDialog.mock.calls[0][0]).toMatch(/^Delete this note\?/)
  expect(given.ondelete).not.toHaveBeenCalled()
  expect(given.onsave).not.toHaveBeenCalled()
  expect(box()).toBeTruthy()
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
    screen.getByRole('button', { name: 'Add a note for the agent' }),
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

it('a draft moves with its key and leaves nothing under the old one', async () => {
  const given = props(null, { draftKey: 'k-one' })
  const view = render(ItemNote, given)
  await fireEvent.click(
    screen.getByRole('button', { name: 'Add a note for the agent' }),
  )
  await fireEvent.input(box(), { target: { value: 'half' } })
  await waitFor(() => expect(sessionStorage.getItem('k-one')).toBe('half'))
  await view.rerender({ ...given, draftKey: 'k-two' })
  await waitFor(() => expect(sessionStorage.getItem('k-two')).toBe('half'))
  expect(sessionStorage.getItem('k-one')).toBeNull()
})

it('asked to open again while open, keeps what is in the box', async () => {
  const view = render(ItemNote, props())
  const note = view.component as unknown as { edit: () => Promise<void> }
  await note.edit()
  await fireEvent.input(box(), { target: { value: 'half' } })
  await note.edit()
  expect(box().value).toBe('half')
})
