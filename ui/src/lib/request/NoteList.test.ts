import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import NoteList from './NoteList.svelte'
import { attempt, item, request } from './fixtures'
import type { LedgerNote, LedgerNoteRef } from '../types'

const api = vi.hoisted(() => ({
  addNote: vi.fn(),
  editNote: vi.fn(),
  deleteNote: vi.fn(),
  notesChannel: vi.fn(),
}))
vi.mock('../api', () => ({ api }))
vi.mock('../confirm.svelte', () => ({ confirmDialog: vi.fn() }))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

beforeEach(() => {
  sessionStorage.clear()
  for (const call of Object.values(api)) call.mockReset()
  api.notesChannel.mockResolvedValue({ listening: false, last_seen: null })
  api.addNote.mockResolvedValue({ revision: 4 })
  api.editNote.mockResolvedValue({ revision: 4 })
  api.deleteNote.mockResolvedValue({ revision: 4 })
})
afterEach(cleanup)

const at = (minute: number) =>
  `2026-01-01T00:${String(minute).padStart(2, '0')}:00+00:00`
const take2: LedgerNoteRef = {
  item: 's3',
  job: 'j2',
  output: 'run/s3.mp4',
  kind: 'video',
  attempt: 2,
}
const notes: LedgerNote[] = [
  { id: 'n1', text: 'warmer light', created: at(5), at: at(5), refs: [take2] },
  {
    id: 'n2',
    text: 'slower push-in',
    created: at(6),
    at: at(9),
    refs: [{ item: 's3', job: 'j1', attempt: 1, gone: true }],
  },
]
const req = (over = {}) =>
  request(
    [
      item('s3', {
        title: 'Shot 3',
        attempts: [attempt('j1', 'failed'), attempt('j2')],
        notes,
        ...over,
      }),
      item('s5', { title: 'Shot 5' }),
      item('s9', { title: 'Cut', dropped: true, status: 'dropped' }),
    ],
    { agent_wrote_at: at(7) },
  )
const props = (over = {}) => ({
  notes,
  request: req(),
  item: 's3' as string | null,
  workspace: 'studio',
  about: 'Shot 3',
  onchanged: vi.fn(),
  ...over,
})
const box = () =>
  screen.getByLabelText('Note for the agent on Shot 3') as HTMLTextAreaElement
async function write(text: string) {
  await fireEvent.click(
    screen.getByRole('button', { name: 'Add a note for the agent' }),
  )
  await fireEvent.input(box(), { target: { value: text } })
}

it('shows each note with its line and what it is about, under one label', () => {
  const { container } = render(NoteList, props())
  expect(screen.getAllByText('Note for the agent')).toHaveLength(1)
  const items = [...container.querySelectorAll('.list > li')] as HTMLElement[]
  expect(items.map((li) => li.id)).toEqual([
    'request-note-n1',
    'request-note-n2',
  ])
  const first = within(items[0])
  expect(first.getByText('warmer light')).toBeTruthy()
  expect(first.getByText(/The agent has updated the plan since/)).toBeTruthy()
  const file = first.getByRole('link', { name: 'take 2 · s3.mp4' })
  expect(file.getAttribute('href')).toContain('/outputs/run/s3.mp4')
  const second = within(items[1])
  expect(second.getByText(/ · edited /)).toBeTruthy()
  expect(second.getByText(/No agent session is listening/)).toBeTruthy()
  const gone = second.getByText('take 1')
  expect(gone.getAttribute('title')).toBe('no longer exists')
  expect(gone.closest('a')).toBeNull()
})

it('adds a note with the box’s references, then asks the page to read again', async () => {
  const given = props({ pendingRefs: [take2] })
  render(NoteList, given)
  await write('reroll this one')
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  await waitFor(() => expect(given.onchanged).toHaveBeenCalledTimes(1))
  expect(api.addNote).toHaveBeenCalledWith('studio', {
    request: 'r1',
    item: 's3',
    text: 'reroll this one',
    refs: [{ item: 's3', job: 'j2', output: 'run/s3.mp4' }],
  })
  // asked just before the save, as well as on mount
  expect(api.notesChannel.mock.calls.length).toBeGreaterThanOrEqual(2)
  // the box's references went with the note
  expect(screen.queryByRole('button', { name: /^Remove/ })).toBeNull()
})

it('a box with every reference taken off sends none', async () => {
  render(NoteList, props({ pendingRefs: [take2, { item: 's3', job: 'j1' }] }))
  await write('just the item')
  for (const remove of screen.getAllByRole('button', { name: /^Remove/ }))
    await fireEvent.click(remove)
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  await waitFor(() => expect(api.addNote).toHaveBeenCalled())
  expect(api.addNote.mock.calls[0][1]).toEqual({
    request: 'r1',
    item: 's3',
    text: 'just the item',
  })
})

it("the request's box picks any live item, once each, as a reference", async () => {
  const ongo = vi.fn()
  render(
    NoteList,
    props({
      notes: [],
      item: null,
      about: 'this request',
      pickItems: true,
      ongo,
    }),
  )
  await fireEvent.click(
    screen.getByRole('button', { name: 'Add a note for the agent' }),
  )
  const picker = () =>
    screen.getByLabelText('Add an item this note is about') as HTMLSelectElement
  // keyed, so an item reads as a reference and not as a suggested note
  expect([...picker().options].map((o) => o.textContent)).toEqual([
    '+ Add an item',
    's3 · Shot 3',
    's5 · Shot 5',
  ])
  picker().value = 's5'
  await fireEvent.change(picker())
  expect(picker().value).toBe('')
  // a picked item is not offered again
  expect([...picker().options].map((o) => o.value)).toEqual(['', 's3'])
  picker().value = 's3'
  await fireEvent.change(picker())
  // every item picked: the control goes, the About line stays
  expect(screen.queryByLabelText('Add an item this note is about')).toBeNull()
  expect(screen.getByText('About')).toBeTruthy()
  const chips = screen
    .getByRole('list', { name: 'What the note is about' })
    .querySelectorAll('.ref')
  expect([...chips].map((c) => c.textContent)).toEqual(['Shot 5', 'Shot 3'])
  await fireEvent.click(chips[0])
  expect(ongo).toHaveBeenCalledWith('s5')
  await fireEvent.input(
    screen.getByLabelText('Note for the agent on this request'),
    { target: { value: 'warmer, both' } },
  )
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  await waitFor(() => expect(api.addNote).toHaveBeenCalled())
  expect(api.addNote.mock.calls[0][1]).toEqual({
    request: 'r1',
    text: 'warmer, both',
    refs: [{ item: 's5' }, { item: 's3' }],
  })
})

it('the save button and the waiting line follow whether a session listens', async () => {
  api.notesChannel.mockResolvedValue({ listening: true, last_seen: 'now' })
  render(NoteList, props({ request: req({ notes }), notes }))
  await write('x')
  expect(
    await screen.findByRole('button', { name: 'Send to agent' }),
  ).toBeTruthy()
  expect(
    within(document.getElementById('request-note-n2') as HTMLElement).getByText(
      /An agent session is listening/,
    ),
  ).toBeTruthy()
})

it('edits and deletes a note through the note routes', async () => {
  const given = props()
  render(NoteList, given)
  const second = within(
    document.getElementById('request-note-n2') as HTMLElement,
  )
  await fireEvent.click(second.getByRole('button', { name: 'Edit note' }))
  await fireEvent.input(
    second.getByLabelText('Edit your note for the agent on Shot 3'),
    { target: { value: 'slower still' } },
  )
  await fireEvent.click(second.getByRole('button', { name: 'Save note' }))
  await waitFor(() =>
    expect(api.editNote).toHaveBeenCalledWith('studio', {
      request: 'r1',
      note: 'n2',
      text: 'slower still',
    }),
  )
  const first = within(
    document.getElementById('request-note-n1') as HTMLElement,
  )
  await fireEvent.click(first.getByRole('button', { name: 'Delete note' }))
  await waitFor(() =>
    expect(api.deleteNote).toHaveBeenCalledWith('studio', 'r1', 'n1'),
  )
  expect(given.onchanged).toHaveBeenCalledTimes(2)
})

it('a dropped item shows its notes and takes no more; a filter narrows them', () => {
  render(NoteList, props({ item: 's9', about: 'Cut', notes: [notes[0]] }))
  expect(screen.getByText('warmer light')).toBeTruthy()
  expect(screen.queryByRole('button')).toBeNull()
  cleanup()
  render(NoteList, props({ filter: (n: LedgerNote) => n.id === 'n2' }))
  expect(screen.queryByText('warmer light')).toBeNull()
  expect(screen.getByText('slower push-in')).toBeTruthy()
})

it('a refusal from the note routes is said beside the draft, which stays', async () => {
  const { ApiError } = await import('../apiErrors')
  api.addNote.mockRejectedValue(
    new ApiError('job j9 is not linked to s3', 422, 'invalid_note_ref'),
  )
  render(NoteList, props())
  await write('my words')
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  expect((await screen.findByRole('alert')).textContent).toContain(
    'does not hold any more (job j9 is not linked to s3)',
  )
  expect(box().value).toBe('my words')
})

it('tells the page of a stored note once, and keeps the sent draft out of the tab', async () => {
  const onadded = vi.fn()
  api.addNote.mockResolvedValue({ note: notes[0], revision: 4 })
  render(NoteList, props({ pendingRefs: [take2], onadded }))
  await write('reroll this one')
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
  await waitFor(() => expect(onadded).toHaveBeenCalledTimes(1))
  expect(onadded).toHaveBeenCalledWith(notes[0])
  const kept = Object.keys(sessionStorage).map((k) => sessionStorage.getItem(k))
  expect(kept.join('')).not.toContain('reroll this one')
})

const MOVE = 'For the whole request instead'

it('offers the move only on an item’s new-note box, and not while it is empty', async () => {
  const onmove = vi.fn()
  render(NoteList, props({ onmove }))
  expect(screen.queryByRole('button', { name: MOVE })).toBeNull()
  await fireEvent.click(
    screen.getByRole('button', { name: 'Add a note for the agent' }),
  )
  const button = screen.getByRole('button', { name: MOVE }) as HTMLButtonElement
  expect(button.disabled).toBe(true)
  await fireEvent.input(box(), { target: { value: 'all warmer' } })
  expect(button.disabled).toBe(false)
  // a saved note's edit box has none
  expect(screen.getAllByRole('button', { name: MOVE })).toHaveLength(1)
})

it('has no move on the request’s own box', async () => {
  render(
    NoteList,
    props({
      item: null,
      about: 'this request',
      onmove: vi.fn(),
      notes: [],
    }),
  )
  await fireEvent.click(
    screen.getByRole('button', { name: 'Add a note for the agent' }),
  )
  expect(screen.queryByRole('button', { name: MOVE })).toBeNull()
})

it('moves the draft with the item, or the take it was about, and empties the box', async () => {
  const onmove = vi.fn()
  render(NoteList, props({ onmove }))
  await write('all warmer')
  await fireEvent.click(screen.getByRole('button', { name: MOVE }))
  expect(onmove).toHaveBeenCalledWith('all warmer', [{ item: 's3' }])
  expect(api.addNote).not.toHaveBeenCalled()
  expect(screen.queryByLabelText('Note for the agent on Shot 3')).toBeNull()
  await fireEvent.click(
    screen.getByRole('button', { name: 'Add a note for the agent' }),
  )
  expect(box().value).toBe('')
})

it('moves a take the box was about along with the words', async () => {
  const onmove = vi.fn()
  render(NoteList, props({ onmove, pendingRefs: [take2] }))
  await write('all warmer')
  await fireEvent.click(screen.getByRole('button', { name: MOVE }))
  expect(onmove.mock.calls[0][1]).toEqual([
    expect.objectContaining({ item: 's3', job: 'j2' }),
  ])
})

it('a seeded add merges refs and appends to a draft already there', async () => {
  const view = render(NoteList, {
    ...props({ item: null, about: 'this request', notes: [] }),
    pickItems: true,
  } as never)
  const request = () =>
    screen.getByLabelText(
      'Note for the agent on this request',
    ) as HTMLTextAreaElement
  await view.component.add({ text: 'first', refs: [{ item: 's3' }] })
  await waitFor(() => expect(request().value).toBe('first'))
  await view.component.add({
    text: 'second',
    refs: [{ item: 's3' }, { item: 's5' }],
  })
  await waitFor(() => expect(request().value).toBe('first\n\nsecond'))
  const about = screen.getByRole('list', { name: 'What the note is about' })
  expect(within(about).getAllByRole('listitem')).toHaveLength(2)
})
