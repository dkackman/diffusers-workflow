import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import LinkedNotes from './LinkedNotes.svelte'
import { LinkedRequest } from './linkedRequest.svelte'
import { api, resetApi } from '../test/apiMock'
import { attempt, item, request } from './fixtures'
import type { LedgerNote } from '../types'

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
vi.mock('../confirm.svelte', () => ({ confirmDialog: vi.fn() }))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), news: vi.fn() } }))

beforeEach(() => {
  sessionStorage.clear()
  resetApi()
  api.notesChannel.mockResolvedValue({ listening: false, last_seen: null })
  api.editNote.mockResolvedValue({ revision: 4 })
  api.deleteNote.mockResolvedValue({ revision: 4 })
})
afterEach(cleanup)

const at = '2026-01-01T00:05:00+00:00'
const sent = (id: string): LedgerNote => ({
  id,
  text: 'warmer',
  created: at,
  at,
})
const board = (notes: LedgerNote[] = []) =>
  request([item('s3', { title: 'Shot 3', attempts: [attempt('j1')], notes })])
const jobRef = { item: 's3', job: 'j1' }

async function mount() {
  api.getRequest.mockResolvedValue(board())
  const linked = new LinkedRequest('studio', 'r1')
  await linked.read()
  render(LinkedNotes, {
    linked,
    item: 's3',
    filter: () => true,
    refs: [jobRef],
  })
}
async function send(text: string) {
  await fireEvent.click(
    screen.getByRole('button', { name: 'Add a note for the agent' }),
  )
  await fireEvent.input(screen.getByLabelText(/^Note for the agent on/), {
    target: { value: text },
  })
  await fireEvent.click(screen.getByRole('button', { name: 'Save note' }))
}

it('the next note is about the job again after a note is added', async () => {
  await mount()
  api.addNote.mockResolvedValueOnce({ note: sent('n1'), revision: 4 })
  api.getRequest.mockResolvedValue(board([sent('n1')]))
  await send('first')
  await waitFor(() => expect(api.addNote).toHaveBeenCalledTimes(1))
  api.addNote.mockResolvedValueOnce({ note: sent('n2'), revision: 5 })
  await waitFor(() =>
    expect(screen.getByRole('button', { name: 'Add a note for the agent' })),
  )
  await send('second')
  await waitFor(() => expect(api.addNote).toHaveBeenCalledTimes(2))
  expect(api.addNote.mock.calls[1][1].refs).toEqual([jobRef])
})

it('a lost re-read after an add does not drop the job from the next note', async () => {
  await mount()
  api.addNote.mockResolvedValueOnce({ note: sent('n1'), revision: 4 })
  api.getRequest.mockRejectedValue(
    Object.assign(new Error('boom'), { status: 500 }),
  )
  await send('first')
  await waitFor(() => expect(api.addNote).toHaveBeenCalledTimes(1))
  api.addNote.mockResolvedValueOnce({ note: sent('n2'), revision: 5 })
  await waitFor(() =>
    expect(screen.getByRole('button', { name: 'Add a note for the agent' })),
  )
  await send('second')
  await waitFor(() => expect(api.addNote).toHaveBeenCalledTimes(2))
  expect(api.addNote.mock.calls[0][1].refs).toEqual([jobRef])
  expect(api.addNote.mock.calls[1][1].refs).toEqual([jobRef])
})
