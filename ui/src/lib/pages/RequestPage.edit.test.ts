import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import RequestPage from './RequestPage.svelte'
import { workspace } from '../workspaceState.svelte'
import { fakeLedger } from '../request/fakeLedger'
import { attempt, doneItem, entry, item, request } from '../request/fixtures'
import type { LedgerRequest } from '../types'

// The page's edits against a ledger that keeps the server's rules: a stale
// revision is refused, a dropped item with no job is deleted
const world = vi.hoisted(() => ({
  ledger: null as unknown as import('../request/fakeLedger').FakeLedger,
  confirm: vi.fn(),
  notify: { error: vi.fn(), success: vi.fn(), undo: vi.fn() },
}))
vi.mock('../api', async () => ({
  ApiError: (await import('../apiErrors')).ApiError,
  api: {
    getRequest: () => world.ledger.get(),
    putRequest: (_ws: string, _id: string, body: never) =>
      world.ledger.put(body),
    galleryThumbnailUrl: (name: string) => `/thumb/${name}`,
  },
}))
vi.mock('../toast', () => ({ notify: world.notify }))
vi.mock('../confirm.svelte', () => ({
  confirmDialog: (...args: unknown[]) => world.confirm(...args),
}))

const make = (): LedgerRequest =>
  request(
    [
      doneItem('cast', {
        title: 'Cast sheet',
        attempts: [attempt('j1'), attempt('j2')],
        result: { job: 'j2', run_id: 'r2', output: null },
      }),
      item('s3', {
        title: 'Shot 3',
        depends_on: ['cast'],
        ready: true,
        attempts: [attempt('j3', 'failed')],
        status: 'failed',
      }),
      item('s4', { title: 'Shot 4', depends_on: ['s3'] }),
      item('cut', { title: 'The cut', depends_on: ['s3'] }),
    ],
    { history: [entry(3, 'agent', 'opened')] },
  )

beforeEach(() => {
  sessionStorage.clear()
  vi.clearAllMocks()
  workspace.current = 'studio'
  world.ledger = fakeLedger(make())
  world.confirm.mockResolvedValue(true)
})
afterEach(() => {
  cleanup()
  document.body.innerHTML = ''
})

const start = async () => {
  render(RequestPage, { requestId: 'r1' })
  await screen.findByRole('heading', { name: 'Launch posters' })
}
const row = (key: string) =>
  document.querySelector(`li.item[data-key="${key}"]`) as HTMLElement
const order = () =>
  [...document.querySelectorAll('ol.items > li')].map(
    (li) => (li as HTMLElement).dataset.key,
  )
const whys = () => world.ledger.puts.map((p) => p.why)
const saved = (n: number) =>
  waitFor(() => expect(world.ledger.current.revision).toBe(n))
async function choose(key: string, entryName: string) {
  await fireEvent.click(
    within(row(key)).getByRole('button', { name: /^More actions for/ }),
  )
  const list = await screen.findByRole('dialog')
  await fireEvent.click(within(list).getByRole('button', { name: entryName }))
}
const undo = () => world.notify.undo.mock.calls.at(-1)?.[1]()

it('accepts a take at once, and the toast takes it back', async () => {
  await start()
  const cast = within(row('cast'))
  const first = cast.getByRole('button', {
    name: 'Accept take 1 for Cast sheet',
  })
  await fireEvent.click(first)
  // at once: before the server has answered
  expect(first.getAttribute('aria-pressed')).toBe('true')
  expect(first.textContent).toBe('Accepted')
  expect(world.notify.undo).toHaveBeenCalledWith(
    'Accepted take 1 for Cast sheet',
    expect.any(Function),
  )
  await saved(4)
  expect(world.ledger.current.items[0].accepted?.job).toBe('j1')
  expect(whys()).toEqual(['accept j1 for cast'])
  undo()
  await saved(5)
  expect(world.ledger.current.items[0].accepted).toBeNull()
  await waitFor(() => expect(first.getAttribute('aria-pressed')).toBe('false'))
})

it('a drop of an item others wait on asks first, saying what it does to them', async () => {
  await start()
  expect(row('s4').textContent).toContain('waits on Shot 3')
  await choose('s3', 'Drop from plan')
  await waitFor(() => expect(world.confirm).toHaveBeenCalled())
  expect(world.confirm).toHaveBeenCalledWith(
    'Shot 4 and The cut wait on this item. Dropping it removes that dependency, so they no longer wait for it.',
    { confirmLabel: 'Drop' },
  )
  await waitFor(() => expect(order()).toEqual(['cast', 's4', 'cut']))
  // out of the plan, into the dropped group, and nothing waits on it now
  expect(screen.getByText('Dropped (1)')).toBeTruthy()
  expect(row('s4').textContent).not.toContain('waits on')
  await saved(4)
  const [body] = world.ledger.puts
  expect(body.why).toBe('drop s3')
  expect(body.items.find((i) => i.key === 's4')?.depends_on).toEqual([])
  expect(body.items.find((i) => i.key === 'cut')?.depends_on).toEqual([])
  // focus went to the row that took its place
  expect(document.activeElement?.id).toBe('req-item-s4-title')
  expect(world.notify.undo).toHaveBeenCalledWith(
    'Dropped Shot 3',
    expect.any(Function),
  )
  // undo: the item, its place and the edges come back
  undo()
  await saved(5)
  await waitFor(() => expect(order()).toEqual(['cast', 's3', 's4', 'cut']))
  expect(world.ledger.current.items.map((i) => i.depends_on)).toEqual([
    [],
    ['cast'],
    ['s3'],
    ['s3'],
  ])
})

it('declining the confirm drops nothing', async () => {
  world.confirm.mockResolvedValue(false)
  await start()
  await choose('s3', 'Drop from plan')
  await waitFor(() => expect(world.confirm).toHaveBeenCalled())
  await new Promise((r) => setTimeout(r, 10))
  expect(order()).toEqual(['cast', 's3', 's4', 'cut'])
  expect(world.ledger.puts).toHaveLength(0)
  expect(world.notify.undo).not.toHaveBeenCalled()
})

it('an item nothing was run for is deleted by the drop, and Restore brings it back', async () => {
  await start()
  await choose('cut', 'Drop from plan')
  await waitFor(() => expect(world.confirm).toHaveBeenCalled())
  expect(world.confirm.mock.calls[0][0]).toBe(
    'Nothing has been run for this item, so dropping it deletes it from the plan. This page can restore it only until you leave.',
  )
  await saved(4)
  expect(world.ledger.current.items.map((i) => i.key)).toEqual([
    'cast',
    's3',
    's4',
  ])
  // the page still holds it, among the dropped
  await fireEvent.click(
    within(row('cut')).getByRole('button', { name: 'Restore' }),
  )
  await saved(5)
  expect(world.ledger.current.items.map((i) => i.key)).toEqual([
    'cast',
    's3',
    's4',
    'cut',
  ])
  await waitFor(() => expect(order()).toEqual(['cast', 's3', 's4', 'cut']))
  // remembered, so it went back where it was: nothing more to say
  expect(world.notify.success).not.toHaveBeenCalled()
})

it('a dropped item restored on a fresh page goes last, and the page says what it lost', async () => {
  const held = make()
  held.items.push(
    item('old', {
      title: 'Old shot',
      dropped: true,
      status: 'dropped',
      depends_on: ['cast', 'gone'],
      attempts: [attempt('jo', 'failed')],
    }),
  )
  world.ledger = fakeLedger(held)
  await start()
  await fireEvent.click(
    within(row('old')).getByRole('button', { name: 'Restore' }),
  )
  await saved(4)
  await waitFor(() =>
    expect(world.notify.success).toHaveBeenCalledWith(
      'Restored Old shot at the end of the plan. It no longer waits on gone, which is no longer in the plan.',
    ),
  )
  expect(order()).toEqual(['cast', 's3', 's4', 'cut', 'old'])
})

it('the handle moves an item with the keys: one write for the run, focus kept, each move said', async () => {
  await start()
  const handle = within(row('cut')).getByRole('button', {
    name: 'Reorder The cut, position 4 of 4',
  })
  expect(handle.getAttribute('title')).toBe(
    'Drag to reorder. Ready items run in this order.',
  )
  handle.focus()
  await fireEvent.keyDown(handle, { key: 'ArrowUp' })
  await waitFor(() => expect(order()).toEqual(['cast', 's3', 'cut', 's4']))
  expect(screen.getByTestId('request-live').textContent).toBe(
    'The cut moved to position 3 of 4.',
  )
  await fireEvent.keyDown(handle, { key: 'Home' })
  await waitFor(() => expect(order()).toEqual(['cut', 'cast', 's3', 's4']))
  expect(document.activeElement?.id).toBe('req-item-cut-handle')
  expect(
    within(row('cut')).getByRole('button', {
      name: 'Reorder The cut, position 1 of 4',
    }),
  ).toBeTruthy()
  // nothing is written while the moving goes on
  expect(world.ledger.puts).toHaveLength(0)
  // 800 ms after the last move
  await waitFor(() => expect(world.ledger.puts).toHaveLength(1), {
    timeout: 2500,
  })
  expect(world.ledger.puts[0]).toMatchObject({ why: 'reorder cut' })
  expect(world.ledger.puts[0].items.map((i) => i.key)).toEqual([
    'cut',
    'cast',
    's3',
    's4',
  ])
})

it('leaving the handle writes the order without waiting', async () => {
  await start()
  const handle = within(row('cast')).getByRole('button', { name: /^Reorder/ })
  handle.focus()
  await fireEvent.keyDown(handle, { key: 'End' })
  await waitFor(() => expect(order()).toEqual(['s3', 's4', 'cut', 'cast']))
  expect(world.ledger.puts).toHaveLength(0)
  await fireEvent.blur(document.getElementById('req-item-cast-handle')!)
  await saved(4)
  expect(whys()).toEqual(['reorder cast'])
})

it('the menu moves an item too, written at once', async () => {
  await start()
  await choose('s4', 'Move to top')
  await saved(4)
  expect(world.ledger.current.items.map((i) => i.key)).toEqual([
    's4',
    'cast',
    's3',
    'cut',
  ])
})

it('"Leave a note" in the menu opens the row and its box', async () => {
  await start()
  await choose('s4', 'Leave a note for the agent')
  const box = await screen.findByLabelText('Note for the agent on Shot 4')
  await waitFor(() => expect(document.activeElement).toBe(box))
})

it('a drop of an item the agent has changed is held back, and the row asks', async () => {
  await start()
  world.ledger.agent('retake with warmer light', (items) =>
    items.map((i) =>
      i.key === 's3' ? { ...i, attempts: [...i.attempts, attempt('j9')] } : i,
    ),
  )
  await choose('s3', 'Drop from plan')
  const card = await screen.findByRole('alert')
  // on the row it is about
  expect(row('s3').contains(card)).toBe(true)
  expect(card.textContent).toContain(
    "While you were dropping this, the agent changed it (r4: 'retake with warmer light'): 1 new take.",
  )
  // not dropped: the row is back in the plan, and nothing was written
  expect(order()).toEqual(['cast', 's3', 's4', 'cut'])
  expect(world.ledger.current.revision).toBe(4)
  await fireEvent.click(
    within(card).getByRole('button', { name: 'Drop anyway' }),
  )
  await saved(5)
  expect(world.ledger.current.items.at(-1)).toMatchObject({
    key: 's3',
    dropped: true,
  })
  // never the plan the page read first
  expect(world.ledger.puts.map((p) => p.revision)).toEqual([3, 4])
  expect(screen.queryByRole('alert')).toBeNull()
})

it('"Keep it" leaves the item and writes nothing', async () => {
  await start()
  world.ledger.agent('renamed', (items) =>
    items.map((i) => (i.key === 's3' ? { ...i, title: 'Shot 3b' } : i)),
  )
  await choose('s3', 'Drop from plan')
  const card = await screen.findByRole('alert')
  await fireEvent.click(within(card).getByRole('button', { name: 'Keep it' }))
  expect(screen.queryByRole('alert')).toBeNull()
  expect(world.ledger.current.revision).toBe(4)
  expect(order()).toEqual(['cast', 's3', 's4', 'cut'])
})

it('an accept goes on top of an agent write to another item, and says so', async () => {
  await start()
  world.ledger.agent('added shot 7', (items) => [
    ...items,
    item('s7', { title: 'Shot 7' }),
  ])
  await fireEvent.click(
    within(row('cast')).getByRole('button', {
      name: 'Accept take 2 for Cast sheet',
    }),
  )
  await saved(5)
  expect((await screen.findByRole('status')).textContent).toContain(
    "The agent changed the plan while you were editing (r4: 'added shot 7'). Your change was applied on top.",
  )
  expect(world.ledger.current.items[0].accepted?.job).toBe('j2')
  expect(order()).toContain('s7')
})

it('a note on an item the agent removed is not saved, and the page keeps its words', async () => {
  await start()
  await fireEvent.click(
    within(row('s4')).getByRole('button', { name: 'Shot 4' }),
  )
  await fireEvent.click(
    within(row('s4')).getByRole('button', {
      name: 'Leave a note for the agent',
    }),
  )
  await fireEvent.input(screen.getByLabelText('Note for the agent on Shot 4'), {
    target: { value: 'keep the rain' },
  })
  world.ledger.agent('merged into shot 3', (items) =>
    items.filter((i) => i.key !== 's4'),
  )
  await fireEvent.click(
    within(row('s4')).getByRole('button', { name: 'Save note' }),
  )
  const card = await screen.findByRole('alert')
  expect(card.textContent).toContain(
    "Not applied: the agent removed Shot 4 in r4 ('merged into shot 3').",
  )
  expect(
    (
      within(card).getByLabelText(
        'Your note, which was not saved',
      ) as HTMLTextAreaElement
    ).value,
  ).toBe('keep the rain')
  expect(row('s4')).toBeNull()
  expect(world.ledger.current.revision).toBe(4)
})

it('a write that fails says so, and the edit is taken back', async () => {
  const { ApiError } = await import('../api')
  await start()
  vi.spyOn(world.ledger, 'put').mockRejectedValueOnce(
    new ApiError('the ledger is busy', 503, 'ledger_busy'),
  )
  const first = within(row('cast')).getByRole('button', {
    name: 'Accept take 1 for Cast sheet',
  })
  await fireEvent.click(first)
  await waitFor(() =>
    expect(world.notify.error).toHaveBeenCalledWith(
      'Not saved: the ledger is busy',
    ),
  )
  expect(first.getAttribute('aria-pressed')).toBe('false')
})
