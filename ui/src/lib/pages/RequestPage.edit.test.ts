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
import { PlanStore } from '../request/planStore.svelte'
import {
  attempt,
  doneItem,
  entry,
  item,
  request,
  revisedItem,
} from '../request/fixtures'
import type { LedgerRequest } from '../types'
import { feed } from '../test/feedMock'

// The page's edits against a ledger that keeps the server's rules: a stale
// revision is refused, a dropped item with no job is deleted
const world = vi.hoisted(() => ({
  ledger: null as unknown as import('../request/fakeLedger').FakeLedger,
  confirm: vi.fn(),
  validate: vi.fn(),
  submitJob: vi.fn(),
  deleteRequest: vi.fn(),
  notify: {
    error: vi.fn(),
    failure: vi.fn(),
    success: vi.fn(),
    undo: vi.fn(),
    dismiss: vi.fn(),
    news: vi.fn(),
  },
}))
vi.mock('../api', async () => ({
  ApiError: (await import('../apiErrors')).ApiError,
  api: {
    getRequest: () => world.ledger.get(),
    putRequest: (_ws: string, _id: string, body: never) =>
      world.ledger.put(body),
    addNote: (_ws: string, body: never) => world.ledger.addNote(body),
    notesChannel: async () => ({ listening: false, last_seen: null }),
    galleryThumbnailUrl: (name: string) => `/thumb/${name}`,
    validate: (...args: unknown[]) => world.validate(...args),
    submitJob: (...args: unknown[]) => world.submitJob(...args),
    deleteRequest: (...args: unknown[]) => world.deleteRequest(...args),
  },
}))
vi.mock('../toast', () => ({ notify: world.notify }))
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))
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
    expect.any(String),
  )
  await saved(4)
  expect(world.ledger.current.items[0].accepted?.job).toBe('j1')
  expect(whys()).toEqual(['accept j1 for cast'])
  undo()
  await saved(5)
  expect(world.ledger.current.items[0].accepted).toBeNull()
  await waitFor(() => expect(first.getAttribute('aria-pressed')).toBe('false'))
})

it("an accept's Undo pressed twice while its toast fades takes it back once", async () => {
  await start()
  await fireEvent.click(
    within(row('cast')).getByRole('button', {
      name: 'Accept take 1 for Cast sheet',
    }),
  )
  await saved(4)
  const push = vi.spyOn(PlanStore.prototype, 'push')
  undo()
  undo()
  await saved(5)
  // the second press found the toast used: one write back, not two
  expect(push).toHaveBeenCalledTimes(1)
  push.mockRestore()
  await new Promise((r) => setTimeout(r, 20))
  expect(whys()).toEqual([
    'accept j1 for cast',
    'clear the accepted take of cast',
  ])
  expect(world.ledger.current.revision).toBe(5)
})

it("a drop's Undo pressed once another request is open does nothing to it", async () => {
  const view = render(RequestPage, { requestId: 'r1' })
  await screen.findByRole('heading', { name: 'Launch posters' })
  await choose('s3', 'Drop from plan')
  await saved(4)
  // the page goes on to another request that has an item of the same key
  world.ledger = fakeLedger(
    request(
      [item('s3', { title: 'Other shot' }), item('x', { title: 'Extra' })],
      { id: 'r2', title: 'Other request' },
    ),
  )
  await view.rerender({ requestId: 'r2' })
  await screen.findByRole('heading', { name: 'Other request' })
  undo()
  await waitFor(() =>
    expect(world.notify.error).toHaveBeenCalledWith(
      'Nothing undone: another request is open now.',
    ),
  )
  await new Promise((r) => setTimeout(r, 20))
  expect(world.ledger.puts).toHaveLength(0)
})

it("the user's own accept is not marked as an agent's change", async () => {
  await start()
  await fireEvent.click(
    within(row('cast')).getByRole('button', {
      name: 'Accept take 1 for Cast sheet',
    }),
  )
  await saved(4)
  // the user's own PUT moves the feed; the read after it is the save's
  const reads = world.ledger.gets
  feed.move('studio', 'requests')
  await waitFor(() => expect(world.ledger.gets).toBe(reads + 1))
  await new Promise((r) => setTimeout(r, 20))
  expect(document.querySelector('li.row-changed')).toBeNull()
  // an agent's edit after it still is
  world.ledger.agent('retitled shot 4', (items) =>
    items.map((i) => (i.key === 's4' ? { ...i, title: 'Shot 4, wide' } : i)),
  )
  feed.move('studio', 'requests')
  await waitFor(() => expect(row('s4').classList).toContain('row-changed'))
  expect(document.querySelectorAll('li.row-changed')).toHaveLength(1)
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
    expect.any(String),
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

it('"Add a note" in the menu opens the row and its box', async () => {
  await start()
  await choose('s4', 'Add a note for the agent')
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
  // and the toast offering to undo a drop that did not happen is gone
  const [, , toastId] = world.notify.undo.mock.calls[0]
  await waitFor(() =>
    expect(world.notify.dismiss).toHaveBeenCalledWith(toastId),
  )
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

it('a note on an item the agent removed is not saved, and the box keeps its words', async () => {
  await start()
  await fireEvent.click(
    within(row('s4')).getByRole('button', { name: 'Shot 4' }),
  )
  await fireEvent.click(
    within(row('s4')).getByRole('button', {
      name: 'Add a note for the agent',
    }),
  )
  const box = screen.getByLabelText(
    'Note for the agent on Shot 4',
  ) as HTMLTextAreaElement
  await fireEvent.input(box, { target: { value: 'keep the rain' } })
  world.ledger.agent('merged into shot 3', (items) =>
    items.filter((i) => i.key !== 's4'),
  )
  await fireEvent.click(
    within(row('s4')).getByRole('button', { name: 'Save note' }),
  )
  const alert = await within(row('s4')).findByRole('alert')
  expect(alert.textContent).toContain(
    'This item was dropped or removed from the plan, so the note is not saved.',
  )
  expect(box.value).toBe('keep the rain')
  expect(world.ledger.current.revision).toBe(4)
  // reading again shows the plan as it is
  await fireEvent.click(
    within(alert).getByRole('button', { name: 'Reload the plan' }),
  )
  await waitFor(() => expect(row('s4')).toBeNull())
})

it('a plan write sends no notes, and the notes stored stay', async () => {
  await start()
  await fireEvent.click(
    within(row('s4')).getByRole('button', { name: 'Shot 4' }),
  )
  await fireEvent.click(
    within(row('s4')).getByRole('button', {
      name: 'Add a note for the agent',
    }),
  )
  await fireEvent.input(screen.getByLabelText('Note for the agent on Shot 4'), {
    target: { value: 'keep the rain' },
  })
  await fireEvent.click(
    within(row('s4')).getByRole('button', { name: 'Save note' }),
  )
  await within(row('s4')).findByText('keep the rain')
  expect(world.ledger.added).toEqual([
    { request: 'r1', item: 's4', text: 'keep the rain' },
  ])
  expect(world.ledger.puts).toEqual([])
  await fireEvent.click(
    within(row('cast')).getByRole('button', {
      name: 'Accept take 1 for Cast sheet',
    }),
  )
  await saved(5)
  expect(world.ledger.puts[0].items.some((i) => 'notes' in i)).toBe(false)
  expect(
    world.ledger.current.items.find((i) => i.key === 's4')?.notes,
  ).toMatchObject([{ id: 'n1', text: 'keep the rain' }])
  expect(within(row('s4')).getByText('keep the rain')).toBeTruthy()
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
    expect(world.notify.failure).toHaveBeenCalledWith(
      'save that',
      expect.objectContaining({ message: 'the ledger is busy' }),
    ),
  )
  expect(first.getAttribute('aria-pressed')).toBe('false')
})

// For the specs that spend: a ready item, one that waits on it, and a done
// one that could take another run
const runnable = (): LedgerRequest =>
  request(
    [
      doneItem('cast', { title: 'Cast sheet', workflow: 'sheet' }),
      item('s4', {
        title: 'Shot 4',
        ready: true,
        workflow: 'shot',
        arguments: { seed: 4 },
      }),
      item('cut', { title: 'The cut', depends_on: ['s4'] }),
    ],
    { history: [entry(3, 'agent', 'opened')] },
  )
const priced = () => {
  world.validate.mockResolvedValue({
    valid: true,
    error: null,
    errors: [],
    warnings: [],
    plan: {
      fingerprint: 'fp',
      steps: 1,
      list_entries: {},
      cached_steps: 0,
      elided_steps: [],
      downloads_required: [],
      estimate: {
        minutes: 6,
        basis: 'observed',
        device: 'cuda',
        measured_on: 'cuda',
        partial: false,
      },
    },
  })
  world.submitJob.mockResolvedValue({ id: 'job-new' })
}

it('the one filled button runs what is ready, through the priced dialog', async () => {
  world.ledger = fakeLedger(runnable())
  priced()
  await start()
  const run = screen.getByRole('button', { name: 'Run 1 ready…' })
  // the page's only filled button
  expect(run.className).toBe('')
  await fireEvent.click(run)
  const dialog = await screen.findByRole('dialog', {
    name: 'Run what is ready',
  })
  await within(dialog).findByText(/about 6 min on cuda/)
  // opening it spent nothing
  expect(world.submitJob).not.toHaveBeenCalled()
  expect(dialog.querySelector('.wont li')?.textContent).toBe(
    'The cut: waits on Shot 4 (ready)',
  )
  const reads = world.ledger.gets
  await fireEvent.click(within(dialog).getByRole('button', { name: 'Run 1' }))
  await waitFor(() =>
    expect(world.notify.success).toHaveBeenCalledWith('Queued 1 job'),
  )
  expect(world.submitJob).toHaveBeenCalledWith(
    expect.objectContaining({
      workflow_path: 'shot',
      arguments: { seed: 4 },
      request: 'r1',
      item: 's4',
      acknowledged_cost: expect.objectContaining({ fingerprint: 'fp' }),
    }),
  )
  // and the page reads the request again at once, not at the next refresh
  await waitFor(() => expect(world.ledger.gets).toBe(reads + 1))
  // nothing on the page said queued before the server did
  expect(row('s4').querySelector('.chip')?.textContent).toBe('ready')
})

it('a ready row has Run…, a done one Another take… on the row and in its menu; a waiting one neither', async () => {
  world.ledger = fakeLedger(runnable())
  priced()
  await start()
  expect(within(row('cut')).queryByRole('button', { name: /^Run/ })).toBeNull()
  await fireEvent.click(within(row('s4')).getByRole('button', { name: 'Run…' }))
  const single = await screen.findByRole('dialog', { name: 'Run Shot 4' })
  expect(within(single).queryByRole('checkbox')).toBeNull()
  await fireEvent.click(within(single).getByRole('button', { name: 'Cancel' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  // in plain view on the row: the next take is the usual step after a note
  await fireEvent.click(
    within(row('cast')).getByRole('button', {
      name: 'Run another take of Cast sheet',
    }),
  )
  expect(
    await screen.findByRole('dialog', {
      name: 'Run another take of Cast sheet',
    }),
  ).toBeTruthy()
  await fireEvent.click(
    within(screen.getByRole('dialog')).getByRole('button', { name: 'Cancel' }),
  )
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  await choose('cast', 'Run another take…')
  const again = await screen.findByRole('dialog', {
    name: 'Run another take of Cast sheet',
  })
  expect(
    within(again).getByText(
      'Nothing is accepted, so the new take becomes the result when it finishes.',
    ),
  ).toBeTruthy()
  expect(world.submitJob).not.toHaveBeenCalled()
})

it('a closed request offers no run at all', async () => {
  const closed = runnable()
  closed.state = 'done'
  closed.items[1].ready = false
  world.ledger = fakeLedger(closed)
  await start()
  expect(screen.queryByRole('button', { name: /ready…$/ })).toBeNull()
  expect(screen.queryByRole('button', { name: 'Run…' })).toBeNull()
})

const headerMenu = async (entryName: string) => {
  await fireEvent.click(
    screen.getByRole('button', { name: 'More actions for this request' }),
  )
  const list = await screen.findByRole('dialog')
  await fireEvent.click(within(list).getByRole('button', { name: entryName }))
}

it('Mark done and Reopen write the state, with why', async () => {
  await start()
  await headerMenu('Mark done')
  await saved(4)
  expect(world.ledger.puts[0]).toMatchObject({
    state: 'done',
    why: 'marked done',
    revision: 3,
  })
  await waitFor(() =>
    expect(document.querySelector('.reqhead .tag')?.textContent).toBe('done'),
  )
  await headerMenu('Reopen')
  await saved(5)
  expect(world.ledger.puts[1]).toMatchObject({ state: 'open', why: 'reopened' })
  await headerMenu('Abandon')
  await saved(6)
  expect(world.ledger.current.state).toBe('abandoned')
})

it('a closed request can be deleted, after a confirm, and the page goes to the list', async () => {
  const closed = make()
  closed.state = 'done'
  world.ledger = fakeLedger(closed)
  world.deleteRequest.mockResolvedValue({ deleted: true })
  await start()
  await headerMenu('Delete…')
  await waitFor(() => expect(world.deleteRequest).toHaveBeenCalled())
  expect(world.confirm.mock.calls[0][0]).toContain(
    'Its jobs and their outputs are kept.',
  )
  expect(world.confirm.mock.calls[0][1]).toMatchObject({
    title: 'Delete "Launch posters"?',
    confirmLabel: 'Delete',
    danger: true,
  })
  expect(world.deleteRequest).toHaveBeenCalledWith('studio', 'r1', 3)
  await waitFor(() => expect(location.hash).toBe('#/ws/studio/requests'))
  expect(world.notify.success).toHaveBeenCalledWith('Deleted Launch posters')
})

it('an open request has no Delete, and a declined confirm deletes nothing', async () => {
  await start()
  await fireEvent.click(
    screen.getByRole('button', { name: 'More actions for this request' }),
  )
  const list = await screen.findByRole('dialog')
  expect(within(list).queryByRole('button', { name: 'Delete…' })).toBeNull()
  cleanup()
  document.body.innerHTML = ''
  const closed = make()
  closed.state = 'abandoned'
  world.ledger = fakeLedger(closed)
  world.confirm.mockResolvedValue(false)
  await start()
  await headerMenu('Delete…')
  await waitFor(() => expect(world.confirm).toHaveBeenCalled())
  expect(world.deleteRequest).not.toHaveBeenCalled()
})

it('a delete refused while a job is in flight says so and stays', async () => {
  const { ApiError } = await import('../api')
  const closed = make()
  closed.state = 'done'
  world.ledger = fakeLedger(closed)
  world.deleteRequest.mockRejectedValue(
    new ApiError('1 linked job is active', 409, 'request_has_active_jobs'),
  )
  location.hash = '#/ws/studio/requests/r1'
  await start()
  await headerMenu('Delete…')
  await waitFor(() =>
    expect(world.notify.error).toHaveBeenCalledWith(
      'Not deleted: One of its jobs is queued or running. Cancel it or let it finish, then delete.',
    ),
  )
  expect(location.hash).toBe('#/ws/studio/requests/r1')
})

// A revised item: its kept take stands, its prompt changed since, and the
// shot after it waits
const revisedPlan = (): LedgerRequest =>
  request(
    [
      revisedItem('a', {
        title: 'Board A',
        workflow: 'board',
        arguments: { seed: 7 },
        result: {
          job: 'job-a',
          run_id: 'run-a',
          output: { name: 'run/a.png', kind: 'image' },
        },
      }),
      item('b', {
        title: 'Shot B',
        depends_on: ['a'],
        workflow: 'shot',
      }),
    ],
    { history: [entry(3, 'agent', 'opened')] },
  )

it('a revised item shows its result, why, and Run take 2 beside it; its dependent waits', async () => {
  world.ledger = fakeLedger(revisedPlan())
  priced()
  await start()
  const a = row('a')
  expect(a.querySelector('.chip')?.textContent).toBe('revised')
  expect(a.querySelector('img')).toBeTruthy()
  expect(
    within(a).getByText('The prompt board-01 changed since take 1.'),
  ).toBeTruthy()
  expect(row('b').querySelector('.chip')?.textContent).toBe('waiting')
  await fireEvent.click(within(a).getByRole('button', { name: 'Run take 2…' }))
  const dialog = await screen.findByRole('dialog', {
    name: 'Run another take of Board A',
  })
  await within(dialog).findByText(/about 6 min on cuda/)
  await fireEvent.click(within(dialog).getByRole('button', { name: 'Run 1' }))
  await waitFor(() =>
    expect(world.submitJob).toHaveBeenCalledWith(
      expect.objectContaining({
        workflow_path: 'board',
        arguments: { seed: 7 },
        request: 'r1',
        item: 'a',
      }),
    ),
  )
})

it('Run what is ready lists a revised item, and Keep this take makes it done', async () => {
  world.ledger = fakeLedger(revisedPlan())
  priced()
  await start()
  await fireEvent.click(screen.getByRole('button', { name: 'Run 1 ready…' }))
  const dialog = await screen.findByRole('dialog', {
    name: 'Run what is ready',
  })
  expect(within(dialog).getByText('Board A')).toBeTruthy()
  await fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  await fireEvent.click(
    within(row('a')).getByRole('button', { name: 'Keep this take' }),
  )
  await saved(4)
  expect(whys()).toEqual(['keep job-a for a'])
  const sent = world.ledger.puts[0].items.find((i) => i.key === 'a')
  expect(sent).toMatchObject({
    accepted: { job: 'job-a', run_id: null },
    workflow: 'board',
    arguments: { seed: 7 },
  })
  await waitFor(() =>
    expect(row('a').querySelector('.chip')?.textContent).toBe('done'),
  )
  expect(
    within(row('a')).queryByRole('button', { name: 'Keep this take' }),
  ).toBeNull()
})
