import { expect, it, vi } from 'vitest'
import { ApiError } from '../apiErrors'
import type { LedgerRequest } from '../types'
import { fakeLedger } from './fakeLedger'
import { attempt, doneItem, item, request } from './fixtures'
import { MAX_RETRIES, PlanStore } from './planStore.svelte'
import { quiet, useQuietClock } from '../test/quiet'

useQuietClock()

// a done with two takes; b ready after a; c waits on b; d has never run
const plan = (): LedgerRequest =>
  request([
    doneItem('a', { attempts: [attempt('j1'), attempt('j2')] }),
    item('b', {
      depends_on: ['a'],
      ready: true,
      attempts: [attempt('jb', 'failed')],
    }),
    item('c', { depends_on: ['b'] }),
    item('d', { ready: true }),
  ])

function setup(initial = plan()) {
  const ledger = fakeLedger(initial)
  const reads: number[] = []
  const store = new PlanStore({
    get: () => ledger.get(),
    put: (body) => ledger.put(body),
    onread: (_previous, next) => reads.push(next.revision),
  })
  store.take(ledger.current)
  return { ledger, store, reads }
}
const live = (store: PlanStore) =>
  store.view?.items.filter((i) => !i.dropped).map((i) => i.key)
const settled = () => new Promise((r) => setTimeout(r, 0))

it('shows an edit at once and writes it as one intent at the read revision', async () => {
  const { ledger, store } = setup()
  const release = ledger.hold()
  const saving = store.push({ kind: 'accept', key: 'a', job: 'j1' })
  // optimistic: the view has it, the read does not
  expect(store.view?.items[0].accepted?.job).toBe('j1')
  expect(store.read?.items[0].accepted).toBeNull()
  release()
  expect(await saving).toEqual({ status: 'saved', stripped: [] })
  expect(ledger.puts).toHaveLength(1)
  expect(ledger.puts[0]).toMatchObject({ revision: 3, why: 'accept j1 for a' })
  expect(store.read?.revision).toBe(4)
  expect(store.queue).toHaveLength(0)
  expect(store.notice).toBeNull()
})

it('a poll during a pending write replaces the read without undoing the edit', async () => {
  const { ledger, store } = setup()
  const release = ledger.hold()
  const saving = store.push({ kind: 'drop', key: 'b' })
  expect(live(store)).toEqual(['a', 'c', 'd'])
  // a refresh lands with the same revision and b now running
  const polled = structuredClone(ledger.current)
  polled.items[3].title = 'D, as the poll has it'
  store.take(polled)
  expect(store.read).toBe(polled)
  expect(live(store)).toEqual(['a', 'c', 'd'])
  expect(store.view?.items.find((i) => i.key === 'd')?.title).toBe(
    'D, as the poll has it',
  )
  release()
  expect((await saving).status).toBe('saved')
  expect(live(store)).toEqual(['a', 'c', 'd'])
})

it('a poll that already carries the write does not show the edit twice', async () => {
  const { ledger, store } = setup()
  const release = ledger.hold()
  const saving = store.push({ kind: 'move', key: 'd', before: 'a' })
  // the server took it; a refresh sees it before the write's own reply
  release()
  await settled()
  store.take(ledger.current)
  expect(live(store)).toEqual(['d', 'a', 'b', 'c'])
  await saving
  expect(live(store)).toEqual(['d', 'a', 'b', 'c'])
})

it('a refresh older than what a write returned does not put the old plan back', async () => {
  const { ledger, store } = setup()
  const old = ledger.current
  await store.push({ kind: 'accept', key: 'a', job: 'j1' })
  store.take(old)
  expect(store.read?.revision).toBe(4)
  expect(store.view?.items[0].accepted?.job).toBe('j1')
})

it('an overtaken refresh is not taken', async () => {
  const { ledger, store } = setup()
  const next = { ...ledger.current, revision: ledger.current.revision + 1 }
  ledger.current = next
  let current = true
  const refreshing = store.refresh(() => current)
  // a later read overtook this one before it landed
  current = false
  await refreshing
  expect(store.read?.revision).toBe(next.revision - 1)
  // the one that is current is taken
  current = true
  await store.refresh(() => current)
  expect(store.read?.revision).toBe(next.revision)
})

it('writes one at a time, each over what the one before it returned', async () => {
  const { ledger, store } = setup()
  const first = store.push({ kind: 'accept', key: 'a', job: 'j1' })
  const second = store.push({ kind: 'move', key: 'd', before: 'a' })
  expect(ledger.puts).toHaveLength(1)
  await Promise.all([first, second])
  expect(ledger.puts.map((p) => [p.revision, p.why])).toEqual([
    [3, 'accept j1 for a'],
    [4, 'reorder d'],
  ])
  // the second write carries the first: it was built from the reply
  expect(ledger.puts[1].items.find((i) => i.key === 'a')?.accepted?.job).toBe(
    'j1',
  )
  expect(ledger.gets).toBe(0)
})

it('the agent writing between the read and the write: a move is made again on top, and says so', async () => {
  const { ledger, store } = setup()
  ledger.agent('added shot 7', (items) => [...items, item('e')])
  const outcome = await store.push({ kind: 'move', key: 'd', before: 'a' })
  expect(outcome.status).toBe('saved')
  // the stale plan was sent once (refused), then never again
  expect(ledger.puts.map((p) => p.revision)).toEqual([3, 4])
  expect(ledger.puts[1].items.map((i) => i.key)).toEqual([
    'd',
    'a',
    'b',
    'c',
    'e',
  ])
  expect(ledger.current.items.map((i) => i.key)).toEqual([
    'd',
    'a',
    'b',
    'c',
    'e',
  ])
  expect(store.notice?.entry).toMatchObject({
    revision: 4,
    by: 'agent',
    why: 'added shot 7',
  })
  expect(store.conflicts).toHaveLength(0)
})

it('a move is made again on top of an agent write, whatever happened to the item', async () => {
  const { ledger, store } = setup()
  ledger.agent('retitled', (items) =>
    items.map((i) => (i.key === 'b' ? { ...i, title: 'New name' } : i)),
  )
  const outcome = await store.push({ kind: 'move', key: 'b', before: null })
  expect(outcome.status).toBe('saved')
  expect(ledger.current.items.filter((i) => !i.dropped).at(-1)).toMatchObject({
    key: 'b',
    title: 'New name',
  })
})

it('a plan write keeps the notes stored, whatever the page held', async () => {
  const { ledger, store } = setup()
  const note = { id: 'n1', text: 'warmer', created: 't', at: 't' }
  await ledger.addNote({ request: 'r1', item: 'b', text: 'warmer' })
  ledger.current.items[1].notes = [note]
  await store.push({ kind: 'move', key: 'b', before: 'a' })
  expect(ledger.puts.at(-1)?.items.some((i) => 'notes' in i)).toBe(false)
  expect(ledger.current.items.find((i) => i.key === 'b')?.notes).toEqual([note])
})

it('a drop of an item the agent changed is not written: the page asks', async () => {
  const { ledger, store } = setup()
  ledger.agent('retake with warmer light', (items) =>
    items.map((i) =>
      i.key === 'b' ? { ...i, attempts: [...i.attempts, attempt('jb2')] } : i,
    ),
  )
  const outcome = await store.drop('b')
  expect(outcome.status).toBe('asked')
  expect(ledger.puts).toHaveLength(1) // the refused one only
  expect(ledger.current.items.find((i) => i.key === 'b')?.dropped).toBe(false)
  // the optimistic drop is gone from the view: the row is back, with a card
  expect(live(store)).toEqual(['a', 'b', 'c', 'd'])
  expect(store.conflicts).toHaveLength(1)
  expect(store.conflicts[0]).toMatchObject({
    kind: 'changed',
    key: 'b',
    title: 'Item b',
    changes: ['1 new take'],
    entry: { revision: 4, why: 'retake with warmer light' },
  })
  expect(store.notice).toBeNull()
  expect(store.memory).toEqual({})
})

it('"anyway" makes the held-back edit over the item as it now is', async () => {
  const { ledger, store } = setup()
  ledger.agent('retake', (items) =>
    items.map((i) =>
      i.key === 'b' ? { ...i, attempts: [...i.attempts, attempt('jb2')] } : i,
    ),
  )
  await store.drop('b')
  const outcome = await store.again(store.conflicts[0].id)
  expect(outcome.status).toBe('saved')
  expect(store.conflicts).toHaveLength(0)
  const b = ledger.current.items.find((i) => i.key === 'b')
  expect(b?.dropped).toBe(true)
  // the new take went with it, and the dependent lost its edge
  expect(b?.attempts).toHaveLength(2)
  expect(ledger.current.items.find((i) => i.key === 'c')?.depends_on).toEqual(
    [],
  )
  expect(ledger.puts.at(-1)?.revision).toBe(4)
})

it('"keep it" writes nothing', async () => {
  const { ledger, store } = setup()
  ledger.agent('retitled', (items) =>
    items.map((i) => (i.key === 'b' ? { ...i, title: 'Other' } : i)),
  )
  await store.drop('b')
  store.dismiss(store.conflicts[0].id)
  await quiet()
  expect(store.conflicts).toHaveLength(0)
  expect(ledger.puts).toHaveLength(1)
  expect(ledger.current.revision).toBe(4)
})

it('a refresh that changes the item before the write is sent asks too, before it is sent', async () => {
  const { ledger, store } = setup()
  const release = ledger.hold()
  const first = store.push({ kind: 'move', key: 'd', before: 'a' })
  const dropping = store.drop('b')
  // while the move is out, the agent adds a take to b and a poll sees it
  ledger.agent('retake', (items) =>
    items.map((i) =>
      i.key === 'b' ? { ...i, attempts: [...i.attempts, attempt('jb2')] } : i,
    ),
  )
  store.take(ledger.current)
  release()
  await first
  expect((await dropping).status).toBe('asked')
  // the drop was never sent, at any revision
  expect(ledger.puts.map((p) => p.why)).toEqual(['reorder d', 'reorder d'])
})

it('an accept goes on top when the agent changed a different item', async () => {
  const { ledger, store } = setup()
  ledger.agent('reworded d', (items) =>
    items.map((i) => (i.key === 'd' ? { ...i, title: 'D2' } : i)),
  )
  const outcome = await store.push({ kind: 'accept', key: 'a', job: 'j2' })
  expect(outcome.status).toBe('saved')
  expect(ledger.current.items[0].accepted?.job).toBe('j2')
  expect(ledger.current.items.find((i) => i.key === 'd')?.title).toBe('D2')
  expect(store.notice?.entry?.why).toBe('reworded d')
})

it('an edit that no longer applies is not written, and says which and why', async () => {
  const { ledger, store } = setup()
  ledger.agent('merged into shot 2', (items) =>
    items.filter((i) => i.key !== 'd'),
  )
  const outcome = await store.push({ kind: 'move', key: 'd', before: 'a' })
  expect(outcome.status).toBe('gone')
  expect(ledger.puts).toHaveLength(1)
  expect(store.conflicts[0]).toMatchObject({
    kind: 'gone',
    refusal: 'item_gone',
    key: 'd',
    title: 'Item d',
    intent: { kind: 'move', key: 'd' },
    entry: { revision: 4, why: 'merged into shot 2' },
  })
  // an accept of a take the agent's rewrite no longer lists
  ledger.agent('cleared takes', (items) =>
    items.map((i) => (i.key === 'a' ? { ...i, attempts: [attempt('j2')] } : i)),
  )
  expect(
    (await store.push({ kind: 'accept', key: 'a', job: 'j1' })).status,
  ).toBe('gone')
  expect(store.conflicts[1].refusal).toBe('attempt_gone')
})

it('nothing is sent for an edit the plan already holds', async () => {
  const { ledger, store } = setup()
  const outcome = await store.push({ kind: 'move', key: 'a', before: 'b' })
  expect(outcome.status).toBe('unchanged')
  expect(ledger.puts).toHaveLength(0)
})

it('of two queued edits, the first failing does not stop or stale the second', async () => {
  const { ledger, store } = setup()
  const put = vi.spyOn(ledger, 'put')
  put.mockRejectedValueOnce(new ApiError('ledger is busy', 503, 'ledger_busy'))
  const first = store.push({ kind: 'accept', key: 'a', job: 'j1' })
  const second = store.push({ kind: 'move', key: 'd', before: 'a' })
  await expect(first).rejects.toMatchObject({ code: 'ledger_busy' })
  expect((await second).status).toBe('saved')
  // the second was built from the read, without the edit that failed
  expect(ledger.current.items.map((i) => i.key)).toEqual(['d', 'a', 'b', 'c'])
  expect(ledger.current.items[1].accepted).toBeNull()
  expect(store.view?.items[1].accepted).toBeNull()
  expect(store.queue).toHaveLength(0)
})

it('gives up after three more tries, and never sends one revision twice', async () => {
  const { ledger, store } = setup()
  // the agent writes before every one of the page's writes
  const put = ledger.put.bind(ledger)
  let n = 0
  vi.spyOn(ledger, 'put').mockImplementation((body) => {
    ledger.agent(`write ${++n}`, (items) => items)
    return put(body)
  })
  const saving = store.push({ kind: 'move', key: 'd', before: 'a' })
  await expect(saving).rejects.toMatchObject({ code: 'stale_revision' })
  expect(ledger.puts).toHaveLength(1 + MAX_RETRIES)
  const revisions = ledger.puts.map((p) => p.revision)
  expect(new Set(revisions).size).toBe(revisions.length)
  expect(store.queue).toHaveLength(0)
  // the move is not shown as made
  expect(live(store)).toEqual(['a', 'b', 'c', 'd'])
})

it('a read that fails after a stale write ends the edit with that failure', async () => {
  const { ledger, store } = setup()
  ledger.agent('moved on', (items) => items)
  vi.spyOn(ledger, 'get').mockRejectedValueOnce(new Error('offline'))
  await expect(
    store.push({ kind: 'move', key: 'd', before: 'a' }),
  ).rejects.toThrow('offline')
  expect(ledger.puts).toHaveLength(1)
  expect(store.queue).toHaveLength(0)
})

it('a run of staged moves is one write, and a move back to the start is none', async () => {
  const { ledger, store } = setup()
  void store.stage({ kind: 'move', key: 'd', before: 'c' })
  void store.stage({ kind: 'move', key: 'd', before: 'b' })
  const last = store.stage({ kind: 'move', key: 'd', before: 'a' })
  expect(live(store)).toEqual(['d', 'a', 'b', 'c'])
  expect(store.queue).toHaveLength(1)
  expect(ledger.puts).toHaveLength(0)
  store.commit()
  expect((await last).status).toBe('saved')
  expect(ledger.puts).toHaveLength(1)
  expect(ledger.puts[0].items.map((i) => i.key)).toEqual(['d', 'a', 'b', 'c'])

  void store.stage({ kind: 'move', key: 'd', before: 'b' })
  const back = store.stage({ kind: 'move', key: 'd', before: 'a' })
  store.commit()
  expect((await back).status).toBe('unchanged')
  expect(ledger.puts).toHaveLength(1)
})

it('another edit sends the staged move first', async () => {
  const { ledger, store } = setup()
  void store.stage({ kind: 'move', key: 'd', before: 'a' })
  await store.push({ kind: 'accept', key: 'a', job: 'j1' })
  expect(ledger.puts.map((p) => p.why)).toEqual([
    'reorder d',
    'accept j1 for a',
  ])
})

it('undoing a drop restores the item, its place and what depended on it', async () => {
  const { ledger, store } = setup()
  await store.drop('b')
  expect(ledger.current.items.map((i) => [i.key, i.dropped])).toEqual([
    ['a', false],
    ['c', false],
    ['d', false],
    ['b', true],
  ])
  expect(ledger.current.items[1].depends_on).toEqual([])
  const outcome = await store.restore('b')
  expect(outcome).toEqual({ status: 'saved', stripped: [] })
  expect(ledger.current.items.map((i) => i.key)).toEqual(['a', 'b', 'c', 'd'])
  expect(ledger.current.items[2].depends_on).toEqual(['b'])
  expect(ledger.puts.map((p) => p.why)).toEqual(['drop b', 'restore b'])
  expect(store.memory).toEqual({})
})

it('an item the server deleted after a drop stays restorable until the page forgets', async () => {
  const { ledger, store } = setup()
  // d has no job linked: the server deletes it outright
  await store.drop('d')
  expect(ledger.current.items.map((i) => i.key)).toEqual(['a', 'b', 'c'])
  expect(store.view?.items.at(-1)).toMatchObject({ key: 'd', dropped: true })
  await store.restore('d')
  expect(ledger.current.items.map((i) => i.key)).toEqual(['a', 'b', 'c', 'd'])
  expect(store.view?.items.filter((i) => i.key === 'd')).toHaveLength(1)
})

it('a restore on a reloaded page goes last and reports the dependencies left behind', async () => {
  const start = plan()
  start.items.push(
    item('x', {
      dropped: true,
      status: 'dropped',
      depends_on: ['a', 'zz'],
      attempts: [attempt('jx', 'failed')],
    }),
  )
  const { ledger, store } = setup(start)
  const outcome = await store.restore('x')
  expect(outcome).toEqual({ status: 'saved', stripped: ['zz'] })
  expect(ledger.current.items.map((i) => i.key)).toEqual([
    'a',
    'b',
    'c',
    'd',
    'x',
  ])
  expect(ledger.current.items[4].depends_on).toEqual(['a'])
})

it('a reply for a request the page has left changes nothing', async () => {
  const { ledger, store } = setup()
  const release = ledger.hold()
  const saving = store.push({ kind: 'accept', key: 'a', job: 'j1' })
  store.reset()
  expect((await saving).status).toBe('unchanged')
  const other = request([item('z')], { id: 'r2', revision: 1 })
  store.take(other)
  release()
  await quiet()
  expect(store.read).toBe(other)
  expect(store.queue).toHaveLength(0)
})

it('tells the page of each read it takes, with the one it replaced', async () => {
  const { store, reads } = setup()
  await store.push({ kind: 'accept', key: 'a', job: 'j1' })
  expect(reads).toEqual([3, 4])
})

it("retries past the user's own note write without calling it a change", async () => {
  const { ledger, store } = setup()
  ledger.elsewhere('note added')
  const outcome = await store.push({ kind: 'accept', key: 'a', job: 'j1' })
  expect(outcome.status).toBe('saved')
  expect(ledger.puts).toHaveLength(2)
  expect(store.notice).toBeNull()
})

it("names the agent's write, or any other, past a note of the user's", async () => {
  const { ledger, store } = setup()
  ledger.elsewhere('retitled elsewhere')
  ledger.elsewhere('note added')
  await store.push({ kind: 'accept', key: 'a', job: 'j1' })
  expect(store.notice?.entry?.why).toBe('retitled elsewhere')
})

/** The server takes the write and the reply never arrives */
const loseReply = (ledger: ReturnType<typeof fakeLedger>) => {
  const put = ledger.put.bind(ledger)
  vi.spyOn(ledger, 'put').mockImplementationOnce(async (body) => {
    await put(body)
    throw new TypeError('Failed to fetch')
  })
}

it('can restore a drop that was written though its reply was lost', async () => {
  const { ledger, store } = setup()
  loseReply(ledger)
  await expect(store.drop('d')).rejects.toThrow('Failed to fetch')
  store.take(ledger.current)
  expect(ledger.current.items.map((i) => i.key)).not.toContain('d')
  expect(store.memory.d).toBeDefined()
  expect((await store.restore('d')).status).toBe('saved')
  expect(live(store)).toContain('d')
})

it('forgets a failed drop once a read shows the item live', async () => {
  const { ledger, store } = setup()
  vi.spyOn(ledger, 'put').mockRejectedValueOnce(new TypeError('offline'))
  await expect(store.drop('d')).rejects.toThrow('offline')
  expect(store.memory.d).toBeDefined()
  store.take(structuredClone(ledger.current))
  expect(store.memory.d).toBeUndefined()
})

it('forgets a drop the server refused', async () => {
  const { ledger, store } = setup()
  vi.spyOn(ledger, 'put').mockRejectedValueOnce(
    new ApiError('no', 422, 'invalid'),
  )
  await expect(store.drop('d')).rejects.toThrow('no')
  expect(store.memory.d).toBeUndefined()
})

it('forgets a drop once the agent has put the item back, keeping its edges as the agent has them', async () => {
  const { ledger, store } = setup()
  await store.drop('a')
  expect(store.memory.a).toBeDefined()
  // back, with b no longer depending on it
  ledger.agent('a is back', (items) => [
    doneItem('a'),
    ...items
      .filter((i) => i.key !== 'a')
      .map((i) => (i.key === 'b' ? { ...i, depends_on: [] } : i)),
  ])
  store.take(ledger.current)
  expect(store.memory.a).toBeUndefined()
})

it('writes what is queued and staged after the page has left, telling the page nothing', async () => {
  const { ledger, store, reads } = setup()
  const release = ledger.hold()
  const first = store.push({ kind: 'accept', key: 'a', job: 'j1' })
  const staged = store.stage({ kind: 'move', key: 'd', before: 'a' })
  store.leave()
  release()
  expect((await first).status).toBe('saved')
  expect((await staged).status).toBe('saved')
  expect(ledger.puts.length).toBeGreaterThanOrEqual(2)
  expect(ledger.current.items.map((i) => i.key)).toEqual(['d', 'a', 'b', 'c'])
  expect(reads).toEqual([3])
})

it('says a conflict met after the page left, which no card will show', async () => {
  const ledger = fakeLedger(plan())
  const onunshown = vi.fn()
  const store = new PlanStore({
    get: () => ledger.get(),
    put: (body) => ledger.put(body),
    onunshown,
  })
  store.take(ledger.current)
  ledger.agent('cut d', (items) => items.filter((i) => i.key !== 'd'))
  const saving = store.push({ kind: 'drop', key: 'd' })
  store.leave()
  await saving
  expect(onunshown).toHaveBeenCalledTimes(1)
})
