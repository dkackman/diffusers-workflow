import { expect, it } from 'vitest'
import { attempt, doneItem, item, request } from './fixtures'
import {
  apply,
  fold,
  isRefused,
  project,
  withGhosts,
  type Intent,
} from './intents'
import type { LedgerRequest } from '../types'

// a done with two takes; b ready after a; c waits on b; d free; x dropped
const plan = (): LedgerRequest =>
  request([
    doneItem('a', {
      attempts: [attempt('j1'), attempt('j2'), attempt('j3', 'failed')],
      result: { job: 'j2', run_id: 'run-2', output: null },
    }),
    item('b', { depends_on: ['a'], ready: true }),
    item('c', { depends_on: ['b', 'a'] }),
    item('d', { ready: true }),
    item('x', { dropped: true, status: 'dropped', depends_on: ['b', 'zz'] }),
  ])

const body = (intent: Intent, over = plan()) => {
  const applied = apply(intent, over)
  if (isRefused(applied)) throw new Error(`refused: ${applied.refusal}`)
  return applied.body
}
const refusal = (intent: Intent, over = plan()) => {
  const applied = apply(intent, over)
  return isRefused(applied) ? applied.refusal : null
}
const keys = (r: { items: { key: string; dropped: boolean }[] }) =>
  r.items.filter((i) => !i.dropped).map((i) => i.key)

it('writes the whole plan back at the revision read, and nothing the server refuses', () => {
  const sent = body({ kind: 'accept', key: 'a', job: 'j1' })
  expect(Object.keys(sent).sort()).toEqual([
    'goal',
    'items',
    'revision',
    'state',
    'title',
    'why',
  ])
  expect(sent.revision).toBe(3)
  expect(sent.items.map((i) => i.key)).toEqual(['a', 'b', 'c', 'd', 'x'])
  expect(sent.why.length).toBeLessThanOrEqual(200)
})

it('accepts a succeeded take, and shows it as the result before the server answers', () => {
  const sent = body({ kind: 'accept', key: 'a', job: 'j1' })
  expect(sent.why).toBe('accept j1 for a')
  expect(sent.items[0].accepted).toEqual({ job: 'j1', run_id: null })
  expect(sent.items[0].result?.job).toBe('j1')
  // only that item changed
  expect(sent.items.slice(1)).toEqual(plan().items.slice(1))
})

it('clears an accepted take back to the latest', () => {
  const held = plan()
  held.items[0].accepted = { job: 'j1', run_id: 'run-1' }
  held.items[0].result = { job: 'j1', run_id: 'run-1', output: null }
  const sent = body({ kind: 'accept', key: 'a', job: null }, held)
  expect(sent.why).toBe('clear the accepted take of a')
  expect(sent.items[0].accepted).toBeNull()
  expect(sent.items[0].result?.job).toBe('j2')
})

it('refuses an accept by name when it no longer means anything', () => {
  expect(refusal({ kind: 'accept', key: 'gone', job: 'j1' })).toBe('item_gone')
  expect(refusal({ kind: 'accept', key: 'x', job: 'j1' })).toBe('item_dropped')
  expect(refusal({ kind: 'accept', key: 'a', job: 'j9' })).toBe('attempt_gone')
  // a take that did not succeed cannot be the result
  expect(refusal({ kind: 'accept', key: 'a', job: 'j3' })).toBe('attempt_gone')
  expect(refusal({ kind: 'accept', key: 'a', job: null })).toBe('no_change')
  const held = plan()
  held.items[0].accepted = { job: 'j1', run_id: null }
  expect(refusal({ kind: 'accept', key: 'a', job: 'j1' }, held)).toBe(
    'no_change',
  )
})

it('a drop takes the key out of every live dependent in the same write', () => {
  const sent = body({ kind: 'drop', key: 'b' })
  expect(sent.why).toBe('drop b')
  const by = Object.fromEntries(sent.items.map((i) => [i.key, i]))
  expect(by.b.dropped).toBe(true)
  expect(by.c.depends_on).toEqual(['a'])
  // a dropped item keeps the dependencies it was stored with
  expect(by.x.depends_on).toEqual(['b', 'zz'])
  // dropped items sit after the live ones
  expect(sent.items.map((i) => i.key)).toEqual(['a', 'c', 'd', 'b', 'x'])
  // and the dependent, its last unmet dependency gone, reads as ready
  expect(by.c.ready).toBe(true)
})

it('refuses a drop of what is gone, and has nothing to do for one already dropped', () => {
  expect(refusal({ kind: 'drop', key: 'gone' })).toBe('item_gone')
  expect(refusal({ kind: 'drop', key: 'x' })).toBe('no_change')
})

it('undoing a drop puts the item, its place and its edges back', () => {
  const before = plan()
  const dropped = project({ kind: 'drop', key: 'b' }, before)
  if (isRefused(dropped)) throw new Error('refused')
  const sent = body(
    {
      kind: 'restore',
      key: 'b',
      memory: { item: before.items[1], before: 'c', dependents: ['c'] },
    },
    dropped.request,
  )
  expect(sent.why).toBe('restore b')
  expect(sent.items.map((i) => i.key)).toEqual(['a', 'b', 'c', 'd', 'x'])
  const by = Object.fromEntries(sent.items.map((i) => [i.key, i]))
  expect(by.b.dropped).toBe(false)
  expect(by.b.depends_on).toEqual(['a'])
  expect(by.c.depends_on).toEqual(['a', 'b'])
  expect(by.c.ready).toBe(false)
})

it('restores an item the server deleted outright from what the page remembers', () => {
  const before = plan()
  const without = {
    ...before,
    items: before.items.filter((i) => i.key !== 'd'),
  }
  expect(refusal({ kind: 'restore', key: 'd' }, without)).toBe('item_gone')
  const sent = body(
    {
      kind: 'restore',
      key: 'd',
      memory: { item: before.items[3], before: null, dependents: [] },
    },
    without,
  )
  expect(keys(sent)).toEqual(['a', 'b', 'c', 'd'])
})

it('a restore with nothing remembered goes last and leaves dead dependencies behind', () => {
  const applied = apply({ kind: 'restore', key: 'x' }, plan())
  if (isRefused(applied)) throw new Error('refused')
  expect(keys(applied.body)).toEqual(['a', 'b', 'c', 'd', 'x'])
  expect(applied.body.items[4].depends_on).toEqual(['b'])
  expect(applied.stripped).toEqual(['zz'])
  expect(refusal({ kind: 'restore', key: 'a' })).toBe('no_change')
})

it('a restore does not put back an edge that would now close a cycle', () => {
  // b was dropped while c depended on it; the agent has since made b's
  // stored self depend on c
  const held = request([
    item('c'),
    item('b', { dropped: true, status: 'dropped', depends_on: ['c'] }),
  ])
  const sent = body(
    {
      kind: 'restore',
      key: 'b',
      memory: { item: held.items[1], before: null, dependents: ['c'] },
    },
    held,
  )
  expect(sent.items.find((i) => i.key === 'c')?.depends_on).toEqual([])
  expect(sent.items.find((i) => i.key === 'b')?.depends_on).toEqual(['c'])
})

it('a move is relative to an anchor, so it survives items the agent added', () => {
  const sent = body({ kind: 'move', key: 'd', before: 'b' })
  expect(sent.why).toBe('reorder d')
  expect(sent.items.map((i) => i.key)).toEqual(['a', 'd', 'b', 'c', 'x'])
  const grown = plan()
  grown.items.splice(1, 0, item('new1'))
  grown.items.splice(3, 0, item('new2'))
  expect(keys(body({ kind: 'move', key: 'd', before: 'b' }, grown))).toEqual([
    'a',
    'new1',
    'd',
    'b',
    'new2',
    'c',
  ])
  expect(keys(body({ kind: 'move', key: 'a', before: null }))).toEqual([
    'b',
    'c',
    'd',
    'a',
  ])
})

it('refuses a move by name, and dropped items do not move', () => {
  expect(refusal({ kind: 'move', key: 'gone', before: 'a' })).toBe('item_gone')
  expect(refusal({ kind: 'move', key: 'x', before: 'a' })).toBe('item_dropped')
  expect(refusal({ kind: 'move', key: 'a', before: 'x' })).toBe('anchor_gone')
  expect(refusal({ kind: 'move', key: 'a', before: 'zz' })).toBe('anchor_gone')
  expect(refusal({ kind: 'move', key: 'a', before: 'b' })).toBe('no_change')
  expect(refusal({ kind: 'move', key: 'd', before: null })).toBe('no_change')
  expect(refusal({ kind: 'move', key: 'a', before: 'a' })).toBe('no_change')
})

it("a note changes one comment: an item's by key, the request's own by the body", () => {
  const sent = body({ kind: 'note', key: 'b', text: 'warmer' })
  expect(sent.why).toBe('comment on b')
  expect(sent.items[1].comment).toEqual({ text: 'warmer' })
  expect('comment' in sent).toBe(false)
  const own = body({ kind: 'note', key: null, text: 'all of it' })
  expect(own.why).toBe('comment on the request')
  expect(own.comment).toEqual({ text: 'all of it' })
  expect(own.items).toEqual(plan().items)
  const held = plan()
  held.comment = { text: 'x', at: 't' }
  expect(body({ kind: 'note', key: null, text: null }, held).comment).toBeNull()
  expect(refusal({ kind: 'note', key: null, text: 'x' }, held)).toBe(
    'no_change',
  )
  expect(refusal({ kind: 'note', key: 'x', text: 'n' })).toBe('item_dropped')
  expect(refusal({ kind: 'note', key: 'gone', text: 'n' })).toBe('item_gone')
})

it('a change of state says why in words the history can show', () => {
  expect(body({ kind: 'state', state: 'done' })).toMatchObject({
    state: 'done',
    why: 'marked done',
  })
  expect(body({ kind: 'state', state: 'abandoned' }).why).toBe('abandoned')
  expect(refusal({ kind: 'state', state: 'open' })).toBe('no_change')
  expect(
    body({ kind: 'state', state: 'open' }, { ...plan(), state: 'done' }).why,
  ).toBe('reopened')
})

it('folds the waiting edits over a read, passing over one that no longer applies', () => {
  const view = fold(plan(), [
    { kind: 'drop', key: 'b' },
    { kind: 'accept', key: 'gone', job: 'j1' },
    { kind: 'move', key: 'd', before: 'a' },
    { kind: 'accept', key: 'a', job: 'j1' },
    // not shown before the server has them
    { kind: 'note', key: 'a', text: 'unsaved' },
    { kind: 'state', state: 'done' },
  ])
  expect(keys(view)).toEqual(['d', 'a', 'c'])
  expect(view.items[1].accepted?.job).toBe('j1')
  expect(view.items[1].comment).toBeNull()
  expect(view.state).toBe('open')
  expect(view.revision).toBe(3)
})

it('shows an item the server deleted after a drop until the page forgets it', () => {
  const before = plan()
  const without = {
    ...before,
    items: before.items.filter((i) => i.key !== 'd'),
  }
  const memory = { d: { item: before.items[3], before: null, dependents: [] } }
  const view = withGhosts(without, memory)
  expect(view.items.at(-1)).toMatchObject({
    key: 'd',
    dropped: true,
    status: 'dropped',
  })
  // still in the plan: nothing to add
  expect(withGhosts(before, memory)).toBe(before)
})
