import { expect, it } from 'vitest'
import shared from '../../../../tests/fixtures/plan_change_cases.json'
import type { LedgerItem } from '../types'
import { entry, item, request } from './fixtures'
import { agentChanges, lineOf, planDiff, snapshotOf } from './planDiff'

const lines = (...args: Parameters<typeof planDiff>) =>
  planDiff(...args).map(lineOf)

it('names added, dropped, restored and renamed items', () => {
  const before = request([
    item('a', { title: 'Shot 1' }),
    item('b', { title: 'Shot 2' }),
    item('c', { title: 'Shot 3', dropped: true, status: 'dropped' }),
  ])
  const after = request([
    item('a', { title: 'Opening shot' }),
    item('b', { title: 'Shot 2', dropped: true, status: 'dropped' }),
    item('c', { title: 'Shot 3' }),
    item('d', { title: 'Shot 4' }),
    item('e', { title: 'Never mind', dropped: true, status: 'dropped' }),
  ])
  expect(lines(before, after)).toEqual([
    'Renamed Shot 1 to Opening shot',
    'Dropped Shot 2',
    'Restored Shot 3',
    'Added Shot 4',
  ])
})

it('names an item the agent removed outright as dropped', () => {
  const before = request([
    item('a', { title: 'Shot 1' }),
    item('b', { title: 'Shot 2' }),
    item('c', { title: 'Shot 3', dropped: true, status: 'dropped' }),
  ])
  // b and c are gone from the read: the ledger deletes an item with no jobs
  const after = request([item('a', { title: 'Shot 1' })])
  expect(lines(before, after)).toEqual(['Dropped Shot 2'])
  expect(lines(snapshotOf(before), after)).toEqual(['Dropped Shot 2'])
})

it('names the arguments that changed', () => {
  const before = request([
    item('a', { title: 'Shot 1', arguments: { seed: 1, prompt: 'a cat' } }),
    item('b', { title: 'Shot 2' }),
  ])
  const after = request([
    item('a', {
      title: 'Shot 1',
      workflow: 'other',
      arguments: { seed: 2, prompt: 'a cat', steps: 4 },
    }),
    item('b', { title: 'Shot 2', depends_on: ['a'] }),
  ])
  expect(lines(before, after)).toEqual([
    'Changed Shot 1: its workflow',
    'Changed Shot 1: its arguments (seed, steps)',
    'Changed Shot 2: what it waits on',
  ])
  // a snapshot diffs the same as the read it was taken of
  expect(lines(snapshotOf(before), after)).toEqual(lines(before, after))
})

it('a long argument compares by fingerprint', () => {
  const long = 'a red fox in the snow, '.repeat(20)
  const snap = snapshotOf(
    request([item('a', { title: 'Shot 1', arguments: { prompt: long } })]),
  )
  expect(snap.items.a.args.prompt).toMatch(/^#[0-9a-f]{8}$/)
  expect(JSON.stringify(snap)).not.toContain('red fox')
  const same = request([
    item('a', { title: 'Shot 1', arguments: { prompt: long } }),
  ])
  expect(lines(snap, same)).toEqual([])
  const other = request([
    item('a', { title: 'Shot 1', arguments: { prompt: long + '!' } }),
  ])
  expect(lines(snap, other)).toEqual(['Changed Shot 1: its arguments (prompt)'])
})

it('nothing changed, no lines', () => {
  const read = request([
    item('a', { title: 'Shot 1', arguments: { seed: 1 } }),
    item('b', { title: 'Shot 2', depends_on: ['a'] }),
  ])
  expect(lines(read, structuredClone(read))).toEqual([])
  expect(lines(snapshotOf(read), { ...read, revision: 9 })).toEqual([])
})

it('one read bringing two agent writes lists them all under the newer', () => {
  const before = request([item('a', { title: 'Shot 1' })])
  const after = request(
    [item('a', { title: 'Opening shot', arguments: { seed: 2 } })],
    {
      revision: 6,
      history: [
        entry(4, 'agent', 'renamed'),
        entry(5, 'agent', 'reseeded'),
        entry(6, 'user', 'from another tab'),
      ],
    },
  )
  const brought = agentChanges(before, after)
  expect(Object.keys(brought ?? {})).toEqual(['5'])
  expect(brought?.[5].map(lineOf)).toEqual([
    'Renamed Shot 1 to Opening shot',
    'Changed Opening shot: its arguments (seed)',
  ])
})

it('lists each change under the writer that made it on a mixed read', () => {
  const before = request(
    [item('a', { title: 'Shot 1' }), item('b', { title: 'Shot 2' })],
    { revision: 1 },
  )
  const after = request(
    [
      item('a', { title: 'Shot 1', arguments: { seed: 2 } }),
      item('b', { title: 'Shot 2', dropped: true, status: 'dropped' }),
      item('c', { title: 'Shot 3' }),
    ],
    {
      revision: 4,
      history: [
        entry(2, 'agent', 'reseeded', { changed: { a: ['arguments'] } }),
        entry(3, 'user', 'dropped b', { changed: { b: ['dropped'] } }),
        entry(4, 'agent', 'added c', { changed: { c: ['added'] } }),
      ],
    },
  )
  const brought = agentChanges(before, after) ?? {}
  expect(
    Object.entries(brought).map(([r, list]) => [r, list.map(lineOf)]),
  ).toEqual([
    ['2', ['Changed Shot 1: its arguments (seed)']],
    ['3', ['Dropped Shot 2']],
    ['4', ['Added Shot 3']],
  ])
})

it('keeps the 50 newest seen-snapshots and drops the oldest', async () => {
  const { readSnapshot, writeSnapshot, SEEN_KEPT } = await import('./planDiff')
  const snap = snapshotOf(request([item('a')]))
  for (let i = 0; i < SEEN_KEPT + 3; i++) writeSnapshot('ws', `r${i}`, snap)
  expect(readSnapshot('ws', 'r0')).toBeNull()
  expect(readSnapshot('ws', 'r2')).toBeNull()
  expect(readSnapshot('ws', 'r3')).not.toBeNull()
  expect(readSnapshot('ws', `r${SEEN_KEPT + 2}`)).not.toBeNull()
})

// The engine owns this rule (item_changes); the same cases run there
type SharedItem = Partial<LedgerItem> & { key: string }
const stored = (items: unknown) =>
  request(
    (items as SharedItem[]).map((i) => item(i.key, { title: i.key, ...i })),
  )

it.each(shared.cases)('agrees with the engine: $name', (c) => {
  const found: Record<string, string[]> = {}
  for (const change of planDiff(stored(c.before), stored(c.after)))
    (found[change.key] ??= []).push(change.what ?? change.kind)
  for (const what of Object.values(found)) what.sort()
  expect(found).toEqual(c.changes)
})
