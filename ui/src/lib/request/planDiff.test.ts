import { expect, it } from 'vitest'
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
  expect(brought?.revision).toBe(5)
  expect(brought?.changes.map(lineOf)).toEqual([
    'Renamed Shot 1 to Opening shot',
    'Changed Opening shot: its arguments (seed)',
  ])
})
