import { expect, it } from 'vitest'
import { dropWarnings, restoredText } from './dropConfirm'
import { attempt, doneItem, item, request } from './fixtures'
import { planState } from './itemState'

const rows = () =>
  planState(
    request([
      doneItem('a', { title: 'Cast sheet' }),
      item('b', {
        title: 'Shot 3',
        depends_on: ['a'],
        status: 'failed',
        attempts: [attempt('jb', 'failed')],
      }),
      item('c', { title: 'Shot 4', depends_on: ['b'] }),
      item('d', { title: 'The cut', depends_on: ['b', 'c'] }),
      item('e', {
        title: 'Score',
        status: 'running',
        attempts: [attempt('je', 'running')],
      }),
      item('f', {
        title: 'Poster',
        status: 'queued',
        attempts: [attempt('jf', 'queued')],
      }),
    ]),
  ).items
const row = (key: string) => {
  const found = rows().find((r) => r.item.key === key)
  if (!found) throw new Error(key)
  return found
}

it('asks when other items wait on it, naming them and what the drop does to them', () => {
  expect(dropWarnings(row('b'))).toEqual([
    'Shot 4 and The cut wait on this item. Dropping it removes that dependency, so they no longer wait for it.',
  ])
  expect(dropWarnings(row('a'))).toEqual([
    'Shot 3 waits on this item. Dropping it removes that dependency, so it no longer waits for it.',
  ])
})

it('asks when nothing has been run, because the server deletes it outright', () => {
  expect(dropWarnings(row('d'))).toEqual([
    'Nothing has been run for this item, so dropping it deletes it from the plan. This page can restore it only until you leave.',
  ])
})

it('asks when its job is queued or running, because the drop does not cancel it', () => {
  const says = ['Its job keeps running. Dropping does not cancel it.']
  expect(dropWarnings(row('e'))).toEqual(says)
  expect(dropWarnings(row('f'))).toEqual(says)
})

it('gives every reason that applies, and none for a plain drop', () => {
  expect(dropWarnings(row('c'))).toHaveLength(2)
  const plain = planState(
    request([
      item('z', { status: 'failed', attempts: [attempt('j', 'failed')] }),
    ]),
  ).items[0]
  expect(dropWarnings(plain)).toEqual([])
})

it('says where a restored item went and which dependencies it lost', () => {
  expect(restoredText('Shot 3', [])).toBe(
    'Restored Shot 3 at the end of the plan.',
  )
  expect(restoredText('Shot 3', ['cast'])).toBe(
    'Restored Shot 3 at the end of the plan. It no longer waits on cast, which is no longer in the plan.',
  )
  expect(restoredText('Shot 3', ['a', 'b'])).toContain(
    'a and b, which are no longer in the plan',
  )
})
