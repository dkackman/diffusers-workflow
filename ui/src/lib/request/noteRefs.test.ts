import { expect, it } from 'vitest'
import { addRef, refAttempt, refLabel, writtenRefs } from './noteRefs'
import { attempt, item, request } from './fixtures'

const req = request([
  item('s3', {
    title: 'Shot 3',
    attempts: [attempt('j1', 'failed'), attempt('j2')],
  }),
])

it('adds a reference once: the same item, job and file are one', () => {
  const take = { item: 's3', job: 'j2', output: 'run/s3.mp4' }
  const one = addRef([], take)
  expect(one).toEqual([take])
  expect(addRef(one, { ...take, kind: 'video' as const })).toBe(one)
  expect(addRef(one, { item: 's3', job: 'j2' })).toHaveLength(2)
  expect(addRef(one, { item: 's3' })).toHaveLength(2)
})

it('numbers a take by its place among the item’s attempts', () => {
  expect(refAttempt({ item: 's3', job: 'j2', attempt: 7 }, req)).toBe(7)
  expect(refAttempt({ item: 's3', job: 'j2' }, req)).toBe(2)
  expect(refAttempt({ item: 's3', job: 'nope' }, req)).toBeNull()
  expect(refAttempt({ item: 's3' }, req)).toBeNull()
})

it('labels a file, a take and a planned item', () => {
  expect(refLabel({ item: 's3', job: 'j2', output: 'run/a/s3.mp4' }, req)).toBe(
    'take 2 · s3.mp4',
  )
  expect(refLabel({ item: 's3', job: 'j1' }, req)).toBe('take 1')
  expect(refLabel({ item: 's3' }, req)).toBe('Shot 3')
  expect(refLabel({ item: 'gone' }, req)).toBe('gone')
})

it('sends only what a write takes', () => {
  expect(
    writtenRefs([
      { item: 's3', job: 'j2', output: 'o.mp4', kind: 'video', attempt: 2 },
      { item: 's5', gone: true },
    ]),
  ).toEqual([{ item: 's3', job: 'j2', output: 'o.mp4' }, { item: 's5' }])
})
