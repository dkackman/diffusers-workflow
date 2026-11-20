import { expect, it } from 'vitest'
import {
  activeJob,
  bandsOf,
  figureOf,
  opensByDefault,
  planState,
  resultMark,
  takesLine,
  takesOf,
  toneOf,
} from './itemState'
import { attempt, doneItem, item, request } from './fixtures'
import type { LedgerItem } from '../types'

const stateOf = (req: ReturnType<typeof request>, key: string) => {
  const found = planState(req).items.find((s) => s.item.key === key)
  if (!found) throw new Error(`no item ${key}`)
  return found
}

it("takes the server's ready, and says waiting for an unmet dependency", () => {
  const plan = planState(
    request([
      doneItem('cast'),
      item('shot', { depends_on: ['cast'], ready: true }),
      item('cut', { depends_on: ['shot'] }),
    ]),
  )
  expect(plan.items.map((s) => s.display)).toEqual(['done', 'ready', 'waiting'])
  expect(plan.items[2].blockers).toEqual([
    { key: 'shot', title: 'Item shot', display: 'ready' },
  ])
  expect(plan.items[0].neededBy.map((r) => r.key)).toEqual(['shot'])
  expect(plan.items[1].after.map((r) => r.display)).toEqual(['done'])
  expect(plan.counts).toMatchObject({ done: 1, ready: 1, waiting: 1 })
})

it('a planned item with no stored workflow is planned, and not runnable', () => {
  const state = stateOf(request([item('a', { workflow: null })]), 'a')
  expect(state.display).toBe('planned')
  expect(state.noWorkflow).toBe(true)
  expect(state.runnable).toBe(false)
  expect(state.needs).toEqual([])
})

it('separates every attempt cancelled from a failure', () => {
  const plan = planState(
    request([
      item('a', {
        status: 'failed',
        attempts: [attempt('j1', 'cancelled'), attempt('j2', 'cancelled')],
      }),
      item('b', {
        status: 'failed',
        attempts: [attempt('j3', 'cancelled'), attempt('j4', 'failed')],
      }),
    ]),
  )
  expect(plan.items.map((s) => s.display)).toEqual(['cancelled', 'failed'])
  // both can be run again; only the failure asks for the user
  expect(plan.items.map((s) => s.runnable)).toEqual([true, true])
  expect(plan.items.map((s) => s.needs)).toEqual([[], ['failed']])
  expect(plan.needs.failed).toBe(1)
})

it('a failed item with an unmet dependency or no workflow cannot be run again', () => {
  const failed: Partial<LedgerItem> = {
    status: 'failed',
    attempts: [attempt('j', 'failed')],
  }
  const plan = planState(
    request([
      item('dep'),
      item('a', { ...failed, depends_on: ['dep'] }),
      item('b', { ...failed, workflow: null }),
    ]),
  )
  expect(plan.items.map((s) => s.runnable)).toEqual([false, false, false])
})

it('nothing is runnable or needs the user in a closed request', () => {
  const plan = planState(
    request(
      [
        item('a'),
        item('b', { status: 'failed', attempts: [attempt('j', 'failed')] }),
        doneItem('c'),
      ],
      { state: 'done' },
    ),
  )
  expect(plan.items.map((s) => s.runnable)).toEqual([false, false, false])
  expect(plan.items.map((s) => s.canRetake)).toEqual([false, false, false])
  expect(plan.items.flatMap((s) => s.needs)).toEqual([])
})

it('a done item can take another run, and wants a choice past one good take', () => {
  const two = doneItem('a', {
    attempts: [attempt('j1'), attempt('j2')],
    result: { job: 'j2', run_id: 'r2', output: null },
  })
  const state = stateOf(request([two]), 'a')
  expect(state.canRetake).toBe(true)
  expect(state.runnable).toBe(false)
  expect(state.needs).toEqual(['choose'])
  const chosen = { ...two, accepted: { job: 'j1', run_id: 'r1' } }
  expect(stateOf(request([chosen]), 'a').needs).toEqual([])
  const gone = { ...chosen, accepted_missing: true }
  expect(stateOf(request([gone]), 'a').needs).toEqual(['accepted_missing'])
})

it('keeps dropped items out of the live plan and its positions', () => {
  const plan = planState(
    request([
      item('a'),
      item('x', { dropped: true, status: 'dropped' }),
      item('b', { ready: true }),
    ]),
  )
  expect(plan.live.map((s) => [s.item.key, s.position])).toEqual([
    ['a', 1],
    ['b', 2],
  ])
  expect(plan.dropped.map((s) => s.position)).toEqual([0])
  expect(plan.counts.dropped).toBe(1)
  expect(plan.dropped[0].needs).toEqual([])
})

it('opens a row by default when it is running, failed, unchosen or holds a waiting note', () => {
  const plan = planState(
    request([
      item('run', { status: 'running', attempts: [attempt('j', 'running')] }),
      item('bad', { status: 'failed', attempts: [attempt('k', 'failed')] }),
      doneItem('two', { attempts: [attempt('a'), attempt('b')] }),
      doneItem('one'),
      item('gone', { dropped: true, status: 'dropped' }),
    ]),
  )
  expect(plan.items.map((s) => opensByDefault(s, false))).toEqual([
    true,
    true,
    true,
    false,
    false,
  ])
  expect(opensByDefault(plan.items[3], true)).toBe(true)
  const gone = planState(
    request([
      doneItem('a', {
        accepted: { job: 'job-a', run_id: null },
        accepted_missing: true,
      }),
    ]),
  )
  expect(opensByDefault(gone.items[0], false)).toBe(true)
  expect(opensByDefault(plan.items[4], true)).toBe(false)
})

it('marks the accepted take, or the latest standing as the result', () => {
  const attempts = [attempt('j1'), attempt('j2', 'failed'), attempt('j3')]
  const result = { job: 'j3', run_id: 'r3', output: null }
  const latest = doneItem('a', { attempts, result })
  expect(takesOf(latest).map((t) => [t.n, t.mark])).toEqual([
    [1, null],
    [2, null],
    [3, 'latest'],
  ])
  expect(resultMark(latest)).toBe('latest')
  const accepted = doneItem('a', {
    attempts,
    accepted: { job: 'j1', run_id: 'r1' },
    result: { job: 'j1', run_id: 'r1', output: null },
  })
  expect(takesOf(accepted).map((t) => t.mark)).toEqual(['accepted', null, null])
  expect(resultMark(accepted)).toBe('accepted')
  // the accepted run's files are gone: the latest stands in for it
  const gone = { ...accepted, accepted_missing: true, result }
  expect(takesOf(gone).map((t) => t.mark)).toEqual([
    'accepted-gone',
    null,
    'latest',
  ])
  expect(resultMark(gone)).toBe('latest')
  expect(resultMark(item('b'))).toBeNull()
})

it('says how many takes a done item has, and that none worked for a failed one', () => {
  const plan = planState(
    request([
      doneItem('one'),
      doneItem('three', {
        attempts: [attempt('a'), attempt('b'), attempt('c')],
      }),
      item('bad', {
        status: 'failed',
        attempts: [attempt('x', 'failed'), attempt('y', 'cancelled')],
      }),
      item('new'),
    ]),
  )
  expect(plan.items.map(takesLine)).toEqual([
    null,
    '3 takes',
    '2 attempts, none succeeded',
    null,
  ])
})

it('gives the estimate before a run, minutes so far during, minutes spent after', () => {
  const plan = planState(
    request([
      item('a', { estimate_minutes: 6 }),
      item('b'),
      item('c', {
        status: 'running',
        attempts: [
          attempt('old', 'failed'),
          attempt('now', 'running', { minutes: 2 }),
        ],
      }),
      doneItem('d', { spent_minutes: 6.25 }),
      item('e', {
        status: 'queued',
        attempts: [attempt('q', 'queued', { minutes: null })],
      }),
    ]),
  )
  expect(plan.items.map(figureOf)).toEqual([
    'est. ~6 min',
    '',
    '2 min so far',
    '6.3 min',
    '',
  ])
  expect(activeJob(plan.items[2].item)).toBe('now')
  expect(activeJob(plan.items[4].item)).toBe('q')
  expect(activeJob(plan.items[3].item)).toBeNull()
})

it('bands a plan by status, leaving dropped items out', () => {
  expect(toneOf('ready')).toBe('rest')
  expect(
    bandsOf({
      done: 4,
      running: 1,
      planned: 3,
      waiting: 1,
      dropped: 2,
      failed: 0,
    }),
  ).toEqual([
    { tone: 'done', count: 4, label: '4 done' },
    { tone: 'running', count: 1, label: '1 running' },
    { tone: 'rest', count: 4, label: '4 not started' },
  ])
})
