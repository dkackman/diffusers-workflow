import { expect, it } from 'vitest'
import { formatDay } from '../format'
import { planState } from './itemState'
import { agentSaid, needsLine, needsParts, standing } from './standing'
import {
  attempt,
  doneItem,
  entry,
  item,
  request,
  revisedItem,
} from './fixtures'
import type { LedgerItem, LedgerRequest } from '../types'

const say = (items: LedgerItem[], over: Partial<LedgerRequest> = {}) => {
  const req = request(items, over)
  return standing(req, planState(req))
}
const running = (key: string, title: string) =>
  item(key, {
    title,
    status: 'running',
    attempts: [attempt(`j${key}`, 'running')],
  })
const failed = (key: string, over: Partial<LedgerItem> = {}) =>
  item(key, {
    status: 'failed',
    attempts: [attempt(`j${key}`, 'failed')],
    ...over,
  })

it('names the one item rendering, and counts two', () => {
  expect(say([doneItem('a'), running('b', 'Shot 3'), item('c')]).sentence).toBe(
    'Rendering Shot 3. 1 of 3 done.',
  )
  expect(say([running('a', 'A'), running('b', 'B')]).sentence).toBe(
    'Rendering 2 items. 0 of 2 done.',
  )
})

it('says what is queued when nothing is rendering', () => {
  const queued = item('a', {
    status: 'queued',
    attempts: [attempt('j', 'queued')],
  })
  expect(say([queued, item('b', { ready: true })]).sentence).toBe(
    '1 queued behind other jobs. 0 of 2 done.',
  )
})

it('says what is ready when nothing is in flight', () => {
  const said = say([
    doneItem('a'),
    item('b', { ready: true }),
    item('c', { ready: true }),
  ])
  expect(said.sentence).toBe('Nothing is running. 1 of 3 done.')
  expect(said.needs).toBe('Needs you: 2 ready to run.')
})

it('follows waiting items down to what they are stuck on', () => {
  expect(
    say([
      failed('shot', { title: 'Shot 3' }),
      item('grade', { depends_on: ['shot'] }),
      item('cut', { depends_on: ['grade'] }),
    ]).sentence,
  ).toBe('2 items wait on Shot 3, which failed. 0 of 3 done.')
  expect(
    say([
      item('score', { title: 'Score', workflow: null }),
      item('cut', { depends_on: ['score'] }),
    ]).sentence,
  ).toBe(
    '1 item waits on Score, which is planned with no workflow. 0 of 2 done.',
  )
  expect(
    say([
      item('a', {
        title: 'Board',
        status: 'failed',
        attempts: [attempt('j', 'cancelled')],
      }),
      item('b', { depends_on: ['a'] }),
    ]).sentence,
  ).toBe('1 item waits on Board, which was cancelled. 0 of 2 done.')
})

it('says everything is done while the request is still open', () => {
  const said = say([doneItem('a'), doneItem('b')])
  expect(said.sentence).toBe(
    'Everything is done. The agent has not closed this request. 2 of 2 done.',
  )
  expect(said.needs).toBeNull()
})

it('says who closed a request, and when', () => {
  const at = '2026-03-04T12:00:00+00:00'
  const day = formatDay(Date.parse(at) / 1000)
  expect(
    say([doneItem('a'), item('b')], {
      state: 'done',
      history: [
        entry(1, 'agent', 'opened', { state: 'open' }),
        entry(2, 'agent', 'all delivered', { state: 'done', at }),
        entry(3, 'user', 'comment on a'),
      ],
    }).sentence,
  ).toBe(`Marked done by the agent on ${day}. 1 of 2 done.`)
  expect(
    say([], {
      state: 'abandoned',
      history: [entry(4, 'user', 'not needed', { state: 'abandoned', at })],
    }).sentence,
  ).toBe(`Abandoned by you on ${day}.`)
  // a ledger written before history carried the state: its last entry
  expect(
    say([], { state: 'done', history: [entry(2, 'agent', 'closing', { at })] })
      .sentence,
  ).toBe(`Marked done by the agent on ${day}.`)
})

it('a job still running in a closed request is what is said first', () => {
  expect(say([running('a', 'Shot 1')], { state: 'abandoned' }).sentence).toBe(
    'Rendering Shot 1. 0 of 1 done.',
  )
})

it('has a sentence for a plan with nothing to do, and for no plan', () => {
  expect(say([failed('a'), doneItem('b')])).toEqual({
    sentence: 'Nothing is running. 1 of 2 done.',
    needs: 'Needs you: 1 failed.',
  })
  expect(say([]).sentence).toBe('Nothing is planned yet.')
})

it('lists every reason the user is needed, in one line', () => {
  expect(
    needsLine({
      held: 0,
      ready: 2,
      revised: 0,
      failed: 1,
      choose: 1,
      accepted_missing: 0,
    }),
  ).toBe('Needs you: 2 ready to run, 1 failed, 1 take to choose.')
  expect(
    needsLine({
      held: 0,
      ready: 0,
      revised: 0,
      failed: 0,
      choose: 2,
      accepted_missing: 1,
    }),
  ).toBe('Needs you: 2 takes to choose, 1 accepted take missing.')
  expect(
    needsLine({
      held: 0,
      ready: 0,
      revised: 0,
      failed: 0,
      choose: 0,
      accepted_missing: 0,
    }),
  ).toBeNull()
})

it("gives the agent's latest word on an open request, and none on a closed one", () => {
  const req = (over: Partial<LedgerRequest>) =>
    request([item('a')], {
      history: [
        entry(1, 'agent', 'opened'),
        entry(2, 'user', 'note added'),
        entry(3, 'agent', 'board-01 now blue; awaiting go-ahead to draw'),
        entry(4, 'user', 'note added'),
      ],
      ...over,
    })
  const said = agentSaid(req({}))
  expect(said).toEqual({
    revision: 3,
    why: 'board-01 now blue; awaiting go-ahead to draw',
    at: '2026-01-01T00:03:00+00:00',
  })
  expect(agentSaid(req({ state: 'done' }))).toBeNull()
  expect(
    agentSaid(req({ history: [entry(1, 'user', 'made by hand')] })),
  ).toBeNull()
})

it('counts a revised item as ready to run, and as needing the user', () => {
  expect(say([revisedItem('a'), item('b', { depends_on: ['a'] })])).toEqual({
    sentence: 'Nothing is running. 0 of 2 done.',
    needs: 'Needs you: 1 revised since its take.',
  })
  expect(
    needsLine({
      held: 0,
      ready: 1,
      revised: 2,
      failed: 0,
      choose: 0,
      accepted_missing: 0,
    }),
  ).toBe('Needs you: 1 ready to run, 2 revised since their takes.')
})

it('the lead leaves the count to the needs line', () => {
  const said = say([item('a', { ready: true }), item('b', { ready: true })])
  expect(said.sentence).toBe('Nothing is running. 0 of 2 done.')
  expect(said.sentence).not.toMatch(/ready/)
  expect(said.needs).toBe('Needs you: 2 ready to run.')
})

const hold = {
  until: '2026-11-21T12:00:00+00:00',
  state: 'holding' as const,
  seconds_left: 30,
}

it('needs parts name the first item of each reason', () => {
  const req = request([
    doneItem('cast'),
    failed('plate'),
    item('b', { ready: true }),
    item('a', { ready: true }),
    failed('blocked', { depends_on: ['b'] }),
    failed('bad'),
  ])
  expect(needsParts(planState(req))).toEqual([
    { reason: 'ready', text: '2 ready to run', key: 'b' },
    { reason: 'failed', text: '2 failed', key: 'plate' },
  ])
  expect(needsParts(planState(request([doneItem('a')])))).toEqual([])
})

it("the parts come in the copy's order", () => {
  const req = request([
    doneItem('two', { attempts: [attempt('x'), attempt('y')] }),
    doneItem('gone', {
      accepted: { job: 'job-gone', run_id: null },
      accepted_missing: true,
    }),
    revisedItem('rev'),
    failed('bad'),
    item('go', { ready: true }),
    item('held', { ready: true, review: hold }),
  ])
  const parts = needsParts(planState(req))
  expect(parts.map((p) => p.text)).toEqual([
    '1 waiting for your OK',
    '1 ready to run',
    '1 failed',
    '1 revised since its take',
    '1 take to choose',
    '1 accepted take missing',
  ])
  expect(parts.map((p) => p.key)).toEqual([
    'held',
    'go',
    'bad',
    'rev',
    'two',
    'gone',
  ])
  expect(standing(req, planState(req)).needs).toBe(
    `Needs you: ${parts.map((p) => p.text).join(', ')}.`,
  )
})
