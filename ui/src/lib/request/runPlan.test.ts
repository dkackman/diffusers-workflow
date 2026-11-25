import { expect, it } from 'vitest'
import { describeEstimate } from '../plan'
import type { Plan, PlanEstimate } from '../types'
import { attempt, doneItem, item, request, revisedItem } from './fixtures'
import { planState } from './itemState'
import { clockTime } from './reviewText'
import {
  afterwardsLine,
  ALL_CACHED,
  becomesReady,
  heldLine,
  heldPlan,
  itemPlan,
  planNotes,
  planSaid,
  quoteLine,
  readyPlan,
} from './runPlan'

// cast done; s3 running; s4 and title ready; s5 waits on s4; cut waits on
// s3 and s4; score has nothing stored; poster failed; x dropped
const plan = () =>
  planState(
    request([
      doneItem('cast', { title: 'Cast sheet' }),
      item('s3', {
        title: 'Shot 3',
        status: 'running',
        attempts: [attempt('j3', 'running')],
      }),
      item('s4', {
        title: 'Shot 4',
        ready: true,
        workflow: 'shot',
        arguments: { seed: 4 },
        estimate_minutes: 6,
      }),
      item('s5', { title: 'Shot 5', depends_on: ['s4'] }),
      item('cut', { title: 'The cut', depends_on: ['s3', 's4'] }),
      item('title', { title: 'Title card', ready: true, workflow: 'card' }),
      item('score', { title: 'Score', workflow: null }),
      item('poster', {
        title: 'Poster',
        status: 'failed',
        attempts: [attempt('jp', 'failed')],
      }),
      item('q', {
        title: 'Queued one',
        status: 'queued',
        attempts: [attempt('jq', 'queued')],
      }),
      item('x', { title: 'Old', dropped: true, status: 'dropped' }),
    ]),
  )

it('runs what is ready, in plan order, with what each job would run', () => {
  const run = readyPlan(plan())
  expect(run.single).toBe(false)
  expect(run.candidates.filter((c) => !c.optional)).toEqual([
    {
      key: 's4',
      title: 'Shot 4',
      workflow: 'shot',
      arguments: { seed: 4 },
      planMinutes: 6,
    },
    {
      key: 'title',
      title: 'Title card',
      workflow: 'card',
      arguments: {},
      planMinutes: null,
    },
  ])
})

it('says why each other item will not run, and nothing of done or dropped ones', () => {
  const reasons = Object.fromEntries(
    readyPlan(plan()).skipped.map((skip) => [skip.title, skip.reason]),
  )
  expect(reasons).toEqual({
    'Shot 3': 'already running',
    'Shot 5': 'waits on Shot 4 (ready)',
    'The cut': 'waits on Shot 3 (running), Shot 4 (ready)',
    Score: 'no stored workflow: ask the agent to run it',
    'Queued one': 'already queued',
  })
})

it('failed items that can run join the ready plan, optional', () => {
  const run = readyPlan(plan())
  // after the others, as Task 2's "can run again" rule says
  expect(run.candidates.map((c) => [c.key, c.optional ?? false])).toEqual([
    ['s4', false],
    ['title', false],
    ['poster', true],
  ])
  expect(run.candidates[2]).toMatchObject({ workflow: 'shot', arguments: {} })
  expect(run.skipped.map((s) => s.key)).not.toContain('poster')
})

it('a failed item that cannot run stays skipped', () => {
  const req = request([
    item('s3', { title: 'Shot 3', status: 'running' }),
    item('f', {
      title: 'Again',
      status: 'failed',
      depends_on: ['s3'],
      attempts: [attempt('jf', 'failed')],
    }),
    item('g', {
      title: 'Nothing stored',
      status: 'failed',
      workflow: null,
      attempts: [attempt('jg', 'failed')],
    }),
  ])
  const run = readyPlan(planState(req))
  expect(run.candidates).toEqual([])
  expect(run.skipped.map((s) => [s.key, s.reason])).toEqual([
    ['s3', 'already running'],
    ['f', 'waits on Shot 3 (running)'],
    ['g', 'no stored workflow: ask the agent to run it'],
  ])
})

it('no figure reads no time estimate with the reason as detail', () => {
  const unknown = (over: Partial<PlanEstimate>) =>
    ({
      minutes: null,
      basis: 'unknown',
      device: 'cuda',
      ...over,
    }) as PlanEstimate
  expect(
    quoteLine(
      unknown({ unpriced_reason: 'task-only workflow (compose_text)' }),
    ),
  ).toEqual({
    text: 'No time estimate for this workflow.',
    tone: 'plain',
    detail: 'task-only workflow (compose_text)',
  })
  expect(quoteLine(unknown({}))).toEqual({
    text: 'No time estimate for this workflow.',
    tone: 'plain',
    detail: 'no measured cost',
  })
  expect(quoteLine(null)).toEqual({
    text: 'No time estimate for this workflow.',
    tone: 'plain',
  })
  // a figure reads as describeEstimate gives it
  const priced = {
    minutes: 5,
    basis: 'observed',
    device: 'cuda',
    measured_on: 'cuda',
  } as PlanEstimate
  expect(quoteLine(priced)).toEqual(describeEstimate(priced))
})

it('says what becomes ready when the chosen items finish, one step on only', () => {
  const state = plan()
  expect(becomesReady(state, ['s4'])).toEqual([{ key: 's5', title: 'Shot 5' }])
  // the cut also waits on Shot 3, which is not among these
  expect(becomesReady(state, ['s4', 'title']).map((n) => n.key)).toEqual(['s5'])
  expect(becomesReady(state, ['s3', 's4']).map((n) => n.key)).toEqual([
    's5',
    'cut',
  ])
  expect(becomesReady(state, [])).toEqual([])
  expect(afterwardsLine(state, ['s3', 's4'])).toBe(
    'When these finish, 2 more become ready: Shot 5, The cut. This page will not start them for you.',
  )
  expect(afterwardsLine(state, ['s4'])).toBe(
    'When this finishes, 1 more becomes ready: Shot 5. This page will not start it for you.',
  )
  expect(afterwardsLine(state, ['title'])).toBeNull()
})

it('a single run is the one item: a ready one, a failed one again', () => {
  expect(itemPlan(plan(), 's4')).toMatchObject({
    single: true,
    retake: null,
    candidates: [{ key: 's4' }],
    skipped: [],
  })
  expect(itemPlan(plan(), 'poster').candidates[0].key).toBe('poster')
  expect(itemPlan(plan(), 'gone').candidates).toEqual([])
  expect(itemPlan(plan(), 's5')).toMatchObject({
    candidates: [],
    skipped: [{ key: 's5', reason: 'waits on Shot 4 (ready)' }],
  })
})

it('another take of a done item says what stands as the result meanwhile', () => {
  expect(itemPlan(plan(), 'cast').retake).toBe(
    'Nothing is accepted, so the new take becomes the result when it finishes.',
  )
  const chosen = planState(
    request([
      doneItem('cast', {
        attempts: [attempt('j1'), attempt('j2'), attempt('j3')],
        accepted: { job: 'j2', run_id: 'r2' },
        result: { job: 'j2', run_id: 'r2', output: null },
      }),
    ]),
  )
  expect(itemPlan(chosen, 'cast').retake).toBe(
    'Take 2 stays the accepted result until you accept another.',
  )
})

it('says what the plan estimated only when the quote is far from it', () => {
  expect(planSaid(4, 9)).toBe('the plan said about 4 min')
  expect(planSaid(4, 2.9)).toBe('the plan said about 4 min')
  expect(planSaid(4, 5)).toBeNull()
  expect(planSaid(4, 3)).toBeNull()
  expect(planSaid(null, 9)).toBeNull()
  expect(planSaid(4, null)).toBeNull()
  expect(planSaid(0, 3)).toBeNull()
})

const quote = (over: Partial<Plan>): Plan =>
  ({
    steps: 3,
    list_entries: {},
    cached_steps: 0,
    elided_steps: [],
    downloads_required: [],
    fingerprint: 'fp',
    workspace: 'studio',
    estimate: {
      minutes: 5,
      basis: 'observed',
      device: 'cuda',
      measured_on: 'cuda',
      partial: false,
      cached_minutes: null,
      runs: 3,
    },
    ...over,
  }) as Plan

it('adds what the cache would serve and what must be downloaded first', () => {
  expect(planNotes(quote({}))).toEqual([
    { text: '0 of 3 steps cached', tone: 'plain' },
  ])
  expect(planNotes(quote({ cached_steps: null }))).toEqual([])
  const notes = planNotes(
    quote({
      cached_steps: 1,
      downloads_required: [
        { repo: 'org/model', gb: 12, access_blocked: false, gated: false },
      ],
    }),
  )
  expect(notes.map((n) => n.text)).toEqual([
    '1 of 3 steps cached',
    'needs download: org/model (12 GB)',
  ])
  expect(notes[1].tone).toBe('warn')
})

it('warns that a run with every step cached makes nothing new', () => {
  expect(planNotes(quote({ cached_steps: 3 }))).toEqual([
    { text: ALL_CACHED, tone: 'warn' },
  ])
})

const UNTIL = '2026-11-21T15:30:00+00:00'
const hold = (state: 'holding' | 'redirected' | 'due') => ({
  until: UNTIL,
  state,
  seconds_left: state === 'due' ? 0 : 600,
})

it('runs the held rows named that are ready, and none that is not', () => {
  const held = planState(
    request([
      item('a', { title: 'A', ready: true, review: hold('holding') }),
      item('b', { title: 'B', review: hold('holding') }),
      item('c', { title: 'C', ready: true }),
    ]),
  )
  const run = heldPlan(held, ['a', 'b'])
  expect(run.candidates.map((c) => c.key)).toEqual(['a'])
  expect(run.single).toBe(false)
  expect(heldPlan(held, ['b']).candidates).toEqual([])
})

it("gives a held item's dialog row its line, warning when a session may be starting it", () => {
  const go = {
    text: `Waiting for your OK until ${clockTime(UNTIL)}. Running it now is your go-ahead.`,
    tone: 'plain',
  }
  expect(heldLine(null, true)).toBeNull()
  expect(heldLine(hold('holding'), true)).toEqual(go)
  expect(heldLine(hold('redirected'), false)).toEqual(go)
  expect(heldLine(hold('due'), false)).toEqual(go)
  expect(heldLine(hold('due'), true)).toEqual({
    text: 'Review time is up and the agent is connected: it may be starting this now.',
    tone: 'warn',
  })
})

it('runs what is ready with a revised item among them, and skips one that waits', () => {
  const req = request([
    revisedItem('a', { workflow: 'board', arguments: { seed: 1 } }),
    item('b', { depends_on: ['a'] }),
    item('c', { ready: true }),
    revisedItem('d', { depends_on: ['b'] }),
  ])
  const run = readyPlan(planState(req))
  expect(run.candidates.map((c) => c.key)).toEqual(['a', 'c'])
  expect(run.candidates[0]).toMatchObject({
    workflow: 'board',
    arguments: { seed: 1 },
  })
  expect(run.skipped.map((s) => [s.key, s.reason])).toEqual([
    ['b', 'waits on Item a (revised)'],
    ['d', 'waits on Item b (waiting)'],
  ])
})

it('one revised item runs a new take, and says what stands meanwhile', () => {
  const req = request([revisedItem('a', { workflow: 'board' })])
  const run = itemPlan(planState(req), 'a')
  expect(run.candidates.map((c) => c.workflow)).toEqual(['board'])
  expect(run.retake).toBe(
    'Take 1 stays the accepted result until you accept another.',
  )
})
