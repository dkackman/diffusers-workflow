import { expect, it, vi, type Mock } from 'vitest'
import { ApiError } from '../apiErrors'
import type { JobDetail, Plan, ValidationResult } from '../types'
import type { Candidate } from './runPlan'
import { PRICING_LANES, RunQueue, type RunApi } from './runQueue.svelte'

const candidate = (key: string, over: Partial<Candidate> = {}): Candidate => ({
  key,
  title: `Item ${key}`,
  workflow: `flow-${key}`,
  arguments: { seed: 1 },
  planMinutes: null,
  ...over,
})
const quote = (key: string, minutes = 5) =>
  ({
    fingerprint: `fp-${key}`,
    steps: 2,
    cached_steps: 0,
    downloads_required: [],
    estimate: { minutes, basis: 'observed' },
  }) as unknown as Plan
const valid = (key: string, minutes = 5): ValidationResult =>
  ({
    valid: true,
    error: null,
    errors: [],
    warnings: [],
    plan: quote(key, minutes),
  }) as ValidationResult
const job = (id: string) => ({ id, acknowledged: 'bound' }) as JobDetail

type Fakes = { validate: Mock; submitJob: Mock }
function setup(keys: string[], over: Partial<Fakes> = {}) {
  const api: Fakes = {
    validate: vi.fn(async (body: { workflow_path?: string }) =>
      valid((body.workflow_path ?? '').replace('flow-', '')),
    ),
    submitJob: vi.fn(async (body: { item?: string }) =>
      job(`job-${body.item}`),
    ),
    ...over,
  }
  const queue = new RunQueue(
    keys.map((key) => candidate(key)),
    { workspace: 'studio', requestId: 'r1', api: api as unknown as RunApi },
  )
  return { api, queue }
}
const states = (queue: RunQueue) => queue.rows.map((row) => row.state)

it('prices every candidate with what its job would run, three at a time', async () => {
  let out = 0
  let most = 0
  const release: (() => void)[] = []
  const { api, queue } = setup(['a', 'b', 'c', 'd', 'e'], {
    validate: vi.fn(async (body: { workflow_path?: string }) => {
      most = Math.max(most, ++out)
      await new Promise<void>((resolve) => release.push(resolve))
      out--
      return valid((body.workflow_path ?? '').replace('flow-', ''))
    }) as Mock,
  })
  expect(states(queue)).toEqual(Array(5).fill('pricing'))
  expect(queue.settled).toBe(false)
  const pricing = queue.price()
  await Promise.resolve()
  expect(api.validate).toHaveBeenCalledTimes(PRICING_LANES)
  expect(api.validate.mock.calls[0][0]).toEqual({
    workflow_path: 'flow-a',
    arguments: { seed: 1 },
    workspace: 'studio',
  })
  // nothing can be spent until every row has its answer
  await queue.submit()
  expect(api.submitJob).not.toHaveBeenCalled()
  while (release.length || out) {
    release.shift()?.()
    await new Promise((r) => setTimeout(r, 0))
  }
  await pricing
  expect(most).toBe(PRICING_LANES)
  expect(states(queue)).toEqual(Array(5).fill('priced'))
  expect(queue.settled).toBe(true)
  expect(queue.rows[1].plan?.fingerprint).toBe('fp-b')
})

it('a candidate validation refuses will not run, with its first error', async () => {
  const { queue } = setup(['a', 'b', 'c', 'd'], {
    validate: vi.fn(async (body: { workflow_path?: string }) => {
      if (body.workflow_path === 'flow-b')
        return {
          valid: false,
          error: null,
          errors: [{ message: 'seed must be an integer', path: 'seed' }],
          warnings: [],
        }
      if (body.workflow_path === 'flow-c') throw new Error('offline')
      if (body.workflow_path === 'flow-d')
        throw new ApiError('workflow_path must name a workflow', 400)
      return valid('a')
    }) as Mock,
  })
  await queue.price()
  expect(states(queue)).toEqual(['priced', 'refused', 'refused', 'refused'])
  expect(queue.rows[1].error).toBe('would be refused: seed must be an integer')
  expect(queue.rows[2].error).toBe('could not be checked: offline')
  // the server refusing the question is what a submit would meet too
  expect(queue.rows[3].error).toBe(
    'would be refused: workflow_path must name a workflow',
  )
  expect(queue.rows[1].ticked).toBe(false)
  expect(queue.chosen.map((row) => row.candidate.key)).toEqual(['a'])
})

it('closing drops the validations still out, and prices no more', async () => {
  const signals: AbortSignal[] = []
  const { api, queue } = setup(['a', 'b', 'c', 'd'], {
    validate: vi.fn(
      (_body: unknown, signal?: AbortSignal) =>
        new Promise((_resolve, reject) => {
          if (signal) signals.push(signal)
          signal?.addEventListener('abort', () =>
            reject(new DOMException('aborted', 'AbortError')),
          )
        }),
    ) as Mock,
  })
  const pricing = queue.price()
  await Promise.resolve()
  queue.close()
  await pricing
  expect(signals.every((signal) => signal.aborted)).toBe(true)
  expect(api.validate).toHaveBeenCalledTimes(PRICING_LANES)
  // not an error on the row: the dialog is gone
  expect(queue.rows.every((row) => row.error === null)).toBe(true)
})

it('queues the ticked rows in plan order, each bound to the plan it was quoted', async () => {
  const { api, queue } = setup(['a', 'b', 'c'])
  await queue.price()
  queue.rows[1].ticked = false
  await queue.submit()
  expect(api.submitJob.mock.calls.map(([body]) => body)).toEqual([
    {
      workflow_path: 'flow-a',
      arguments: { seed: 1 },
      request: 'r1',
      item: 'a',
      acknowledged_cost: quote('a'),
    },
    {
      workflow_path: 'flow-c',
      arguments: { seed: 1 },
      request: 'r1',
      item: 'c',
      acknowledged_cost: quote('c'),
    },
  ])
  expect(states(queue)).toEqual(['queued', 'priced', 'queued'])
  expect(queue.rows[0].job).toBe('job-a')
  expect(queue.finished).toBe(true)
  expect(queue.queued).toHaveLength(2)
})

it('a row is queued only when the server says so', async () => {
  let answer: (job: JobDetail) => void = () => {}
  const { queue } = setup(['a'], {
    submitJob: vi.fn(
      () => new Promise<JobDetail>((resolve) => (answer = resolve)),
    ) as Mock,
  })
  await queue.price()
  const sending = queue.submit()
  await Promise.resolve()
  expect(states(queue)).toEqual(['sending'])
  expect(queue.finished).toBe(false)
  expect(queue.busy).toBe(true)
  answer(job('job-a'))
  await sending
  expect(states(queue)).toEqual(['queued'])
  expect(queue.busy).toBe(false)
})

it('a price that moved re-prices that row in place, and the others go on', async () => {
  const moved = quote('a', 9)
  const acknowledge = { fingerprint: 'fp-a2', minutes: 9, downloads: [] }
  const { api, queue } = setup(['a', 'b'], {
    submitJob: vi.fn(async (body: { item?: string }) => {
      if (body.item === 'a')
        throw new ApiError("The run's shape changed", 409, undefined, {
          reason: 'fingerprint',
          plan: moved,
          acknowledge,
        })
      return job('job-b')
    }) as Mock,
  })
  await queue.price()
  await queue.submit()
  expect(states(queue)).toEqual(['repriced', 'queued'])
  expect(queue.rows[0].plan?.estimate.minutes).toBe(9)
  expect(queue.finished).toBe(false)
  // the Run button does not send it again: it is the user's to accept
  expect(queue.chosen).toEqual([])
  api.submitJob.mockResolvedValue(job('job-a'))
  await queue.accept('a')
  expect(api.submitJob.mock.calls.at(-1)?.[0]).toMatchObject({
    item: 'a',
    acknowledged_cost: acknowledge,
  })
  expect(states(queue)).toEqual(['queued', 'queued'])
  expect(queue.finished).toBe(true)
})

it("says a refused link in plain words, by the server's code", async () => {
  const refusals: Record<string, [number, string | undefined, string]> = {
    a: [
      422,
      'request_closed',
      'This request is closed. Reopen it to run its items.',
    ],
    b: [422, 'item_dropped', 'This item was dropped from the plan.'],
    c: [422, 'unknown_item', 'This item is no longer in the plan.'],
    d: [422, 'unknown_request', 'This request no longer exists.'],
    e: [
      400,
      undefined,
      "The item's stored workflow or arguments were refused: no variable 'sed'",
    ],
    f: [500, undefined, 'the worker is down'],
  }
  const { queue } = setup(Object.keys(refusals), {
    submitJob: vi.fn(async (body: { item?: string }) => {
      const [status, code] = refusals[body.item ?? '']
      const text =
        status === 400
          ? "no variable 'sed'"
          : status === 500
            ? 'the worker is down'
            : 'server words'
      throw new ApiError(text, status, code)
    }) as Mock,
  })
  await queue.price()
  await queue.submit()
  expect(states(queue)).toEqual(Array(6).fill('failed'))
  expect(queue.rows.map((row) => row.error)).toEqual(
    Object.values(refusals).map(([, , words]) => words),
  )
  expect(queue.finished).toBe(false)
})

it('a 409 with no plan to re-quote from is a failure, not a new price', async () => {
  const { queue } = setup(['a'], {
    submitJob: vi.fn(async () => {
      throw new ApiError('The run could not be planned', 409, undefined, {
        reason: 'unplannable',
        plan: null,
      })
    }) as Mock,
  })
  await queue.price()
  await queue.submit()
  expect(states(queue)).toEqual(['failed'])
  expect(queue.rows[0].error).toBe('The run could not be planned')
})

it('a row validation gave no plan is acknowledged plainly, not bound', async () => {
  const { api, queue } = setup(['a'], {
    validate: vi.fn(async () => ({ ...valid('a'), plan: null })) as Mock,
  })
  await queue.price()
  await queue.submit()
  expect(api.submitJob.mock.calls[0][0]).toMatchObject({
    acknowledged_cost: true,
  })
})

it('with nothing ticked there is nothing to run, and it is not finished', async () => {
  const { api, queue } = setup(['a'])
  await queue.price()
  queue.rows[0].ticked = false
  await queue.submit()
  expect(api.submitJob).not.toHaveBeenCalled()
  expect(queue.finished).toBe(false)
})
