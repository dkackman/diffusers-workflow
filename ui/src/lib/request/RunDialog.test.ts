import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import RunDialog from './RunDialog.svelte'
import { attempt, doneItem, item, request } from './fixtures'
import { planState } from './itemState'
import { ALL_CACHED, itemPlan, readyPlan } from './runPlan'
import type { Plan } from '../types'

const world = vi.hoisted(() => ({
  validate: vi.fn(),
  submitJob: vi.fn(),
  notify: { error: vi.fn(), success: vi.fn(), undo: vi.fn() },
}))
vi.mock('../api', async () => ({
  ApiError: (await import('../apiErrors')).ApiError,
  api: {
    validate: (...args: unknown[]) => world.validate(...args),
    submitJob: (...args: unknown[]) => world.submitJob(...args),
  },
}))
vi.mock('../toast', () => ({ notify: world.notify }))

const quote = (minutes: number | null, over: Partial<Plan> = {}): Plan =>
  ({
    steps: 2,
    list_entries: {},
    cached_steps: 0,
    elided_steps: [],
    downloads_required: [],
    fingerprint: `fp-${minutes}`,
    workspace: 'studio',
    estimate: {
      minutes,
      basis: minutes === null ? 'unknown' : 'observed',
      device: 'cuda',
      measured_on: 'cuda',
      partial: false,
      cached_minutes: null,
      runs: 3,
    },
    ...over,
  }) as Plan
const answer = (plan: Plan | null) => ({
  valid: true,
  error: null,
  errors: [],
  warnings: [],
  plan,
})

// cast done (two takes, take 1 accepted); s4 and title ready; s5 waits on
// s4; score has nothing stored
const plan = () =>
  planState(
    request([
      doneItem('cast', {
        title: 'Cast sheet',
        attempts: [attempt('j1'), attempt('j2')],
        accepted: { job: 'j1', run_id: 'r1' },
        result: { job: 'j1', run_id: 'r1', output: null },
      }),
      item('s4', {
        title: 'Shot 4',
        ready: true,
        workflow: 'shot',
        estimate_minutes: 4,
      }),
      item('s5', { title: 'Shot 5', depends_on: ['s4'] }),
      item('title', { title: 'Title card', ready: true, workflow: 'card' }),
      item('score', { title: 'Score', workflow: null }),
    ]),
  )

const PLANS: Record<string, Plan | null> = {
  shot: quote(9),
  card: quote(5.5),
}
beforeEach(() => {
  vi.clearAllMocks()
  world.validate.mockImplementation(async (body: { workflow_path: string }) =>
    answer(PLANS[body.workflow_path] ?? null),
  )
  world.submitJob.mockImplementation(async (body: { item: string }) => ({
    id: `job-${body.item}`,
  }))
})
afterEach(() => {
  cleanup()
  document.body.innerHTML = ''
})

function open(key?: string) {
  const state = plan()
  const onclose = vi.fn()
  render(RunDialog, {
    run: key ? itemPlan(state, key) : readyPlan(state),
    plan: state,
    workspace: 'studio',
    requestId: 'r1',
    onclose,
  })
  return { onclose }
}
const runRow = (key: string) =>
  document.querySelector(`.runrow[data-key="${key}"]`) as HTMLElement
const runButton = () =>
  screen.getByRole('button', {
    name: /^(Run \d+|Queuing…)$/,
  }) as HTMLButtonElement

it('starts on Cancel, and cannot spend until every row is priced', async () => {
  let land: (value: unknown) => void = () => {}
  world.validate.mockImplementation(
    (body: { workflow_path: string }) =>
      new Promise((resolve) => {
        if (body.workflow_path === 'card') land = resolve
        else resolve(answer(PLANS.shot))
      }),
  )
  open()
  const dialog = await screen.findByRole('dialog', {
    name: 'Run what is ready',
  })
  await waitFor(() =>
    expect(document.activeElement).toBe(
      within(dialog).getByRole('button', { name: 'Cancel' }),
    ),
  )
  // one priced, one still out
  await within(runRow('s4')).findByText(/~9 min on cuda/)
  expect(
    within(runRow('title')).getByRole('status', { name: 'pricing Title card' }),
  ).toBeTruthy()
  expect(runButton().disabled).toBe(true)
  land(answer(PLANS.card))
  await within(runRow('title')).findByText(/~5.5 min on cuda/)
  expect(runButton().disabled).toBe(false)
  expect(runButton().textContent).toBe('Run 2')
  expect(world.submitJob).not.toHaveBeenCalled()
})

it('prices each item with its workflow, and says what the plan had estimated', async () => {
  open()
  await within(runRow('s4')).findByText(/~9 min on cuda/)
  expect(world.validate.mock.calls.map(([body]) => body)).toEqual([
    { workflow_path: 'shot', arguments: {}, workspace: 'studio' },
    { workflow_path: 'card', arguments: {}, workspace: 'studio' },
  ])
  const s4 = within(runRow('s4'))
  expect(s4.getByText('shot')).toBeTruthy()
  // the plan said 4, the quote is 9: more than a quarter off
  expect(s4.getByText('the plan said ~4 min')).toBeTruthy()
  expect(within(runRow('title')).queryByText(/the plan said/)).toBeNull()
  expect(document.querySelector('.total')?.textContent).toBe('total ~14.5 min')
})

it('the total follows the ticked rows, and counts one with no figure', async () => {
  PLANS.card = quote(null)
  open()
  await within(runRow('title')).findByText('no measured cost')
  expect(document.querySelector('.total')?.textContent).toBe(
    'total ~9 min, plus 1 with no figure',
  )
  await fireEvent.click(screen.getByRole('checkbox', { name: 'Run Shot 4' }))
  expect(document.querySelector('.total')?.textContent).toBe('total: no figure')
  expect(runButton().textContent).toBe('Run 1')
  PLANS.card = quote(5.5)
})

it('lists what will not run and why, and what becomes ready afterwards', async () => {
  open()
  await within(runRow('s4')).findByText(/~9 min/)
  const wont = document.querySelector('.wont') as HTMLElement
  expect(within(wont).getByRole('heading').textContent).toBe('Will not run')
  expect([...wont.querySelectorAll('li')].map((li) => li.textContent)).toEqual([
    'Shot 5: waits on Shot 4 (ready)',
    'Score: no stored workflow: ask the agent to run it',
  ])
  expect(
    screen.getByText(
      'Afterwards: When these finish, 1 more becomes ready: Shot 5. This page will not start it for you.',
    ),
  ).toBeTruthy()
  // without Shot 4, nothing follows
  await fireEvent.click(screen.getByRole('checkbox', { name: 'Run Shot 4' }))
  expect(screen.queryByText(/^Afterwards/)).toBeNull()
})

it('an item validation refuses moves to "Will not run" with its first error', async () => {
  world.validate.mockImplementation(async (body: { workflow_path: string }) =>
    body.workflow_path === 'card'
      ? {
          valid: false,
          error: null,
          errors: [{ message: 'no variable named sed', path: null }],
          warnings: [],
        }
      : answer(PLANS.shot),
  )
  open()
  await within(runRow('s4')).findByText(/~9 min/)
  expect(runRow('title')).toBeNull()
  expect(document.querySelector('.wont li')?.textContent).toBe(
    'Title card: would be refused: no variable named sed',
  )
  expect(runButton().textContent).toBe('Run 1')
})

it('warns when every step is cached: the run would make nothing new', async () => {
  PLANS.shot = quote(9, { cached_steps: 2 })
  open()
  const warning = await within(runRow('s4')).findByText(ALL_CACHED)
  expect(warning.closest('p')?.className).toContain('warn')
  PLANS.shot = quote(9)
})

it('queues in plan order, each bound to its quote, then closes with a toast', async () => {
  const { onclose } = open()
  await within(runRow('title')).findByText(/~5.5 min/)
  await fireEvent.click(runButton())
  await waitFor(() => expect(onclose).toHaveBeenCalledWith(2))
  expect(world.submitJob.mock.calls.map(([body]) => body)).toEqual([
    {
      workflow_path: 'shot',
      arguments: {},
      request: 'r1',
      item: 's4',
      acknowledged_cost: PLANS.shot,
    },
    {
      workflow_path: 'card',
      arguments: {},
      request: 'r1',
      item: 'title',
      acknowledged_cost: PLANS.card,
    },
  ])
  expect(world.notify.success).toHaveBeenCalledWith('Queued 2 jobs')
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
})

it('a row whose price moved re-prices in place; the others queue, and the dialog stays', async () => {
  const { ApiError } = await import('../apiErrors')
  const { onclose } = open()
  await within(runRow('title')).findByText(/~5.5 min/)
  const acknowledge = { fingerprint: 'fp-new', minutes: 12, downloads: [] }
  world.submitJob.mockImplementationOnce(async () => {
    throw new ApiError("The run's shape changed", 409, undefined, {
      reason: 'fingerprint',
      plan: quote(12),
      acknowledge,
    })
  })
  await fireEvent.click(runButton())
  const moved = await within(runRow('s4')).findByRole('alert')
  expect(moved.textContent).toContain('The price moved. Now ~12 min.')
  // the other row went on, and shows its job
  const queued = await within(runRow('title')).findByRole('link', {
    name: 'Open job',
  })
  expect(queued.getAttribute('href')).toBe('#/ws/studio/jobs/job-title')
  expect(within(runRow('title')).getByText('queued')).toBeTruthy()
  expect(onclose).not.toHaveBeenCalled()
  expect(screen.getByRole('button', { name: 'Close' })).toBeTruthy()
  // nothing more to spend from the main button
  expect(runButton().disabled).toBe(true)

  await fireEvent.click(
    within(moved).getByRole('button', { name: 'Run at this price' }),
  )
  await waitFor(() => expect(onclose).toHaveBeenCalledWith(2))
  expect(world.submitJob.mock.calls.at(-1)?.[0]).toMatchObject({
    item: 's4',
    acknowledged_cost: acknowledge,
  })
  expect(world.notify.success).toHaveBeenCalledWith('Queued 2 jobs')
})

it('a refused submit shows on its row in plain words, and the dialog stays open', async () => {
  const { ApiError } = await import('../apiErrors')
  world.submitJob.mockRejectedValue(
    new ApiError('request r1 is done', 422, 'request_closed'),
  )
  const { onclose } = open('s4')
  await within(runRow('s4')).findByText(/~9 min/)
  await fireEvent.click(runButton())
  expect((await within(runRow('s4')).findByRole('alert')).textContent).toBe(
    'This request is closed. Reopen it to run its items.',
  )
  expect(onclose).not.toHaveBeenCalled()
  expect(world.notify.success).not.toHaveBeenCalled()
  await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  await waitFor(() => expect(onclose).toHaveBeenCalledWith(0))
})

it('a single item has one row, no checkbox and no total', async () => {
  open('s4')
  await screen.findByRole('dialog', { name: 'Run Shot 4' })
  await within(runRow('s4')).findByText(/~9 min/)
  expect(screen.queryByRole('checkbox')).toBeNull()
  expect(document.querySelector('.total')).toBeNull()
  expect(document.querySelectorAll('.runrow')).toHaveLength(1)
  expect(runButton().textContent).toBe('Run 1')
  expect(
    screen.getByText(
      'Afterwards: When this finishes, 1 more becomes ready: Shot 5. This page will not start it for you.',
    ),
  ).toBeTruthy()
})

it('another take of a done item says what stands as the result meanwhile', async () => {
  PLANS.shot = quote(9)
  world.validate.mockResolvedValue(answer(quote(3)))
  open('cast')
  await screen.findByRole('dialog', { name: 'Run another take of Cast sheet' })
  expect(
    screen.getByText(
      'Take 1 stays the accepted result until you accept another.',
    ),
  ).toBeTruthy()
})

it('cancelling queues nothing and drops the pricing still out', async () => {
  let signal: AbortSignal | undefined
  world.validate.mockImplementation(
    (_body: unknown, given?: AbortSignal) =>
      new Promise(() => {
        signal = given
      }),
  )
  const { onclose } = open()
  await fireEvent.click(await screen.findByRole('button', { name: 'Cancel' }))
  await waitFor(() => expect(onclose).toHaveBeenCalledWith(0))
  expect(signal?.aborted).toBe(true)
  expect(world.submitJob).not.toHaveBeenCalled()
})
