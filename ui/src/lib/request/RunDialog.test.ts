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
import { api } from '../test/apiMock'
import { attempt, doneItem, item, request } from './fixtures'
import { planState } from './itemState'
import { listening } from './listening.svelte'
import { clockTime } from './reviewText'
import { ALL_CACHED, heldPlan, itemPlan, readyPlan } from './runPlan'
import type { LedgerReview, Plan } from '../types'

const world = vi.hoisted(() => ({
  notify: { error: vi.fn(), success: vi.fn(), undo: vi.fn() },
}))
vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
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
  api.validate.mockImplementation(async (body) =>
    answer(PLANS[(body as { workflow_path: string }).workflow_path] ?? null),
  )
  api.submitJob.mockImplementation(async (body) => ({
    id: `job-${(body as { item: string }).item}`,
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
    name: /^(Run \d+ jobs?|Queuing…)$/,
  }) as HTMLButtonElement

it('starts on Cancel, and cannot spend until every row is priced', async () => {
  let land: (value: unknown) => void = () => {}
  api.validate.mockImplementation(
    (body) =>
      new Promise((resolve) => {
        if ((body as { workflow_path: string }).workflow_path === 'card')
          land = resolve
        else resolve(answer(PLANS.shot))
      }),
  )
  open()
  const dialog = await screen.findByRole('dialog', {
    name: 'Run what can run now',
  })
  await waitFor(() =>
    expect(document.activeElement).toBe(
      within(dialog).getByRole('button', { name: 'Cancel' }),
    ),
  )
  // one priced, one still out
  await within(runRow('s4')).findByText(/about 9 min on cuda/)
  expect(
    within(runRow('title')).getByRole('status', { name: 'pricing Title card' }),
  ).toBeTruthy()
  expect(runButton().disabled).toBe(true)
  land(answer(PLANS.card))
  await within(runRow('title')).findByText(/about 5.5 min on cuda/)
  expect(runButton().disabled).toBe(false)
  expect(runButton().textContent).toBe('Run 2 jobs')
  expect(api.submitJob).not.toHaveBeenCalled()
})

it('prices each item with its workflow, and says what the plan had estimated', async () => {
  open()
  await within(runRow('s4')).findByText(/about 9 min on cuda/)
  expect(api.validate.mock.calls.map(([body]) => body)).toEqual([
    { workflow_path: 'shot', arguments: {}, workspace: 'studio' },
    { workflow_path: 'card', arguments: {}, workspace: 'studio' },
  ])
  const s4 = within(runRow('s4'))
  expect(s4.getByText('shot')).toBeTruthy()
  // the plan said 4, the quote is 9: more than a quarter off
  expect(s4.getByText('the plan said about 4 min')).toBeTruthy()
  expect(within(runRow('title')).queryByText(/the plan said/)).toBeNull()
  expect(document.querySelector('.total')?.textContent).toBe(
    'total about 14.5 min',
  )
})

it('the total follows the ticked rows, and counts one with no figure', async () => {
  PLANS.card = quote(null)
  open()
  await within(runRow('title')).findByText(
    'No time estimate for this workflow.',
  )
  const why = runRow('title').querySelector('details') as HTMLDetailsElement
  expect(why.querySelector('summary')?.textContent).toBe(
    "Why there's no estimate",
  )
  expect(why.querySelector('pre')?.textContent).toBe('no measured cost')
  expect(why.open).toBe(false)
  expect(document.querySelector('.total')?.textContent).toBe(
    'total about 9 min, plus 1 with no figure',
  )
  await fireEvent.click(screen.getByRole('checkbox', { name: 'Run Shot 4' }))
  expect(document.querySelector('.total')?.textContent).toBe('total: no figure')
  expect(runButton().textContent).toBe('Run 1 job')
  PLANS.card = quote(5.5)
})

it('lists what will not run and why, and what becomes ready afterwards', async () => {
  open()
  await within(runRow('s4')).findByText(/about 9 min/)
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
  api.validate.mockImplementation(async (body) =>
    (body as { workflow_path: string }).workflow_path === 'card'
      ? {
          valid: false,
          error: null,
          errors: [{ message: 'no variable named sed', path: null }],
          warnings: [],
        }
      : answer(PLANS.shot),
  )
  open()
  await within(runRow('s4')).findByText(/about 9 min/)
  expect(runRow('title')).toBeNull()
  const li = document.querySelector('.wont li') as HTMLElement
  expect(li.textContent).toContain(
    "Title card: A job would be refused: 1 issue to fix. Ask the agent to fix the item's workflow or arguments.",
  )
  expect(li.querySelector('details pre')?.textContent).toBe(
    'no variable named sed',
  )
  expect(runButton().textContent).toBe('Run 1 job')
})

it('the button counts jobs', async () => {
  open()
  await within(runRow('title')).findByText(/about 5.5 min/)
  expect(runButton().textContent).toBe('Run 2 jobs')
  await fireEvent.click(screen.getByRole('checkbox', { name: 'Run Shot 4' }))
  expect(runButton().textContent).toBe('Run 1 job')
  await fireEvent.click(
    screen.getByRole('checkbox', { name: 'Run Title card' }),
  )
  expect(runButton().textContent).toBe('Run 0 jobs')
  expect(runButton().disabled).toBe(true)
})

it('a gated model is said on its row in warn tone, and the row stays tickable', async () => {
  PLANS.shot = quote(9, {
    downloads_required: [
      { repo: 'org/m', gb: null, gated: true, access_blocked: true },
    ] as Plan['downloads_required'],
  })
  open()
  const text = await within(runRow('s4')).findByText(
    /^This model needs Hugging Face access\./,
  )
  const line = text.closest('.verdict') as HTMLElement
  expect(line.classList.contains('warn')).toBe(true)
  expect(line.textContent).toContain(
    'Accept its terms on Hugging Face, then sign in on the server with hf auth login.',
  )
  const box = screen.getByRole('checkbox', { name: 'Run Shot 4' })
  expect((box as HTMLInputElement).checked).toBe(true)
  expect((box as HTMLInputElement).disabled).toBe(false)
  PLANS.shot = quote(9)
})

it('failed items sit under their heading, unticked', async () => {
  const state = planState(
    request([
      item('s4', { title: 'Shot 4', ready: true, workflow: 'shot' }),
      item('f1', {
        title: 'Failed one',
        status: 'failed',
        workflow: 'card',
        arguments: { seed: 7 },
        attempts: [attempt('jf', 'failed')],
      }),
    ]),
  )
  render(RunDialog, {
    run: readyPlan(state),
    plan: state,
    workspace: 'studio',
    requestId: 'r1',
    onclose: vi.fn(),
  })
  await within(runRow('f1')).findByText(/about 5.5 min/)
  const group = screen.getByRole('list', { name: 'Failed before' })
  expect(
    within(group.parentElement as HTMLElement).getByRole('heading').textContent,
  ).toBe('Failed before. Tick one to run it again.')
  expect(group.contains(runRow('f1'))).toBe(true)
  expect(group.contains(runRow('s4'))).toBe(false)
  const box = screen.getByRole('checkbox', {
    name: 'Run Failed one',
  }) as HTMLInputElement
  expect(box.checked).toBe(false)
  expect(runButton().textContent).toBe('Run 1 job')
  await fireEvent.click(box)
  expect(runButton().textContent).toBe('Run 2 jobs')
  await fireEvent.click(runButton())
  await waitFor(() => expect(api.submitJob).toHaveBeenCalledTimes(2))
  // its stored arguments, exactly as the ledger holds them
  expect(api.submitJob.mock.calls[1][0]).toMatchObject({
    item: 'f1',
    workflow_path: 'card',
    arguments: { seed: 7 },
  })
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
  await within(runRow('title')).findByText(/about 5.5 min/)
  await fireEvent.click(runButton())
  await waitFor(() => expect(onclose).toHaveBeenCalledWith(2))
  expect(api.submitJob.mock.calls.map(([body]) => body)).toEqual([
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
  await within(runRow('title')).findByText(/about 5.5 min/)
  const acknowledge = { fingerprint: 'fp-new', minutes: 12, downloads: [] }
  api.submitJob.mockImplementationOnce(async () => {
    throw new ApiError("The run's shape changed", 409, undefined, {
      reason: 'fingerprint',
      plan: quote(12),
      acknowledge,
    })
  })
  await fireEvent.click(runButton())
  const moved = await within(runRow('s4')).findByRole('alert')
  expect(moved.textContent).toContain('The price moved. Now about 12 min.')
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
  expect(api.submitJob.mock.calls.at(-1)?.[0]).toMatchObject({
    item: 's4',
    acknowledged_cost: acknowledge,
  })
  expect(world.notify.success).toHaveBeenCalledWith('Queued 2 jobs')
})

it('a refused submit shows on its row in plain words, and the dialog stays open', async () => {
  const { ApiError } = await import('../apiErrors')
  api.submitJob.mockRejectedValue(
    new ApiError('request r1 is done', 422, 'request_closed'),
  )
  const { onclose } = open('s4')
  await within(runRow('s4')).findByText(/about 9 min/)
  await fireEvent.click(runButton())
  const alert = await within(runRow('s4')).findByRole('alert')
  expect(alert.querySelector('p')?.textContent).toBe(
    'This request is closed. Reopen it to run its items.',
  )
  expect(onclose).not.toHaveBeenCalled()
  expect(world.notify.success).not.toHaveBeenCalled()
  await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  await waitFor(() => expect(onclose).toHaveBeenCalledWith(0))
})

it('a submit the server failed on leads with plain words, its text folded behind Details', async () => {
  const { ApiError } = await import('../apiErrors')
  const raw = 'worker crashed: CUDA error: out of memory at step 3'
  api.submitJob.mockRejectedValue(new ApiError(raw, 500))
  open('s4')
  await within(runRow('s4')).findByText(/about 9 min/)
  await fireEvent.click(runButton())
  const alert = await within(runRow('s4')).findByRole('alert')
  expect(alert.querySelector('p')?.textContent).toBe('The server hit an error')
  expect(alert.querySelector('details pre')?.textContent).toBe(raw)
  expect(alert.querySelector('details')?.open).toBe(false)
})

it('a single item has one row, no checkbox and no total', async () => {
  open('s4')
  await screen.findByRole('dialog', { name: 'Run Shot 4' })
  await within(runRow('s4')).findByText(/about 9 min/)
  expect(screen.queryByRole('checkbox')).toBeNull()
  expect(document.querySelector('.total')).toBeNull()
  expect(document.querySelectorAll('.runrow')).toHaveLength(1)
  expect(runButton().textContent).toBe('Run 1 job')
  expect(
    screen.getByText(
      'Afterwards: When this finishes, 1 more becomes ready: Shot 5. This page will not start it for you.',
    ),
  ).toBeTruthy()
})

it('another take of a done item says what stands as the result meanwhile', async () => {
  PLANS.shot = quote(9)
  api.validate.mockResolvedValue(answer(quote(3)))
  open('cast')
  await screen.findByRole('dialog', { name: 'Run another take of Cast sheet' })
  expect(
    screen.getByText(
      'Take 1 stays the accepted result until you accept another.',
    ),
  ).toBeTruthy()
})

it('closing while the first job is being sent sends no more, and says when that one lands', async () => {
  let land = () => {}
  api.submitJob.mockImplementation(
    (body) =>
      new Promise((resolve) => {
        land = () => resolve({ id: `job-${(body as { item: string }).item}` })
      }),
  )
  const { onclose } = open()
  await within(runRow('title')).findByText(/about 5.5 min/)
  await fireEvent.click(runButton())
  await waitFor(() => expect(api.submitJob).toHaveBeenCalledTimes(1))
  await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  await waitFor(() => expect(onclose).toHaveBeenCalled())
  const [queued, landing] = onclose.mock.calls[0]
  expect(queued).toBe(0)
  land()
  expect(await landing).toBe(1)
  expect(api.submitJob).toHaveBeenCalledTimes(1)
})

it('cancelling queues nothing and drops the pricing still out', async () => {
  let signal: AbortSignal | undefined
  api.validate.mockImplementation(
    (_body: unknown, given?: AbortSignal) =>
      new Promise(() => {
        signal = given
      }),
  )
  const { onclose } = open()
  await fireEvent.click(await screen.findByRole('button', { name: 'Cancel' }))
  await waitFor(() => expect(onclose).toHaveBeenCalledWith(0))
  expect(signal?.aborted).toBe(true)
  expect(api.submitJob).not.toHaveBeenCalled()
})

const UNTIL = '2026-11-21T15:30:00+00:00'
function openHeld(state: LedgerReview['state'], listens: boolean) {
  listening('studio').listening = listens
  const held = planState(
    request([
      item('title', {
        title: 'Title card',
        ready: true,
        workflow: 'card',
        review: { until: UNTIL, state, seconds_left: 0 },
      }),
      item('s4', { title: 'Shot 4', ready: true, workflow: 'shot' }),
    ]),
  )
  render(RunDialog, {
    run: heldPlan(held, ['title']),
    plan: held,
    workspace: 'studio',
    requestId: 'r1',
    onclose: vi.fn(),
  })
}

it('says on a held row that running it now is the go-ahead', async () => {
  openHeld('holding', true)
  await screen.findByRole('dialog', { name: 'Run what can run now' })
  const line = runRow('title').querySelector('.held') as HTMLElement
  expect(line.textContent?.trim()).toBe(
    `Waiting for your OK until ${clockTime(UNTIL)}. Running it now is your go-ahead.`,
  )
  expect(line.classList.contains('warn')).toBe(false)
  // only the held row is in this run
  expect(runRow('s4')).toBeNull()
  listening('studio').listening = false
})

it('warns on a held row whose review is up while a session listens', async () => {
  openHeld('due', true)
  await screen.findByRole('dialog', { name: 'Run what can run now' })
  const line = runRow('title').querySelector('.held') as HTMLElement
  expect(line.textContent?.trim()).toBe(
    'Review time is up and the agent is connected: it may be starting this now.',
  )
  expect(line.classList.contains('warn')).toBe(true)
  listening('studio').listening = false
})

it('has no held line on a row that is not held', async () => {
  open()
  await screen.findByRole('dialog', { name: 'Run what can run now' })
  expect(runRow('s4').querySelector('.held')).toBeNull()
})

it('unticks a row the live plan has moved past, says why, and never sends it', async () => {
  const items = (over: Parameters<typeof item>[1] = {}) =>
    planState(
      request([
        item('s4', { title: 'Shot 4', ready: true, workflow: 'shot', ...over }),
        item('title', { title: 'Title card', ready: true, workflow: 'card' }),
      ]),
    )
  const state = items()
  const props = {
    run: readyPlan(state),
    plan: state,
    workspace: 'studio',
    requestId: 'r1',
    onclose: vi.fn(),
  }
  const view = render(RunDialog, props)
  await within(runRow('title')).findByText(/about 5.5 min/)
  expect(runButton().textContent).toBe('Run 2 jobs')

  // a poll: the agent queued Shot 4 meanwhile
  await view.rerender({
    ...props,
    plan: items({
      ready: false,
      status: 'queued',
      attempts: [attempt('agent', 'queued')],
    }),
  })
  const box = screen.getByRole('checkbox', {
    name: 'Run Shot 4',
  }) as HTMLInputElement
  expect(box.checked).toBe(false)
  expect(box.disabled).toBe(true)
  expect(runRow('s4').querySelector('.stale')?.textContent?.trim()).toBe(
    'A job was queued for it after this opened.',
  )
  expect(runButton().textContent).toBe('Run 1 job')
  expect(document.querySelector('.total')?.textContent).toBe(
    'total about 5.5 min',
  )
  await fireEvent.click(runButton())
  await waitFor(() => expect(api.submitJob).toHaveBeenCalledTimes(1))
  expect(api.submitJob.mock.calls[0][0]).toMatchObject({ item: 'title' })
})

it('a single item rewritten while its dialog is open cannot be run', async () => {
  const items = (seed: number) =>
    planState(
      request([
        item('s4', {
          title: 'Shot 4',
          ready: true,
          workflow: 'shot',
          arguments: { seed },
        }),
      ]),
    )
  const state = items(1)
  const props = {
    run: itemPlan(state, 's4'),
    plan: state,
    workspace: 'studio',
    requestId: 'r1',
    onclose: vi.fn(),
  }
  const view = render(RunDialog, props)
  await within(runRow('s4')).findByText(/about 9 min/)
  expect(runButton().disabled).toBe(false)
  await view.rerender({ ...props, plan: items(2) })
  expect(runRow('s4').querySelector('.stale')?.textContent).toContain(
    'Its workflow or arguments changed after this opened.',
  )
  expect(runButton().disabled).toBe(true)
})
