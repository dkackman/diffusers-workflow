import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import JobsPage from './JobsPage.svelte'
import { ApiError } from '../apiErrors'
import { formatExact } from '../format'
import { workspace } from '../workspaceState.svelte'
import type { JobSummary } from '../types'

const data = vi.hoisted(() => ({
  jobs: [] as unknown[],
  rerunRestartBatch: vi.fn(),
  confirm: vi.fn(),
  // Every listing asked for, as [workspace, limit, status, request, item]
  listed: [] as unknown[][],
  listError: null as Error | null,
  gate: null as Promise<void> | null,
  // What the server answers for "how many jobs are queued, anywhere"
  queued: 0,
}))
vi.mock('../api', async () => ({
  ApiError: (await import('../apiErrors')).ApiError,
  api: {
    rerunRestartBatch: data.rerunRestartBatch,
    // Filters by request/item the way the server does
    listJobs: (
      ws?: string,
      limit?: number,
      status?: string,
      request?: string,
      item?: string,
    ) => {
      data.listed.push([ws, limit, status, request, item])
      if (data.gate && limit !== 1)
        return data.gate.then(() => ({ jobs: [], total: 0 }))
      if (status === 'queued' && limit === 1)
        return Promise.resolve({ jobs: [], total: data.queued })
      if (data.listError) return Promise.reject(data.listError)
      const jobs = (data.jobs as JobSummary[]).filter(
        (j) =>
          (!request || j.request === request) && (!item || j.item === item),
      )
      return Promise.resolve({ jobs, total: jobs.length })
    },
    // The listing names the default workspace, as the server's always does:
    // a route parse that found it missing would redirect to overview
    listWorkspaces: () =>
      Promise.resolve({
        default: 'default',
        workspaces: [{ name: 'default', default: true }],
      }),
  },
}))
vi.mock('../confirm.svelte', () => ({ confirmDialog: data.confirm }))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

afterEach(() => {
  cleanup()
  data.listed = []
  data.listError = null
  data.gate = null
  data.queued = 0
  workspace.current = 'default'
  data.rerunRestartBatch.mockReset()
  data.confirm.mockReset()
})

const summary = (over: Partial<JobSummary>): JobSummary => ({
  id: 'j',
  workflow: 'w',
  workflow_name: null,
  status: 'succeeded',
  created_at: 1,
  started_at: 1,
  finished_at: 2,
  workspace: 'default',
  run_id: null,
  run_version: null,
  acknowledged: 'none',
  ...over,
})

// A history row written before the workflow was stored has a null
// workflow; the list must still render it, and every other row
it('lists a job recorded without a workflow name', async () => {
  data.jobs = [
    summary({ id: 'old-1', workflow: null, historical: true }),
    summary({ id: 'new-1', workflow: 'text-to-image' }),
  ]
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('text-to-image')).toBeTruthy())
  expect(screen.getByText(/old-1/)).toBeTruthy()
})

it('shows the card a job ran on, and nothing for one with no device', async () => {
  data.jobs = [
    summary({
      id: 'g1',
      workflow: 'on-gpu',
      device: 'cuda:1 NVIDIA GeForce RTX 3090',
    }),
    summary({ id: 'g2', workflow: 'no-gpu', device: null }),
  ]
  const { container } = render(JobsPage)
  await waitFor(() => expect(screen.getByText('no-gpu')).toBeTruthy())
  expect(screen.getByText('cuda:1 NVIDIA GeForce RTX 3090')).toBeTruthy()
  expect(container.textContent).not.toContain('cuda:0')
  expect(
    container.querySelectorAll('[title="the card this job ran on"]'),
  ).toHaveLength(1)
})

it('marks a rerun', async () => {
  data.jobs = [
    summary({ id: 'r1', workflow: 'a-rerun', rerun_of: 'abc' }),
    summary({ id: 'r2', workflow: 'fresh' }),
  ]
  const { container } = render(JobsPage)
  await waitFor(() => expect(screen.getByText('fresh')).toBeTruthy())
  expect(screen.getByTitle('a rerun of abc')).toBeTruthy()
  expect(container.querySelectorAll('[title^="a rerun of"]')).toHaveLength(1)
})

it('shows a queued job as waiting and a started one with its wait', async () => {
  data.jobs = [
    summary({ id: 'q1', workflow: 'wq', status: 'queued', started_at: null }),
    summary({ id: 's1', workflow: 'ws', created_at: 100, started_at: 160 }),
  ]
  const { container } = render(JobsPage)
  await waitFor(() => expect(screen.getByText('ws')).toBeTruthy())
  expect(container.textContent).toContain('waiting')
  expect(
    container.querySelector('[title$="waited 1m 00s in the queue"]'),
  ).toBeTruthy()
})

it('hovering a start time gives it to the second, with the finish', async () => {
  data.jobs = [
    summary({
      id: 's1',
      workflow: 'ws',
      started_at: 1700000000,
      finished_at: 1700000243,
    }),
  ]
  const { container } = render(JobsPage)
  await waitFor(() => expect(screen.getByText('ws')).toBeTruthy())
  const title = container.querySelector('.started')!.getAttribute('title')!
  expect(title).toContain(`started ${formatExact(1700000000)}`)
  expect(title).toContain(`finished ${formatExact(1700000243)}`)
})

it('reads its filters from the link and writes them back to it', async () => {
  location.hash = '#/ws/default/jobs?status=failed'
  data.jobs = [
    summary({ id: 'f1', workflow: 'broke', status: 'failed' }),
    summary({ id: 'ok', workflow: 'fine' }),
  ]
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('broke')).toBeTruthy())
  expect(screen.queryByText('fine')).toBeNull()
  const input = screen.getByPlaceholderText('filter by workflow…')
  await fireEvent.input(input, { target: { value: 'zzz' } })
  expect(location.hash).toBe('#/ws/default/jobs?status=failed&q=zzz')
  await fireEvent.click(screen.getByRole('button', { name: 'Clear filters' }))
  expect(location.hash).toBe('#/ws/default/jobs')
  expect(screen.getByText('fine')).toBeTruthy()
  location.hash = ''
})

it('navigating to the bare jobs route clears the filters', async () => {
  location.hash = '#/ws/default/jobs?status=failed&q=bro'
  data.jobs = [
    summary({ id: 'f1', workflow: 'broke', status: 'failed' }),
    summary({ id: 'ok', workflow: 'fine' }),
  ]
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('broke')).toBeTruthy())
  expect(screen.queryByText('fine')).toBeNull()
  // The Jobs nav link: same route view, so the page is not re-created
  location.hash = '#/ws/default/jobs'
  await waitFor(() => expect(screen.getByText('fine')).toBeTruthy())
  const status = screen.getByTitle('filter by status') as HTMLSelectElement
  expect(status.value).toBe('')
  const input = screen.getByPlaceholderText(
    'filter by workflow…',
  ) as HTMLInputElement
  expect(input.value).toBe('')
  expect(location.hash).toBe('#/ws/default/jobs')
  location.hash = ''
})

it('adds the hold reason to a queued row that has waiting_for', async () => {
  data.jobs = [
    summary({
      id: 'q0',
      workflow: 'plain',
      status: 'queued',
      started_at: null,
    }),
    summary({
      id: 'q1',
      workflow: 'ram',
      status: 'queued',
      started_at: null,
      waiting_for: 'waiting for host RAM: needs ~9 GB, ~2 GB free',
    }),
    summary({
      id: 'q2',
      workflow: 'busy',
      status: 'queued',
      started_at: null,
      waiting_for: 'queued behind 1 running job(s) on every card',
    }),
  ]
  const { container } = render(JobsPage)
  await waitFor(() => expect(screen.getByText('· host RAM')).toBeTruthy())
  expect(screen.getByText('· host RAM').getAttribute('title')).toBe(
    'waiting for host RAM: needs ~9 GB, ~2 GB free',
  )
  expect(screen.getByText('· cards busy').getAttribute('title')).toBe(
    'queued behind 1 running job(s) on every card',
  )
  expect(container.textContent?.match(/·/g)).toHaveLength(2)
})

it("puts a row's reason on its own line, with the full text in its title", async () => {
  const reason = 'server restart: queued, never started (a long tail of text)'
  data.jobs = [
    summary({ id: 'c1', workflow: 'gone', status: 'cancelled', reason }),
    summary({ id: 'ok', workflow: 'fine' }),
  ]
  const { container } = render(JobsPage)
  await waitFor(() => expect(screen.getByText('gone')).toBeTruthy())
  const line = container.querySelector('.reason') as HTMLElement
  expect(line.textContent).toBe(reason)
  expect(line.title).toBe(reason)
  // its own element, a sibling of the name, not text inside it
  expect(line.closest('.name')).toBeNull()
  expect(container.querySelectorAll('.reason')).toHaveLength(1)
})

const tombstone = (id: string, batch: string, created = 1) =>
  summary({
    id,
    workflow: 'w-' + id,
    status: 'cancelled',
    reason: 'server restart: queued, never started',
    restart_batch: batch,
    created_at: created,
  })

const banners = (container: HTMLElement) =>
  [...container.querySelectorAll('.restart')] as HTMLElement[]

it('shows a restart banner for tombstones nothing has rerun', async () => {
  data.jobs = [tombstone('t1', 'dead-a'), tombstone('t2', 'dead-a')]
  const { container } = render(JobsPage)
  await waitFor(() => expect(banners(container)).toHaveLength(1))
  expect(banners(container)[0].textContent).toContain(
    '2 jobs were cancelled by a server restart',
  )
  expect(screen.getByText('Rerun…')).toBeTruthy()
})

it('hides the banner once every tombstone has a rerun row', async () => {
  data.jobs = [
    tombstone('t1', 'dead-a'),
    tombstone('t2', 'dead-a'),
    summary({ id: 'n1', workflow: 'again-1', rerun_of: 't1' }),
    summary({ id: 'n2', workflow: 'again-2', rerun_of: 't2' }),
  ]
  const { container } = render(JobsPage)
  await waitFor(() => expect(screen.getByText('again-2')).toBeTruthy())
  expect(banners(container)).toHaveLength(0)
})

it('shows no banner with no tombstones', async () => {
  data.jobs = [
    summary({ id: 'c1', workflow: 'user-cancel', status: 'cancelled' }),
    summary({ id: 'ok', workflow: 'fine' }),
  ]
  const { container } = render(JobsPage)
  await waitFor(() => expect(screen.getByText('fine')).toBeTruthy())
  expect(banners(container)).toHaveLength(0)
})

it('shows one banner per batch, newest first', async () => {
  data.jobs = [
    tombstone('old1', 'dead-old', 1),
    tombstone('new1', 'dead-new', 10),
    tombstone('new2', 'dead-new', 11),
  ]
  const { container } = render(JobsPage)
  await waitFor(() => expect(banners(container)).toHaveLength(2))
  expect(banners(container).map((b) => b.dataset.batch)).toEqual([
    'dead-new',
    'dead-old',
  ])
  expect(banners(container)[0].textContent).toContain('2 jobs were')
  expect(banners(container)[1].textContent).toContain('1 job was')
})

it('hides the banner under a status filter that hides cancelled rows', async () => {
  location.hash = '#/ws/default/jobs?status=failed'
  data.jobs = [
    tombstone('t1', 'dead-a'),
    summary({ id: 'f1', workflow: 'broke', status: 'failed' }),
  ]
  const { container } = render(JobsPage)
  await waitFor(() => expect(screen.getByText('broke')).toBeTruthy())
  expect(banners(container)).toHaveLength(0)
  location.hash = ''
})

const preview = {
  server_instance: 'dead-a',
  queued: [],
  would_queue: [
    {
      id: 't1',
      workflow: 'w-t1',
      workspace: 'default',
      estimate: {
        basis: 'observed',
        minutes: 2,
        device: 'mps',
        measured_on: 'Apple M2',
      },
    },
    { id: 't2', workflow: 'w-t2', workspace: 'default', estimate: null },
  ],
  skipped: [{ id: 't3', reason: 'its workspace is gone' }],
}

it('previews on click with {} once, and the poll never calls the route', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  try {
    data.jobs = [tombstone('t1', 'dead-a'), tombstone('t2', 'dead-a')]
    data.rerunRestartBatch.mockResolvedValue(preview)
    data.confirm.mockResolvedValue(false)
    const { container } = render(JobsPage)
    await waitFor(() => expect(banners(container)).toHaveLength(1))
    // several poll ticks with the banner up: no preview
    await vi.advanceTimersByTimeAsync(10_000)
    expect(data.rerunRestartBatch).not.toHaveBeenCalled()
    await fireEvent.click(screen.getByText('Rerun…'))
    await waitFor(() => expect(data.confirm).toHaveBeenCalledTimes(1))
    expect(data.rerunRestartBatch).toHaveBeenCalledTimes(1)
    // any member names its batch; the call carries no acknowledgement
    const [member, ...rest] = data.rerunRestartBatch.mock.calls[0]
    expect(['t1', 't2']).toContain(member)
    expect(rest).toEqual([])
    const message = data.confirm.mock.calls[0][0] as string
    expect(message).toContain('w-t1 (t1, default) - ~2 min on mps')
    expect(message).toContain('w-t2 (t2, default) - no estimate')
    expect(message).toContain('total ~2 min, plus 1 with no figure')
    expect(message).toContain('t3: its workspace is gone')
    await vi.advanceTimersByTimeAsync(10_000)
    expect(data.rerunRestartBatch).toHaveBeenCalledTimes(1)
  } finally {
    vi.useRealTimers()
  }
})

it('confirming sends acknowledged_cost: true; cancelling sends nothing more', async () => {
  data.jobs = [tombstone('t1', 'dead-a'), tombstone('t2', 'dead-a')]
  data.rerunRestartBatch.mockResolvedValue(preview)
  data.confirm.mockResolvedValue(false)
  const { container } = render(JobsPage)
  await waitFor(() => expect(banners(container)).toHaveLength(1))
  await fireEvent.click(screen.getByText('Rerun…'))
  await waitFor(() => expect(data.confirm).toHaveBeenCalledTimes(1))
  // cancel: only the preview went out
  expect(data.rerunRestartBatch).toHaveBeenCalledTimes(1)
  const member = data.rerunRestartBatch.mock.calls[0][0]
  expect(data.rerunRestartBatch.mock.calls).toEqual([[member]])

  data.rerunRestartBatch.mockReset()
  data.rerunRestartBatch.mockResolvedValueOnce(preview).mockResolvedValueOnce({
    ...preview,
    would_queue: [],
    queued: [
      { id: 'n1', rerun_of: 't1', workflow: 'w-t1', workspace: 'default' },
      { id: 'n2', rerun_of: 't2', workflow: 'w-t2', workspace: 'default' },
    ],
  })
  data.confirm.mockResolvedValue(true)
  await waitFor(() =>
    expect((screen.getByText('Rerun…') as HTMLButtonElement).disabled).toBe(
      false,
    ),
  )
  await fireEvent.click(screen.getByText('Rerun…'))
  await waitFor(() => expect(data.rerunRestartBatch).toHaveBeenCalledTimes(2))
  expect(data.rerunRestartBatch.mock.calls).toEqual([[member], [member, true]])
  const { notify } = await import('../toast')
  await waitFor(() =>
    expect(notify.success).toHaveBeenCalledWith(
      'Queued 2; skipped 1: t3 (its workspace is gone)',
    ),
  )
})

it('an empty preview says so, with Close only, and hides the banner', async () => {
  data.jobs = [tombstone('t1', 'dead-a')]
  data.rerunRestartBatch.mockResolvedValue({
    server_instance: 'dead-a',
    queued: [],
    would_queue: [],
    skipped: [{ id: 't1', reason: 'already rerun as n9' }],
  })
  data.confirm.mockResolvedValue(true)
  const { container } = render(JobsPage)
  await waitFor(() => expect(banners(container)).toHaveLength(1))
  await fireEvent.click(screen.getByText('Rerun…'))
  await waitFor(() => expect(banners(container)).toHaveLength(0))
  const [message, options] = data.confirm.mock.calls[0]
  expect(message).toContain('Nothing is left to rerun')
  expect(options).toMatchObject({ confirmLabel: 'Close', alert: true })
  // only the preview: nothing was queued
  expect(data.rerunRestartBatch.mock.calls).toEqual([['t1']])
})

const linked = () => [
  summary({ id: 'a', workflow: 'wf-a', request: 'r1', item: 'hero' }),
  summary({ id: 'b', workflow: 'wf-b', request: 'r1', item: 'alt' }),
  summary({ id: 'c', workflow: 'wf-c' }),
]

it('chips a linked job with request › item and leaves an unlinked one bare', async () => {
  location.hash = '#/ws/default/jobs'
  data.jobs = linked()
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('wf-c')).toBeTruthy())
  expect(screen.getByText('r1 › hero')).toBeTruthy()
  expect(screen.getAllByText(/r1 ›/)).toHaveLength(2)
})

it('narrows to a request, then to one item, and clearing restores the list', async () => {
  location.hash = '#/ws/default/jobs?req=r1'
  data.jobs = linked()
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('wf-a')).toBeTruthy())
  expect(screen.queryByText('wf-c')).toBeNull()
  expect(screen.getByText('wf-b')).toBeTruthy()
  const chip = screen.getByTestId('req-filter')
  expect(chip.querySelector('a')?.getAttribute('href')).toBe(
    '#/ws/default/requests/r1',
  )
  cleanup()

  location.hash = '#/ws/default/jobs?req=r1%2Fhero'
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('wf-a')).toBeTruthy())
  expect(screen.queryByText('wf-b')).toBeNull()

  await fireEvent.click(screen.getByLabelText('remove the request filter'))
  await waitFor(() => expect(screen.getByText('wf-c')).toBeTruthy())
  expect(screen.getByText('wf-b')).toBeTruthy()
  expect(screen.queryByTestId('req-filter')).toBeNull()
})

it('a req that arrives on the hash refetches at once, not on the next tick', async () => {
  location.hash = '#/ws/default/jobs'
  data.jobs = linked()
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('wf-c')).toBeTruthy())
  location.hash = '#/ws/default/jobs?req=r1%2Falt'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  await waitFor(() => expect(screen.queryByText('wf-a')).toBeNull(), {
    timeout: 1000,
  })
  expect(screen.getByText('wf-b')).toBeTruthy()
  expect(screen.queryByText('wf-c')).toBeNull()
})

it('on the all-jobs view a request filter is read in one workspace, and the chip links there', async () => {
  workspace.current = 'studio'
  location.hash = '#/status?req=r1'
  data.jobs = linked()
  render(JobsPage, { scope: 'all' })
  await waitFor(() => expect(screen.getByText('wf-a')).toBeTruthy())
  expect(data.listed.every(([ws]) => ws === 'studio')).toBe(true)
  expect(
    screen.getByTestId('req-filter').querySelector('a')?.getAttribute('href'),
  ).toBe('#/ws/studio/requests/r1')
  // and the link says which workspace it was read in
  await waitFor(() => expect(location.hash).toContain('ws=studio'))
})

it('on the all-jobs view a picked workspace scopes the request filter', async () => {
  workspace.current = 'studio'
  location.hash = '#/status?req=r1&ws=shots'
  data.jobs = linked()
  render(JobsPage, { scope: 'all' })
  await waitFor(() => expect(screen.getByText('wf-a')).toBeTruthy())
  expect(data.listed.every(([ws]) => ws === 'shots')).toBe(true)
  expect(
    screen.getByTestId('req-filter').querySelector('a')?.getAttribute('href'),
  ).toBe('#/ws/shots/requests/r1')
})

it("shows a refused filter as the server's message, over no rows", async () => {
  location.hash = '#/ws/default/jobs'
  data.jobs = linked()
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('wf-c')).toBeTruthy())
  data.listError = new ApiError("Request 'R!' must match ^[a-z0-9]", 422)
  location.hash = '#/ws/default/jobs?req=R!'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  await waitFor(() => expect(screen.getByText(/must match/)).toBeTruthy())
  expect(screen.queryByText(/Could not reach the server/)).toBeNull()
  expect(screen.queryByText('wf-c')).toBeNull()
})

const moves = (id: string) =>
  ['front', 'up', 'down'].filter((way) =>
    screen.queryByLabelText(
      way === 'front'
        ? `move ${id} to the front of the queue`
        : `move ${id} ${way} one place`,
    ),
  )

it("a request's queued job mid-queue can move both ways, though it is the only row", async () => {
  location.hash = '#/ws/default/jobs?req=r1'
  data.queued = 5
  data.jobs = [
    summary({ id: 'q3', status: 'queued', request: 'r1', queue_position: 2 }),
    summary({ id: 'other', status: 'queued', queue_position: 4 }),
  ]
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('#3')).toBeTruthy())
  await waitFor(() => expect(moves('q3')).toEqual(['front', 'up', 'down']))
})

it('the last job of the whole queue offers no down', async () => {
  location.hash = '#/ws/default/jobs?req=r1'
  data.queued = 5
  data.jobs = [
    summary({ id: 'q5', status: 'queued', request: 'r1', queue_position: 4 }),
  ]
  render(JobsPage)
  await waitFor(() => expect(moves('q5')).toEqual(['front', 'up']))
})

it('asks for the queue length only while a queued job is on screen', async () => {
  location.hash = '#/ws/default/jobs'
  data.jobs = linked()
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('wf-c')).toBeTruthy())
  expect(data.listed.some(([, , status]) => status === 'queued')).toBe(false)
})

it('loading shows a skeleton, not "No jobs yet"', async () => {
  let open: () => void = () => {}
  data.gate = new Promise<void>((resolve) => (open = resolve))
  render(JobsPage)
  expect(screen.getByLabelText('loading jobs')).toBeTruthy()
  expect(screen.queryByText(/No jobs yet/)).toBeNull()
  open()
  await screen.findByText(/No jobs yet/)
  expect(screen.queryByLabelText('loading jobs')).toBeNull()
})

it('a listing that fails on first load is a load failure with Retry', async () => {
  data.listError = new TypeError('Failed to fetch')
  render(JobsPage)
  await screen.findByText(/Couldn't load jobs: Can't reach the server/)
  expect(screen.queryByText(/No jobs yet/)).toBeNull()
  data.listError = null
  data.jobs = [summary({ id: 'x', workflow: 'back-again' })]
  await fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await screen.findByText('back-again')
})

it('a status filter that matches nothing says so, and Clear filters brings the rows back', async () => {
  data.jobs = [summary({ id: 'a', workflow: 'wf-a' })]
  render(JobsPage)
  await screen.findByText('wf-a')
  await fireEvent.change(screen.getByTitle('filter by status'), {
    target: { value: 'failed' },
  })
  await screen.findByText('Nothing matches those filters.')
  await fireEvent.click(screen.getByRole('button', { name: 'Clear filters' }))
  await screen.findByText('wf-a')
})
