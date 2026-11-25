import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import JobsPage from './JobsPage.svelte'
import { ApiError, NetworkError } from '../apiErrors'
import { formatExact, formatRelative } from '../format'
import { DEFAULT_WORKSPACE, workspace } from '../workspaceState.svelte'
import type { JobSummary } from '../types'
import { api } from '../test/apiMock'
import { quiet, useQuietClock } from '../test/quiet'

useQuietClock()

const data = vi.hoisted(() => ({
  jobs: [] as unknown[],
  confirm: vi.fn(),
  // Every listing asked for, as [workspace, limit, status, request, item]
  listed: [] as unknown[][],
  listError: null as Error | null,
  gate: null as Promise<void> | null,
  // What the server answers for "how many jobs are queued, anywhere"
  queued: 0,
  // job id -> prompt, as `prompts=true` answers; the gallery's files; a
  // running job's progress by id
  prompts: {} as Record<string, string>,
  files: [] as unknown[],
  progress: {} as Record<string, unknown>,
  // Whether each listing asked for prompts
  asked: [] as boolean[],
  // Whether a listing keeps only the workspace it names, as the server does
  byWorkspace: false,
}))
vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
// Filters by request/item the way the server does
api.listJobs.mockImplementation(
  (
    ws?: string,
    limit?: number,
    status?: string,
    request?: string,
    item?: string,
    opts?: { prompts?: boolean },
  ) => {
    data.listed.push([ws, limit, status, request, item])
    if (limit !== 1) data.asked.push(!!opts?.prompts)
    if (data.gate && limit !== 1)
      return data.gate.then(() => ({ jobs: [], total: 0 }))
    if (status === 'queued' && limit === 1)
      return Promise.resolve({ jobs: [], total: data.queued })
    if (data.listError) return Promise.reject(data.listError)
    const states = status?.split(',')
    const matched = (data.jobs as JobSummary[]).filter(
      (j) =>
        (!data.byWorkspace || !ws || j.workspace === ws) &&
        (!request || j.request === request) &&
        (!item || j.item === item) &&
        (!states || states.includes(j.status)),
    )
    // The server keeps the newest `limit` and counts what matched before
    const jobs = limit === undefined ? matched : matched.slice(-limit)
    return Promise.resolve({
      jobs,
      total: matched.length,
      ...(opts?.prompts ? { prompts: data.prompts } : {}),
    })
  },
)
api.gallery.mockImplementation(() =>
  Promise.resolve({ workspace: 'default', files: data.files }),
)
api.getJob.mockImplementation((id: string) =>
  Promise.resolve({ progress: data.progress[id] }),
)
// The request picker's list, in whichever workspace it is asked for
api.listRequests.mockImplementation((ws: string) =>
  Promise.resolve({
    workspace: ws,
    requests: [{ id: 'r1', title: 'Launch posters' }],
  }),
)
// The listing names the default workspace, as the server's always does:
// a route parse that found it missing would redirect to overview
api.listWorkspaces.mockImplementation(() =>
  Promise.resolve({
    default: 'default',
    workspaces: [{ name: 'default', default: true }],
  }),
)
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))
vi.mock('../confirm.svelte', () => ({ confirmDialog: data.confirm }))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

afterEach(() => {
  cleanup()
  data.listed = []
  data.listError = null
  data.gate = null
  data.queued = 0
  data.prompts = {}
  data.files = []
  data.progress = {}
  data.asked = []
  data.byWorkspace = false
  workspace.current = 'default'
  api.rerunRestartBatch.mockReset()
  api.moveJob.mockReset()
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
  // the row shows the start relative (a list), and the time carries the hover
  const time = container.querySelector('.started time')!
  expect(time.textContent?.trim()).toBe(
    formatRelative(1700000000, Date.now() / 1000),
  )
  expect(time.getAttribute('title')).toBe(title)
})

it('rows and the name filter use the file name', async () => {
  location.hash = '#/ws/default/jobs'
  data.jobs = [
    summary({ id: 'c1', workflow: 'E2ECard', workflow_name: 'e2e-card' }),
    summary({ id: 'o1', workflow: 'Other', workflow_name: 'models/other' }),
  ]
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('e2e-card')).toBeTruthy())
  expect(screen.queryByText('E2ECard')).toBeNull()
  expect(screen.getByText('other')).toBeTruthy()
  const input = screen.getByPlaceholderText(
    'search names, prompts, notes and ids…',
  )
  await fireEvent.input(input, { target: { value: 'e2e-c' } })
  expect(screen.getByText('e2e-card')).toBeTruthy()
  expect(screen.queryByText('other')).toBeNull()
  location.hash = ''
})

it('titles the version badge with its ordinal', async () => {
  data.jobs = [summary({ id: 'v1', workflow_name: 'e2e-card', run_version: 3 })]
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('v3')).toBeTruthy())
  expect(screen.getByText('v3').getAttribute('title')).toBe(
    '3rd job of this workflow in this folder',
  )
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
  const input = screen.getByPlaceholderText(
    'search names, prompts, notes and ids…',
  )
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
    'search names, prompts, notes and ids…',
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
    'waiting for host RAM: needs about 9 GB, about 2 GB free',
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
      workflow: 'WT1Id',
      workflow_name: 'models/w-t1',
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
    api.rerunRestartBatch.mockResolvedValue(preview)
    data.confirm.mockResolvedValue(false)
    const { container } = render(JobsPage)
    await waitFor(() => expect(banners(container)).toHaveLength(1))
    // several poll ticks with the banner up: no preview
    await vi.advanceTimersByTimeAsync(10_000)
    expect(api.rerunRestartBatch).not.toHaveBeenCalled()
    await fireEvent.click(screen.getByText('Rerun…'))
    await waitFor(() => expect(data.confirm).toHaveBeenCalledTimes(1))
    expect(api.rerunRestartBatch).toHaveBeenCalledTimes(1)
    // any member names its batch; the call carries no acknowledgement
    const [member, ...rest] = api.rerunRestartBatch.mock.calls[0]
    expect(['t1', 't2']).toContain(member)
    expect(rest).toEqual([])
    const message = data.confirm.mock.calls[0][0] as string
    expect(message).toContain('w-t1 (t1, default) - about 2 min on mps')
    expect(message).not.toContain('WT1Id')
    expect(message).toContain('w-t2 (t2, default) - no estimate')
    expect(message).toContain('total about 2 min, plus 1 with no figure')
    expect(message).toContain('t3: its workspace is gone')
    await vi.advanceTimersByTimeAsync(10_000)
    expect(api.rerunRestartBatch).toHaveBeenCalledTimes(1)
  } finally {
    vi.useRealTimers()
  }
})

it("one batch's rerun leaves the other banners", async () => {
  data.jobs = [tombstone('t1', 'dead-a', 2), tombstone('t2', 'dead-b', 1)]
  api.rerunRestartBatch.mockResolvedValue(preview)
  let answer: (ok: boolean) => void = () => {}
  data.confirm.mockReturnValue(new Promise<boolean>((r) => (answer = r)))
  const { container } = render(JobsPage)
  await waitFor(() => expect(banners(container)).toHaveLength(2))
  const buttons = () =>
    banners(container).map((b) => b.querySelector('button') as HTMLElement)
  await fireEvent.click(buttons()[0])
  await waitFor(() =>
    expect(buttons()[0].getAttribute('aria-busy')).toBe('true'),
  )
  expect(buttons()[1].getAttribute('aria-busy')).toBe('false')
  expect((buttons()[1] as HTMLButtonElement).disabled).toBe(false)
  answer(false)
  await waitFor(() =>
    expect(buttons()[0].getAttribute('aria-busy')).toBe('false'),
  )
})

it('confirming sends acknowledged_cost: true; cancelling sends nothing more', async () => {
  data.jobs = [tombstone('t1', 'dead-a'), tombstone('t2', 'dead-a')]
  api.rerunRestartBatch.mockResolvedValue(preview)
  data.confirm.mockResolvedValue(false)
  const { container } = render(JobsPage)
  await waitFor(() => expect(banners(container)).toHaveLength(1))
  await fireEvent.click(screen.getByText('Rerun…'))
  await waitFor(() => expect(data.confirm).toHaveBeenCalledTimes(1))
  // cancel: only the preview went out
  expect(api.rerunRestartBatch).toHaveBeenCalledTimes(1)
  const member = api.rerunRestartBatch.mock.calls[0][0]
  expect(api.rerunRestartBatch.mock.calls).toEqual([[member]])

  api.rerunRestartBatch.mockReset()
  api.rerunRestartBatch.mockResolvedValueOnce(preview).mockResolvedValueOnce({
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
  await waitFor(() => expect(api.rerunRestartBatch).toHaveBeenCalledTimes(2))
  expect(api.rerunRestartBatch.mock.calls).toEqual([[member], [member, true]])
  const { notify } = await import('../toast')
  await waitFor(() =>
    expect(notify.success).toHaveBeenCalledWith(
      'Queued 2; skipped 1: t3 (its workspace is gone)',
    ),
  )
})

it('an empty preview says so, with Close only, and hides the banner', async () => {
  data.jobs = [tombstone('t1', 'dead-a')]
  api.rerunRestartBatch.mockResolvedValue({
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
  expect(api.rerunRestartBatch.mock.calls).toEqual([['t1']])
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

it('the request filter can be picked as well as cleared', async () => {
  location.hash = '#/ws/default/jobs'
  data.jobs = linked()
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('wf-c')).toBeTruthy())
  const picker = await screen.findByRole('combobox', {
    name: 'filter by request',
  })
  await fireEvent.change(picker, { target: { value: 'r1' } })
  await waitFor(() => expect(screen.queryByText('wf-c')).toBeNull())
  expect(screen.getByText('wf-a')).toBeTruthy()
  expect(location.hash).toBe('#/ws/default/jobs?req=r1')
  expect(data.listed.at(-1)?.[3]).toBe('r1')
  await fireEvent.click(screen.getByLabelText('remove the request filter'))
  await waitFor(() => expect(screen.getByText('wf-c')).toBeTruthy())
  expect(location.hash).toBe('#/ws/default/jobs')
  location.hash = ''
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

it('a double click on a move sends one request', async () => {
  location.hash = '#/ws/default/jobs'
  data.queued = 2
  api.moveJob.mockResolvedValue({})
  data.jobs = [
    summary({ id: 'q1', status: 'queued', queue_position: 0 }),
    summary({ id: 'q2', status: 'queued', queue_position: 1 }),
  ]
  render(JobsPage)
  await waitFor(() => expect(moves('q2')).toEqual(['front', 'up']))
  const up = screen.getByLabelText('move q2 up one place') as HTMLButtonElement
  // Both clicks land before the page renders the first one's busy state
  up.click()
  up.click()
  await waitFor(() => expect(api.moveJob).toHaveBeenCalled())
  await quiet()
  expect(api.moveJob).toHaveBeenCalledTimes(1)
  expect(api.moveJob).toHaveBeenCalledWith('q2', 'up')
})

it("a moving job's buttons are busy", async () => {
  location.hash = '#/ws/default/jobs'
  data.queued = 2
  let finish = () => {}
  api.moveJob.mockReturnValue(new Promise<void>((r) => (finish = r)))
  data.jobs = [
    summary({ id: 'q1', status: 'queued', queue_position: 0 }),
    summary({ id: 'q2', status: 'queued', queue_position: 1 }),
  ]
  render(JobsPage)
  await waitFor(() => expect(moves('q2')).toEqual(['front', 'up']))
  const up = screen.getByLabelText('move q2 up one place')
  expect(up.getAttribute('aria-busy')).toBe('false')
  await fireEvent.click(up)
  await waitFor(() => expect(up.getAttribute('aria-busy')).toBe('true'))
  expect(up.getAttribute('aria-disabled')).toBe('true')
  finish()
  await waitFor(() => expect(up.getAttribute('aria-busy')).toBe('false'))
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
  data.listError = new NetworkError('Failed to fetch')
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

it('a refusal from a quiet poll clears the rows, not just a loud load', async () => {
  location.hash = '#/ws/default/jobs'
  data.jobs = [summary({ id: 'x', workflow: 'still-here' })]
  render(JobsPage)
  await screen.findByText('still-here')
  data.listError = new ApiError('no_workspace', 404)
  // The 3 s poll finds it, with no reload in between
  await waitFor(() => expect(screen.queryByText('still-here')).toBeNull(), {
    timeout: 6000,
  })
}, 10000)

const proofFile = (run_id: string) => ({
  name: `wf/${run_id}/out.png`,
  kind: 'image',
  run_id,
  folder: 'wf',
  url: `/api/outputs/wf/${run_id}/out.png`,
  label: 'out.png',
  mtime: 1,
  size: 1,
  starred: false,
})

it('each row shows its picture, prompt and note', async () => {
  location.hash = ''
  data.jobs = [
    summary({
      id: 'a-1',
      workflow: 'text-to-image',
      run_id: 'run-a',
      note: 'the good one\nsecond line',
    }),
    summary({ id: 'b-2', workflow: 'text-to-image' }),
  ]
  data.prompts = { 'a-1': 'a red fox\nin snow', 'b-2': 'a blue whale' }
  data.files = [proofFile('run-a')]
  const { container } = render(JobsPage)
  await waitFor(() => expect(screen.getByText('a red fox')).toBeTruthy())
  expect(screen.getByText('a blue whale')).toBeTruthy()
  expect(screen.getByText('the good one')).toBeTruthy()
  expect(screen.queryByText('second line')).toBeNull()
  // The row with a run has its picture; the other keeps the same slot
  await waitFor(() =>
    expect(container.querySelectorAll('.row .proof img').length).toBe(1),
  )
  expect(container.querySelectorAll('.row .proof').length).toBe(2)
  // Each id is on its own line, whole
  expect(screen.getByText('a-1').closest('code')).toBeTruthy()
})

it('searching a prompt or an id finds the job', async () => {
  data.jobs = [
    summary({ id: 'a-1', workflow: 'text-to-image' }),
    summary({ id: 'b-2', workflow: 'text-to-image', note: 'keeper' }),
  ]
  data.prompts = { 'a-1': 'a red fox', 'b-2': 'a blue whale' }
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('a red fox')).toBeTruthy())
  const input = screen.getByPlaceholderText(
    'search names, prompts, notes and ids…',
  )
  await fireEvent.input(input, { target: { value: 'whale' } })
  await waitFor(() => expect(screen.queryByText('a red fox')).toBeNull())
  expect(screen.getByText('a blue whale')).toBeTruthy()
  await fireEvent.input(input, { target: { value: 'a-1 fox' } })
  await waitFor(() => expect(screen.queryByText('a blue whale')).toBeNull())
  expect(screen.getByText('a red fox')).toBeTruthy()
  await fireEvent.input(input, { target: { value: 'keeper' } })
  await waitFor(() => expect(screen.getByText('a blue whale')).toBeTruthy())
  expect(screen.queryByText('a red fox')).toBeNull()
  location.hash = ''
})

it('a running row shows its percentage', async () => {
  data.jobs = [summary({ id: 'r-1', status: 'running', finished_at: null })]
  data.progress = { 'r-1': { denoise_step: 5, denoise_total_steps: 20 } }
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('25%')).toBeTruthy())
})

it('prompts are asked for once, not on every poll', async () => {
  data.jobs = [summary({ id: 'a-1' })]
  data.prompts = { 'a-1': 'a red fox' }
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('a red fox')).toBeTruthy())
  // the live read and the finished read each ask once
  expect(data.asked).toEqual([true, true])
})

// n finished jobs, oldest first, with created_at 1..n
const finished = (n: number, from = 1) =>
  Array.from({ length: n }, (_, i) =>
    summary({
      id: `d-${from + i}`,
      workflow: `w${from + i}`,
      created_at: from + i,
    }),
  )
// The finished reads, as [limit] of each
const finishedLimits = () =>
  data.listed
    .filter((l) => l[2] === 'succeeded,failed,cancelled')
    .map((l) => l[1])

it('the page says how many of how many it shows', async () => {
  data.jobs = finished(120)
  const { container } = render(JobsPage)
  await waitFor(() =>
    expect(container.textContent).toContain(
      'Showing the newest 50 of 120 jobs.',
    ),
  )
  expect(screen.getByRole('button', { name: 'Show 50 older' })).toBeTruthy()
  expect(container.querySelectorAll('a[href*="/jobs/"]')).toHaveLength(50)
})

it('Show older reads the next 50', async () => {
  data.jobs = finished(120)
  const { container } = render(JobsPage)
  await fireEvent.click(
    await screen.findByRole('button', { name: 'Show 50 older' }),
  )
  await waitFor(() => expect(finishedLimits()).toContain(100))
  await waitFor(() =>
    expect(container.textContent).toContain(
      'Showing the newest 100 of 120 jobs.',
    ),
  )
  expect(screen.getByRole('button', { name: 'Show 20 older' })).toBeTruthy()
})

const live = (n: number) =>
  Array.from({ length: n }, (_, i) =>
    summary({
      id: `l-${i}`,
      workflow: `live${i}`,
      status: 'running',
      created_at: 1000 + i,
      finished_at: null,
    }),
  )

it('3 live and 49 finished is one whole page: no paging line, no button', async () => {
  data.jobs = [...finished(49), ...live(3)]
  const { container } = render(JobsPage)
  await waitFor(() => expect(screen.getByText('live2')).toBeTruthy())
  expect(container.textContent).not.toContain('Showing the newest')
  expect(screen.queryByRole('button', { name: /older/ })).toBeNull()
})

it('3 live and 120 finished count the rows shown against every job', async () => {
  data.jobs = [...finished(120), ...live(3)]
  const { container } = render(JobsPage)
  await waitFor(() =>
    expect(container.textContent).toContain(
      'Showing the newest 53 of 123 jobs.',
    ),
  )
  expect(screen.getByRole('button', { name: 'Show 50 older' })).toBeTruthy()
})

it('a filter over a partial page says what it searched', async () => {
  location.hash = '#/ws/default/jobs?q=zzz'
  data.jobs = finished(120)
  const { container } = render(JobsPage)
  await waitFor(() =>
    expect(container.textContent).toContain(
      'Searching the newest 50 of 120 jobs.',
    ),
  )
  expect(screen.getByRole('button', { name: 'Search all 120' })).toBeTruthy()
  location.hash = ''
})

it('a live-only filter does not offer Search all', async () => {
  location.hash = '#/ws/default/jobs?status=running'
  data.jobs = [...finished(120), ...live(3)]
  const { container } = render(JobsPage)
  await waitFor(() => expect(screen.getByText('live2')).toBeTruthy())
  expect(container.textContent).not.toContain('Searching the newest')
  expect(screen.queryByRole('button', { name: /Search all/ })).toBeNull()
  location.hash = ''
})

it('no filter, no search line; a whole page, no paging lines', async () => {
  data.jobs = finished(3)
  const { container } = render(JobsPage)
  await waitFor(() => expect(screen.getByText('w3')).toBeTruthy())
  expect(container.textContent).not.toContain('Showing the newest')
  expect(container.textContent).not.toContain('Searching the newest')
})

it('Search all reads every job', async () => {
  location.hash = '#/ws/default/jobs?q=w7'
  data.jobs = finished(120)
  render(JobsPage)
  await fireEvent.click(
    await screen.findByRole('button', { name: 'Search all 120' }),
  )
  await waitFor(() => expect(finishedLimits()).toContain(120))
  // w7 and w70-w79 and w107 match; the oldest ones are only found now
  await waitFor(() => expect(screen.getByText('w7')).toBeTruthy())
  expect(screen.queryByRole('button', { name: /Search all/ })).toBeNull()
  location.hash = ''
})

it('after Search all a poll reads a page, and a newly finished job brings nothing back', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  location.hash = '#/ws/default/jobs?q=w'
  try {
    data.jobs = finished(120)
    const { container } = render(JobsPage)
    await fireEvent.click(
      await screen.findByRole('button', { name: 'Search all 120' }),
    )
    await waitFor(() => expect(finishedLimits()).toContain(120))
    await waitFor(() => expect(screen.getByText('w1')).toBeTruthy())
    const from = data.listed.length
    data.jobs = [...data.jobs, ...finished(1, 121)]
    await vi.advanceTimersByTimeAsync(3500)
    await waitFor(() => expect(screen.getByText('w121')).toBeTruthy())
    const polled = data.listed
      .slice(from)
      .filter((l) => l[2] === 'succeeded,failed,cancelled')
    expect(polled.length).toBeGreaterThan(0)
    expect(polled.every((l) => l[1] === 50)).toBe(true)
    // the older ones stay, and nothing offers a search of the rest
    expect(screen.getByText('w1')).toBeTruthy()
    expect(container.textContent).not.toContain('Searching the newest')
    expect(screen.queryByRole('button', { name: /Search all/ })).toBeNull()
  } finally {
    vi.useRealTimers()
    location.hash = ''
  }
})

it('a filter change does not reset how much is read', async () => {
  data.jobs = finished(120)
  render(JobsPage)
  await fireEvent.click(
    await screen.findByRole('button', { name: 'Show 50 older' }),
  )
  await waitFor(() => expect(finishedLimits()).toContain(100))
  const before = data.listed.length
  await fireEvent.input(
    screen.getByPlaceholderText('search names, prompts, notes and ids…'),
    { target: { value: 'w' } },
  )
  expect(data.listed.length).toBe(before)
  expect(finishedLimits().at(-1)).toBe(100)
  location.hash = ''
})

it('a running job older than the page is listed', async () => {
  data.jobs = [
    summary({
      id: 'run-1',
      workflow: 'still-going',
      status: 'running',
      created_at: 0,
      finished_at: null,
    }),
    ...finished(60),
  ]
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('still-going')).toBeTruthy())
})

it('a second poll keeps the rows and the prompts', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  try {
    data.jobs = [summary({ id: 'a-1', workflow: 'first' })]
    data.prompts = { 'a-1': 'a red fox' }
    render(JobsPage)
    await waitFor(() => expect(screen.getByText('a red fox')).toBeTruthy())
    const asked = data.asked.length
    await vi.advanceTimersByTimeAsync(3500)
    // the poll read again, without prompts, and nothing went missing
    expect(data.asked.length).toBeGreaterThan(asked)
    expect(data.asked.slice(asked).every((a) => !a)).toBe(true)
    expect(screen.getByText('first')).toBeTruthy()
    expect(screen.getByText('a red fox')).toBeTruthy()
  } finally {
    vi.useRealTimers()
  }
})

it("the search field's title says what it searches", async () => {
  render(JobsPage)
  expect(
    screen.getByPlaceholderText('search names, prompts, notes and ids…').title,
  ).toBe('search names, prompts, notes and ids')
})

// Jobs of one workspace, oldest first
const inWorkspace = (ws: string, n: number) =>
  finished(n).map((j) => ({ ...j, id: `${ws}-${j.id}`, workspace: ws }))

it('a workspace switch starts at the first page, with its paging line', async () => {
  data.byWorkspace = true
  workspace.current = 'a'
  data.jobs = [...inWorkspace('a', 120), ...inWorkspace('b', 120)]
  const { container } = render(JobsPage)
  await fireEvent.click(
    await screen.findByRole('button', { name: 'Show 50 older' }),
  )
  await waitFor(() =>
    expect(container.textContent).toContain(
      'Showing the newest 100 of 120 jobs.',
    ),
  )
  workspace.current = 'b'
  await waitFor(() =>
    expect(container.textContent).toContain(
      'Showing the newest 50 of 120 jobs.',
    ),
  )
  const ofB = data.listed.filter(
    (l) => l[0] === 'b' && l[2] === 'succeeded,failed,cancelled',
  )
  expect(ofB.every((l) => l[1] === 50)).toBe(true)
  expect(container.querySelectorAll('a[href*="/jobs/"]')).toHaveLength(50)
})

it('a workspace switch after Search all offers Search all again', async () => {
  location.hash = '#/ws/default/jobs?q=w'
  data.byWorkspace = true
  workspace.current = 'a'
  data.jobs = [...inWorkspace('a', 120), ...inWorkspace('b', 120)]
  const { container } = render(JobsPage)
  await fireEvent.click(
    await screen.findByRole('button', { name: 'Search all 120' }),
  )
  await waitFor(() => expect(finishedLimits()).toContain(120))
  workspace.current = 'b'
  await waitFor(() =>
    expect(container.textContent).toContain(
      'Searching the newest 50 of 120 jobs.',
    ),
  )
  expect(screen.getByRole('button', { name: 'Search all 120' })).toBeTruthy()
  location.hash = ''
})

it('a request filter change starts at the first page', async () => {
  data.jobs = finished(120).map((j, i) => ({
    ...j,
    request: i < 100 ? 'r1' : 'r2',
  }))
  const { container } = render(JobsPage)
  await fireEvent.click(
    await screen.findByRole('button', { name: 'Show 50 older' }),
  )
  await waitFor(() => expect(finishedLimits()).toContain(100))
  location.hash = '#/ws/default/jobs?req=r1'
  await fireEvent(window, new HashChangeEvent('hashchange'))
  await waitFor(() =>
    expect(container.textContent).toContain(
      'Showing the newest 50 of 100 jobs.',
    ),
  )
  location.hash = ''
})

it('an empty jobs page says where to go', async () => {
  data.jobs = []
  location.hash = '#/ws/default/jobs'
  render(JobsPage)
  const link = await screen.findByRole('link', { name: 'Pick a workflow' })
  expect(link.getAttribute('href')).toBe('#/ws/default/workflows')
})

// A new question starts at the first page, and asks once: the load resetting
// the page it had just read ran it a second time
it('another workspace after Show older reads its first page once', async () => {
  data.jobs = finished(120)
  render(JobsPage)
  await fireEvent.click(
    await screen.findByRole('button', { name: 'Show 50 older' }),
  )
  await screen.findByRole('button', { name: 'Show 20 older' })
  const before = data.listed.length
  workspace.current = 'other'
  await screen.findByRole('button', { name: 'Show 50 older' })
  // past any follow-up read, short of the 3 s poll, which asks again by right
  await vi.advanceTimersByTimeAsync(1000)
  const asked = data.listed
    .slice(before)
    .filter((l) => l[0] === 'other' && l[2] === 'succeeded,failed,cancelled')
  expect(asked.map((l) => l[1])).toEqual([50])
  workspace.current = DEFAULT_WORKSPACE
})

it('a status in the link the picker does not offer reads as all', async () => {
  location.hash = '#/ws/default/jobs?status=exploded'
  data.jobs = [summary({ id: 'ok', workflow: 'fine' })]
  render(JobsPage)
  await waitFor(() => expect(screen.getByText('fine')).toBeTruthy())
  const status = screen.getByTitle('filter by status') as HTMLSelectElement
  expect(status.value).toBe('')
  location.hash = ''
})
