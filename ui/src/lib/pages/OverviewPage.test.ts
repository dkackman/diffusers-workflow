import overviewSource from './OverviewPage.svelte?raw'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import OverviewPage from './OverviewPage.svelte'
import { formatMtime } from '../format'
import { feed } from '../test/feedMock'
import { invalidateWorkspaces, workspace } from '../workspace.svelte'
import '../router.svelte'

const data = vi.hoisted(() => ({
  files: [] as unknown[],
  galleryWorkspace: 'studio',
  jobs: [] as unknown[],
  workflows: [] as string[],
  assets: [] as unknown[],
  galleryError: null as Error | null,
  requests: [] as unknown[],
  requestsError: null as Error | null,
  requestsFor: null as ((ws: string) => Promise<unknown>) | null,
}))
const calls = vi.hoisted(() => ({}) as Record<string, number>)
const hit = (name: string) => (calls[name] = (calls[name] ?? 0) + 1)
const deleteWorkspace = vi.hoisted(() => vi.fn())
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))
vi.mock('../api', () => ({
  api: {
    gallery: () => (
      hit('gallery'),
      data.galleryError
        ? Promise.reject(data.galleryError)
        : Promise.resolve({
            files: data.files,
            workspace: data.galleryWorkspace,
          })
    ),
    // Filters the fixture the way the server does: `status` is one state
    // or a comma-separated set, `limit` keeps the newest (last) N
    listJobs: (_ws?: string, limit?: number, status?: string) => {
      hit('jobs')
      const states = status?.split(',')
      let jobs = data.jobs as Array<{ status: string }>
      if (states) jobs = jobs.filter((j) => states.includes(j.status))
      if (limit) jobs = jobs.slice(Math.max(0, jobs.length - limit))
      return Promise.resolve({ jobs, total: jobs.length })
    },
    listRequests: (ws: string) => (
      hit('requests'),
      data.requestsFor
        ? data.requestsFor(ws)
        : data.requestsError
          ? Promise.reject(data.requestsError)
          : Promise.resolve({ workspace: 'studio', requests: data.requests })
    ),
    listWorkflows: () => (
      hit('workflows'),
      Promise.resolve({
        libraries: [
          { origin: 'workspace', root: '/ws/workflows', writable: true },
        ],
        shadowed: [],
        workflows: data.workflows,
        details: {},
      })
    ),
    listAssets: () => (
      hit('assets'),
      Promise.resolve({
        assets: data.assets,
        libraries: [],
        shadowed: [],
        folders: [],
        workspace: 'default',
      })
    ),
    listWorkspaces: () =>
      Promise.resolve({
        workspace_root: '/ws',
        default: 'default',
        workspaces: [
          { name: 'default', default: true },
          {
            name: 'studio',
            default: false,
            usage: { files: 4, bytes: 4096 },
            note: 'client moodboards',
          },
        ],
      }),
    deleteWorkspace: (n: string, a?: boolean) => deleteWorkspace(n, a),
  },
}))
vi.mock('../confirm.svelte', () => ({
  confirmDialog: vi.fn(() => Promise.resolve(true)),
}))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

beforeEach(async () => {
  for (const k of Object.keys(calls)) delete calls[k]
  data.files = []
  data.galleryWorkspace = 'studio'
  data.jobs = []
  data.workflows = []
  data.assets = []
  data.galleryError = null
  data.requests = []
  data.requestsError = null
  data.requestsFor = null
  deleteWorkspace.mockReset()
  localStorage.clear()
  location.hash = '#/ws/studio/overview'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  invalidateWorkspaces()
  workspace.names = undefined
  workspace.root = null
  workspace.usage = {}
})
afterEach(cleanup)

it('sets the tab title to the workspace name', async () => {
  render(OverviewPage)
  await waitFor(() =>
    expect(document.title).toBe('studio · diffusers-workflow'),
  )
  expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('studio')
})

it('an empty workspace shows empty states, not empty frames', async () => {
  render(OverviewPage)
  await waitFor(() =>
    expect(screen.getByText(/nothing generated yet/i)).toBeTruthy(),
  )
  expect(document.querySelectorAll('.filetile')).toHaveLength(0)
  expect(screen.getByText(/no jobs yet/i)).toBeTruthy()
  expect(screen.getByText(/no workflows yet/i)).toBeTruthy()
})

it('shows the workspace note', async () => {
  render(OverviewPage)
  await waitFor(() =>
    expect(screen.getByText('client moodboards')).toBeTruthy(),
  )
  expect(screen.getByText('Your note on this workspace')).toBeTruthy()
})

it('one panel failing leaves the others', async () => {
  data.galleryError = new Error('boom')
  data.jobs = [
    {
      id: 'j1',
      workflow: 'shot',
      status: 'succeeded',
      created_at: 1,
      started_at: 1,
      finished_at: 2,
      workspace: 'studio',
    },
  ]
  data.workflows = ['shot']
  render(OverviewPage)
  await waitFor(() =>
    expect(screen.getByText(/couldn't load recent outputs/i)).toBeTruthy(),
  )
  expect(screen.getAllByRole('button', { name: 'Retry' })).toHaveLength(1)
  expect(screen.getByTitle('job j1').getAttribute('href')).toBe(
    '#/ws/studio/jobs/j1',
  )
  expect(screen.getByRole('link', { name: 'shot' })).toBeTruthy()
})

it('shows recent outputs, jobs and workflows, each linking to its page', async () => {
  data.files = [
    {
      name: 'shot/r1/a.png',
      folder: 'shot',
      subfolder: '',
      url: '/o/a.png',
      kind: 'image',
      size: 1,
      mtime: 2,
      label: 'a',
    },
  ]
  data.jobs = [
    {
      id: 'j1',
      workflow: 'shot',
      status: 'succeeded',
      created_at: 1,
      started_at: 1,
      finished_at: 2,
      workspace: 'studio',
    },
  ]
  data.workflows = ['shot', 'still']
  data.assets = [
    {
      name: 'x.png',
      reference: 'asset:x.png',
      folder: '',
      kind: 'image',
      origin: 'workspace',
    },
    {
      name: 'y.png',
      reference: 'asset:y.png',
      folder: '',
      kind: 'image',
      origin: 'common',
    },
  ]
  render(OverviewPage)
  // the recent output's tile, and the workflow card's proof (also a tile)
  await waitFor(() =>
    expect(document.querySelectorAll('.filetile').length).toBe(2),
  )
  expect(
    screen.getByRole('link', { name: /recent outputs/i }).getAttribute('href'),
  ).toBe('#/ws/studio/gallery')
  expect(screen.getByTitle('job j1').getAttribute('href')).toBe(
    '#/ws/studio/jobs/j1',
  )
  // the workflow with a proof sorts first and carries its picture
  const cards = screen.getAllByRole('link', { name: /^(shot|still)$/ })
  expect(cards[0].textContent).toContain('shot')
  expect(screen.getByText('1 asset')).toBeTruthy() // the workspace's own only
  await waitFor(() => expect(screen.getByText('4 KB')).toBeTruthy())
})

it('a deep queue does not hide the running job, and queued jobs are not "recent"', async () => {
  const job = (id: string, status: string, created_at: number) => ({
    id,
    workflow: 'shot',
    status,
    created_at,
    started_at: null,
    finished_at: null,
    workspace: 'studio',
  })
  data.jobs = [
    job('done1', 'succeeded', 1),
    job('running1', 'running', 2),
    ...[3, 4, 5, 6, 7, 8].map((n) => job(`queued${n}`, 'queued', n)),
  ]
  render(OverviewPage)
  await waitFor(() => expect(screen.getByTitle('job running1')).toBeTruthy())
  expect(document.querySelector('.row-live')).toBeTruthy()
  expect(screen.getByTitle('job done1')).toBeTruthy()
  expect(document.querySelectorAll('.chip.queued')).toHaveLength(0)
  expect(screen.queryByTitle(/^job queued/)).toBeNull()
})

it('delete is offered off the default and goes through the shared action', async () => {
  deleteWorkspace.mockResolvedValue({ name: 'studio', deleted: true })
  render(OverviewPage)
  await waitFor(() =>
    expect(
      screen.getByRole('button', { name: /delete workspace/i }),
    ).toBeTruthy(),
  )
  await fireEvent.click(
    screen.getByRole('button', { name: /delete workspace/i }),
  )
  await waitFor(() => expect(location.hash).toBe('#/ws/default/overview'))
})

it("the default workspace says why it can't be managed, with no disabled buttons", async () => {
  location.hash = '#/ws/default/overview'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  render(OverviewPage)
  const manage = (await screen.findByRole('heading', { name: 'Manage' }))
    .parentElement!
  expect(manage.textContent?.replace(/\s+/g, ' ')).toContain(
    "The default workspace is the server's own folder, so it can't be " +
      'hidden or deleted.',
  )
  expect(manage.querySelectorAll('button')).toHaveLength(0)
  expect(screen.queryByRole('button', { name: /delete workspace/i })).toBeNull()
})

it('off the default, Manage offers hide and a danger delete', async () => {
  render(OverviewPage)
  const remove = await screen.findByRole('button', {
    name: /delete workspace/i,
  })
  expect(remove.classList.contains('danger')).toBe(true)
  expect((remove as HTMLButtonElement).disabled).toBe(false)
  expect(
    screen.getByRole('button', { name: /hide from sidebar/i }),
  ).toBeTruthy()
})

const summary = (over: Record<string, unknown>) => ({
  id: 'r1',
  title: 'Launch posters',
  state: 'open',
  revision: 1,
  items: 3,
  status_counts: { done: 2, planned: 1 },
  spent_minutes: 4,
  remaining_minutes: 2,
  updated_at: null,
  ...over,
})

it('lists open requests with their counts, and links to them all', async () => {
  data.requests = [
    summary({}),
    summary({ id: 'r2', title: 'Old ask', state: 'done' }),
  ]
  render(OverviewPage)
  await waitFor(() => expect(screen.getByText('Launch posters')).toBeTruthy())
  expect(
    screen.getByRole('link', { name: 'Launch posters' }).getAttribute('href'),
  ).toBe('#/ws/studio/requests/r1')
  expect(screen.getByText('1 planned, nothing running')).toBeTruthy()
  expect(screen.getByText('2/3')).toBeTruthy()
  expect(screen.queryByText('Old ask')).toBeNull()
  expect(
    screen.getByRole('link', { name: 'all requests' }).getAttribute('href'),
  ).toBe('#/ws/studio/requests')
})

it('a request row keeps its title beside a long attention line', async () => {
  const needs = {
    held: 2,
    ready: 1,
    failed: 1,
    revised: 1,
    choose: 1,
    accepted_missing: 1,
    notes_waiting: 0,
  }
  data.requests = [summary({ needs })]
  render(OverviewPage)
  const link = await screen.findByRole('link', { name: 'Launch posters' })
  const row = link.closest('li')!
  expect(row.classList.contains('reqrow')).toBe(true)
  const line = row.querySelector('.reqline')!
  expect(line.textContent).toMatch(/^Needs you: 2 waiting for your OK/)
  expect(link.contains(line)).toBe(false)
  expect(link.textContent?.trim()).toBe('Launch posters')
})

it('hides the requests panel when there are none or the call fails', async () => {
  data.requestsError = new Error('boom')
  render(OverviewPage)
  await waitFor(() => expect(screen.getByText(/no jobs yet/i)).toBeTruthy())
  expect(screen.queryByText('all requests')).toBeNull()
})

it("drops another workspace's requests that land after a switch", async () => {
  let land: (reply: unknown) => void = () => {}
  data.requestsFor = (ws) =>
    ws === 'studio'
      ? new Promise((resolve) => (land = resolve))
      : Promise.resolve({
          workspace: ws,
          requests: [summary({ id: 'd1', title: 'Default ask' })],
        })
  render(OverviewPage)
  location.hash = '#/ws/default/overview'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  await waitFor(() => expect(screen.getByText('Default ask')).toBeTruthy())
  land({ workspace: 'studio', requests: [summary({})] })
  await new Promise((r) => setTimeout(r, 10))
  expect(screen.queryByText('Launch posters')).toBeNull()
  expect(screen.getByText('Default ask')).toBeTruthy()
})

const settle = () => new Promise((r) => setTimeout(r, 10))

it('each kind that moves reloads its own panel quietly, and nothing else', async () => {
  data.workflows = ['shot']
  render(OverviewPage)
  await waitFor(() => expect(screen.getByText('shot')).toBeTruthy())
  await settle()
  expect(feed.watched('studio').sort()).toEqual(
    ['assets', 'gallery', 'jobs', 'requests', 'workflows', 'workspaces'].sort(),
  )
  const before = { ...calls }
  feed.move('studio', 'jobs')
  expect(document.querySelector('[aria-label="loading jobs"]')).toBeNull()
  await settle()
  expect(calls.jobs).toBe((before.jobs ?? 0) + 2)
  expect(calls.gallery).toBe(before.gallery)
  expect(calls.workflows).toBe(before.workflows)

  data.workflows = ['shot', 'more']
  feed.move('studio', 'workflows')
  expect(screen.getByText('shot')).toBeTruthy()
  await waitFor(() => expect(screen.getByText('more')).toBeTruthy())
  expect(calls.workflows).toBe((before.workflows ?? 0) + 1)

  feed.move('studio', 'gallery')
  feed.move('studio', 'requests')
  feed.move('studio', 'assets')
  await settle()
  expect(calls.gallery).toBe((before.gallery ?? 0) + 1)
  expect(calls.requests).toBe((before.requests ?? 0) + 1)
  expect(calls.assets).toBe((before.assets ?? 0) + 1)
})

it("another workspace's move, or an unwatched kind, reloads nothing", async () => {
  render(OverviewPage)
  await waitFor(() => expect(screen.getByText(/no jobs yet/i)).toBeTruthy())
  await settle()
  const before = { ...calls }
  feed.move('default', 'jobs')
  feed.move('default', 'gallery')
  feed.move('studio', 'prompts')
  await settle()
  expect(calls).toEqual(before)
})

it('a quiet reload that fails keeps the panel under a refresh line', async () => {
  data.files = [{ name: 'a.png', label: 'a', kind: 'image' }]
  render(OverviewPage)
  await waitFor(() =>
    expect(document.querySelectorAll('.filetile')).toHaveLength(1),
  )
  data.galleryError = new Error('boom')
  feed.move('studio', 'gallery')
  await settle()
  expect(document.querySelectorAll('.filetile')).toHaveLength(1)
  expect(screen.getByText(/couldn't refresh recent outputs/i)).toBeTruthy()
})

const output = (name: string, run_id: string, kind = 'image') => ({
  name,
  folder: name.split('/')[0],
  subfolder: '',
  run_id,
  version: 1,
  url: `/outputs/${name}`,
  kind,
  size: 1,
  mtime: 2,
  label: name.split('/').pop(),
})
const recentTiles = () => [
  ...document.querySelectorAll<HTMLAnchorElement>('.strip a.filetile'),
]

it('a recent tile opens that file in the gallery', async () => {
  data.files = [
    output('shot/r1/a.png', 'r1'),
    output('shot/r1/b.mp4', 'r1', 'video'),
  ]
  render(OverviewPage)
  await waitFor(() => expect(recentTiles()).toHaveLength(2))
  expect(recentTiles().map((a) => a.getAttribute('href'))).toEqual([
    '#/ws/studio/gallery?open=shot%2Fr1%2Fa.png',
    '#/ws/studio/gallery?open=shot%2Fr1%2Fb.mp4',
  ])
  // a clip carries its play badge
  expect(recentTiles()[1].querySelector('.videobadge')).toBeTruthy()
  expect(screen.getByRole('img', { name: 'a.png' })).toBeTruthy()
})

it('tiles keep the workspace they were listed in after a switch', async () => {
  data.files = [output('shot/r1/a.png', 'r1')]
  render(OverviewPage)
  await waitFor(() => expect(recentTiles()).toHaveLength(1))
  location.hash = '#/ws/default/overview'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  // the listing that lands still says it is studio's
  await waitFor(() => expect(calls.gallery).toBe(2))
  await settle()
  const [tile] = recentTiles()
  expect(tile.getAttribute('href')).toBe(
    '#/ws/studio/gallery?open=shot%2Fr1%2Fa.png',
  )
  expect(tile.querySelector('img')?.getAttribute('src')).toContain(
    'workspace=studio',
  )
})

it('a job row is one link, titled with its id', async () => {
  data.files = [
    output('shot/r1/a.wav', 'r1', 'audio'),
    output('shot/r1/a.png', 'r1'),
  ]
  data.jobs = [
    {
      id: 'j1',
      workflow: 'shot-id',
      workflow_name: 'shots/shot',
      status: 'succeeded',
      run_id: 'r1',
      created_at: 1,
      started_at: 1,
      finished_at: 2,
      workspace: 'studio',
    },
  ]
  render(OverviewPage)
  await waitFor(() => expect(screen.getByTitle('job j1')).toBeTruthy())
  const row = screen.getByTitle('job j1')
  expect(row.tagName).toBe('A')
  expect(row.getAttribute('href')).toBe('#/ws/studio/jobs/j1')
  expect(row.querySelector('.chip.succeeded')).toBeTruthy()
  expect(row.textContent).toContain('shot')
  expect(row.textContent).toContain('j1')
  // the run's proof, as a mini tile inside the one link
  await waitFor(() =>
    expect(row.querySelector('.tile-mini img')?.getAttribute('src')).toContain(
      'shot/r1/a.png',
    ),
  )
  expect(row.closest('li')?.querySelectorAll('a')).toHaveLength(1)
})

it('a job with no proof draws the cover slot, and same-workflow rows differ', async () => {
  const job = (id: string, finished: number) => ({
    id,
    workflow: 'text-to-image',
    workflow_name: 'text-to-image',
    status: 'failed',
    run_id: null,
    created_at: finished - 10,
    started_at: finished - 5,
    finished_at: finished,
    workspace: 'studio',
  })
  data.files = []
  data.jobs = [job('j1', 1_700_000_000), job('j2', 1_700_090_000)]
  render(OverviewPage)
  await waitFor(() => expect(screen.getByTitle('job j1')).toBeTruthy())
  const a = screen.getByTitle('job j1')
  const b = screen.getByTitle('job j2')
  // the dashed slot stands where a proof would, in the same 48 px frame
  expect(a.querySelector('.slot.as-cover')).toBeTruthy()
  expect(b.querySelector('.slot.as-cover')).toBeTruthy()
  // same workflow and status, told apart by when they finished
  expect(a.textContent).not.toBe(b.textContent)
  expect(a.querySelector('.when')?.textContent).toBe(
    `${formatMtime(1_700_000_000)}\u00a0·\u00a0j1`,
  )
  const names = screen
    .getAllByRole('link')
    .filter((l) => l.getAttribute('title')?.startsWith('job '))
    .map((l) => l.textContent?.replace(/\s+/g, ' ').trim())
  expect(new Set(names).size).toBe(2)
})

it('same-workflow jobs finishing in one minute differ by their id', async () => {
  const job = (id: string, finished: number) => ({
    id,
    workflow: 'text-to-image',
    workflow_name: 'text-to-image',
    status: 'succeeded',
    run_id: null,
    created_at: finished - 10,
    started_at: finished - 5,
    finished_at: finished,
    workspace: 'studio',
  })
  data.files = []
  data.jobs = [job('abc123', 1_700_000_000), job('def456', 1_700_000_020)]
  render(OverviewPage)
  await waitFor(() => expect(screen.getByTitle('job abc123')).toBeTruthy())
  const a = screen.getByTitle('job abc123')
  const b = screen.getByTitle('job def456')
  expect(a.querySelector('.when')?.textContent).toBe(
    `${formatMtime(1_700_000_000)}\u00a0·\u00a0abc123`,
  )
  expect(a.textContent).not.toBe(b.textContent)
  const name = (l: HTMLElement) => l.textContent?.replace(/\s+/g, ' ').trim()
  expect(name(a)).toContain(formatMtime(1_700_000_000))
  expect(name(a)).toContain('abc123')
  expect(name(a)).not.toBe(name(b))
})

it('a job row puts its time and full id on a second line, id never truncated', async () => {
  data.files = []
  data.jobs = [
    {
      id: 'e2ebrdtake01',
      workflow: 'text-to-image',
      workflow_name: 'text-to-image',
      status: 'failed',
      run_id: null,
      created_at: 1_699_999_990,
      started_at: 1_699_999_995,
      finished_at: 1_700_000_000,
      workspace: 'studio',
    },
  ]
  render(OverviewPage)
  await waitFor(() =>
    expect(screen.getByTitle('job e2ebrdtake01')).toBeTruthy(),
  )
  const row = screen.getByTitle('job e2ebrdtake01')
  const line1 = row.querySelector('.line1')
  const when = row.querySelector('.when')
  const jid = row.querySelector('.jid') as HTMLElement
  expect(line1?.querySelector('.wf')).toBeTruthy()
  expect(line1?.contains(when)).toBe(false)
  expect(when?.contains(jid)).toBe(true)
  expect(jid.textContent).toBe('e2ebrdtake01')
  // the accessible name keeps workflow, time and id
  const name = row.textContent?.replace(/\s+/g, ' ')
  expect(name).toContain('text-to-image')
  expect(name).toContain(formatMtime(1_700_000_000))
  expect(name).toContain('e2ebrdtake01')
})

it('a long job id keeps its row: clipped, with the whole id in its title', async () => {
  const id = 'e2e' + 'x'.repeat(80)
  data.jobs = [
    {
      id,
      workflow: 'text-to-image',
      workflow_name: 'text-to-image',
      status: 'failed',
      run_id: null,
      created_at: 1,
      started_at: 1,
      finished_at: 2,
      workspace: 'studio',
    },
  ]
  render(OverviewPage)
  await waitFor(() => expect(screen.getByTitle(`job ${id}`)).toBeTruthy())
  expect(document.querySelector('.jid')?.getAttribute('title')).toBe(id)
  // jsdom applies no component styles: the rule is read off the source
  const rule = /\.jobs \.when \.jid\s*\{[^}]*\}/.exec(overviewSource)?.[0] ?? ''
  expect(rule).toMatch(/min-width:\s*0/)
  expect(rule).toMatch(/overflow:\s*hidden/)
  expect(rule).toMatch(/text-overflow:\s*ellipsis/)
})

it('the overview shows the card until it is dismissed', async () => {
  data.workflows = ['templates/text-to-image']
  render(OverviewPage)
  const card = await screen.findByRole('region', {
    name: 'New here? Make your first picture',
  })
  await waitFor(() =>
    expect(
      screen
        .getByRole('link', { name: 'Open text-to-image' })
        .getAttribute('href'),
    ).toBe('#/ws/studio/workflows/templates/text-to-image'),
  )
  expect(
    card.compareDocumentPosition(
      screen.getByText('Your note on this workspace'),
    ) & Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy()
  await fireEvent.click(
    screen.getByRole('button', { name: 'Dismiss this introduction' }),
  )
  expect(screen.queryByText('New here? Make your first picture')).toBeNull()
})

it('an empty overview says where to go', async () => {
  render(OverviewPage)
  await waitFor(() => expect(screen.getAllByText(/no jobs yet/i)).toBeTruthy())
  const hrefs = (text: string) =>
    screen
      .getAllByRole('link', { name: text })
      .map((a) => a.getAttribute('href'))
  expect(hrefs('Run a workflow')).toHaveLength(2)
  for (const href of hrefs('Run a workflow'))
    expect(href).toMatch(/^#\/ws\/[^/]+\/workflows$/)
  expect(hrefs('Create one in the editor')[0]).toMatch(/^#\/ws\/[^/]+\/edit$/)
  expect(hrefs('start from an example')).toEqual(['#/shared/examples'])
})
