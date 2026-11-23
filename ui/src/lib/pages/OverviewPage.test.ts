import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import OverviewPage from './OverviewPage.svelte'
import { feed } from '../test/feedMock'
import { invalidateWorkspaces, workspace } from '../workspace.svelte'
import '../router.svelte'

const data = vi.hoisted(() => ({
  files: [] as unknown[],
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
        : Promise.resolve({ files: data.files })
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
    galleryThumbnailUrl: (n: string) => `/thumb/${n}`,
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
  expect(document.querySelectorAll('.frame')).toHaveLength(0)
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
  expect(screen.getByRole('link', { name: 'j1' }).getAttribute('href')).toBe(
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
  // the recent output's frame, and the workflow card's proof (also a .frame)
  await waitFor(() =>
    expect(document.querySelectorAll('.frame').length).toBe(2),
  )
  expect(
    screen.getByRole('link', { name: /recent outputs/i }).getAttribute('href'),
  ).toBe('#/ws/studio/gallery')
  expect(screen.getByRole('link', { name: 'j1' }).getAttribute('href')).toBe(
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
  await waitFor(() =>
    expect(screen.getByRole('link', { name: 'running1' })).toBeTruthy(),
  )
  expect(document.querySelector('.runningnow')).toBeTruthy()
  expect(screen.getByRole('link', { name: 'done1' })).toBeTruthy()
  expect(document.querySelectorAll('.chip.queued')).toHaveLength(0)
  expect(screen.queryByRole('link', { name: /^queued/ })).toBeNull()
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
  expect(screen.getByText('idle: 1 planned, nothing running')).toBeTruthy()
  expect(screen.getByText('2/3')).toBeTruthy()
  expect(screen.queryByText('Old ask')).toBeNull()
  expect(
    screen.getByRole('link', { name: 'all requests' }).getAttribute('href'),
  ).toBe('#/ws/studio/requests')
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
    expect(document.querySelectorAll('.tile')).toHaveLength(1),
  )
  data.galleryError = new Error('boom')
  feed.move('studio', 'gallery')
  await settle()
  expect(document.querySelectorAll('.tile')).toHaveLength(1)
  expect(screen.getByText(/couldn't refresh recent outputs/i)).toBeTruthy()
})
