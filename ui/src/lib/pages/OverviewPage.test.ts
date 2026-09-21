import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import OverviewPage from './OverviewPage.svelte'
import { invalidateWorkspaces, workspace } from '../workspace.svelte'
import '../router.svelte'

const data = vi.hoisted(() => ({
  files: [] as unknown[],
  jobs: [] as unknown[],
  workflows: [] as string[],
  assets: [] as unknown[],
  galleryError: null as Error | null,
}))
const deleteWorkspace = vi.hoisted(() => vi.fn())
vi.mock('../api', () => ({
  api: {
    gallery: () =>
      data.galleryError
        ? Promise.reject(data.galleryError)
        : Promise.resolve({ files: data.files }),
    // Filters the fixture the way the server does: `status` is one state
    // or a comma-separated set, `limit` keeps the newest (last) N
    listJobs: (_ws?: string, limit?: number, status?: string) => {
      const states = status?.split(',')
      let jobs = data.jobs as Array<{ status: string }>
      if (states) jobs = jobs.filter((j) => states.includes(j.status))
      if (limit) jobs = jobs.slice(Math.max(0, jobs.length - limit))
      return Promise.resolve({ jobs, total: jobs.length })
    },
    listWorkflows: () =>
      Promise.resolve({
        workflow_dir: '/ws/workflows',
        workflows: data.workflows,
        details: {},
      }),
    listAssets: () =>
      Promise.resolve({
        assets: data.assets,
        libraries: [],
        shadowed: [],
        folders: [],
        asset_dir: null,
        asset_dirs: [],
      }),
    galleryThumbnailUrl: (n: string) => `/thumb/${n}`,
    listWorkspaces: () =>
      Promise.resolve({
        workspace_root: '/ws',
        default: 'default',
        workspaces: [
          { name: 'default', default: true },
          { name: 'studio', default: false, usage: { files: 4, bytes: 4096 } },
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
  data.files = []
  data.jobs = []
  data.workflows = []
  data.assets = []
  data.galleryError = null
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

it('an empty workspace shows empty states, not empty frames', async () => {
  render(OverviewPage)
  await waitFor(() =>
    expect(screen.getByText(/nothing generated yet/i)).toBeTruthy(),
  )
  expect(document.querySelectorAll('.frame')).toHaveLength(0)
  expect(screen.getByText(/no jobs yet/i)).toBeTruthy()
  expect(screen.getByText(/no workflows yet/i)).toBeTruthy()
})

it('a failed panel shows its own error, and the rest still load', async () => {
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
    expect(
      screen.getByText(/could not load recent outputs: boom/i),
    ).toBeTruthy(),
  )
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
  await waitFor(() =>
    expect(document.querySelectorAll('.frame').length).toBe(1),
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

it('the default workspace cannot be deleted', async () => {
  location.hash = '#/ws/default/overview'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  render(OverviewPage)
  await waitFor(() =>
    expect(
      screen.getByRole('button', { name: /delete workspace/i }),
    ).toBeTruthy(),
  )
  expect(
    (
      screen.getByRole('button', {
        name: /delete workspace/i,
      }) as HTMLButtonElement
    ).disabled,
  ).toBe(true)
})
