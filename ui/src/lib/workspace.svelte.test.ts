import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const listWorkspaces = vi.hoisted(() =>
  vi.fn(() =>
    Promise.resolve({
      workspace_root: '/ws',
      default: 'default',
      workspaces: [
        {
          name: 'default',
          default: true,
          workflows: '/ws/workflows',
          assets: null,
          outputs: '/ws/outputs',
          prompts: null,
        },
        {
          name: 'studio',
          default: false,
          workflows: '/ws/studio/workflows',
          assets: null,
          outputs: '/ws/studio/outputs',
          prompts: null,
        },
      ],
    }),
  ),
)
const errorToast = vi.hoisted(() => vi.fn())
vi.mock('./api', () => ({ api: { listWorkspaces: () => listWorkspaces() } }))
vi.mock('./toast', () => ({ notify: { error: errorToast, success: vi.fn() } }))

beforeEach(() => {
  listWorkspaces.mockClear()
  errorToast.mockClear()
  localStorage.clear()
  location.hash = ''
})
afterEach(() => vi.resetModules())

it('shares one fetch across concurrent and repeated calls', async () => {
  const { loadWorkspaces, workspace } = await import('./workspace.svelte')
  await Promise.all([loadWorkspaces(), loadWorkspaces()])
  expect(listWorkspaces).toHaveBeenCalledTimes(1)
  expect(workspace.names).toEqual(['default', 'studio'])
})

it('refetches only after invalidateWorkspaces()', async () => {
  const { invalidateWorkspaces, loadWorkspaces } =
    await import('./workspace.svelte')
  await loadWorkspaces()
  await loadWorkspaces()
  expect(listWorkspaces).toHaveBeenCalledTimes(1)
  invalidateWorkspaces()
  await loadWorkspaces()
  expect(listWorkspaces).toHaveBeenCalledTimes(2)
})

it('a ws route sets current and is remembered as last used', async () => {
  const { applyRouteWorkspace, lastUsedWorkspace, workspace } =
    await import('./workspace.svelte')
  applyRouteWorkspace({
    kind: 'ws',
    workspace: 'studio',
    section: 'gallery',
    rest: [],
  })
  expect(workspace.current).toBe('studio')
  expect(lastUsedWorkspace()).toBe('studio')
  expect(localStorage.getItem('dw-workspace')).toBe('studio')
})

it('the default workspace clears the stored last-used name', async () => {
  localStorage.setItem('dw-workspace', 'studio')
  const { applyRouteWorkspace, lastUsedWorkspace } =
    await import('./workspace.svelte')
  applyRouteWorkspace({
    kind: 'ws',
    workspace: 'default',
    section: 'overview',
    rest: [],
  })
  expect(localStorage.getItem('dw-workspace')).toBeNull()
  expect(lastUsedWorkspace()).toBe('default')
})

it('a shared or server route keeps scoping to the last-used workspace', async () => {
  localStorage.setItem('dw-workspace', 'studio')
  const { applyRouteWorkspace, workspace } = await import('./workspace.svelte')
  applyRouteWorkspace({ kind: 'shared', section: 'prompts', rest: [] })
  expect(workspace.current).toBe('studio')
  applyRouteWorkspace({ kind: 'server', section: 'models' })
  expect(workspace.current).toBe('studio')
})

it('an unknown workspace in the route redirects to default and says so', async () => {
  location.hash = '#/ws/gone/gallery'
  // The router is what applies the hash to `workspace.current`
  await import('./router.svelte')
  const { loadWorkspaces } = await import('./workspace.svelte')
  await loadWorkspaces()
  expect(location.hash).toBe('#/ws/default/overview')
  expect(errorToast).toHaveBeenCalledWith(expect.stringContaining('gone'))
})

it('a known workspace in the route is left alone', async () => {
  location.hash = '#/ws/studio/gallery'
  await import('./router.svelte')
  const { loadWorkspaces } = await import('./workspace.svelte')
  await loadWorkspaces()
  expect(location.hash).toBe('#/ws/studio/gallery')
  expect(errorToast).not.toHaveBeenCalled()
})
