import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { api } from './test/apiMock'

// Workspaces the server gained after the page loaded - a test pushes to it
const created = vi.hoisted((): string[] => [])
const errorToast = vi.hoisted(() => vi.fn())
vi.mock('./api', async () => (await import('./test/apiMock')).apiModule())
api.listWorkspaces.mockImplementation(() =>
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
      ...created.map((name) => ({
        name,
        default: false,
        workflows: `/ws/${name}/workflows`,
        assets: null,
        outputs: `/ws/${name}/outputs`,
        prompts: null,
      })),
    ],
  }),
)
vi.mock('./toast', () => ({ notify: { error: errorToast, success: vi.fn() } }))

beforeEach(() => {
  api.listWorkspaces.mockClear()
  errorToast.mockClear()
  created.length = 0
  localStorage.clear()
  location.hash = ''
})
afterEach(() => vi.resetModules())

it('shares one fetch across concurrent and repeated calls', async () => {
  const { loadWorkspaces, workspace } = await import('./workspace.svelte')
  await Promise.all([loadWorkspaces(), loadWorkspaces()])
  expect(api.listWorkspaces).toHaveBeenCalledTimes(1)
  expect(workspace.names).toEqual(['default', 'studio'])
})

it('refetches only after invalidateWorkspaces()', async () => {
  const { invalidateWorkspaces, loadWorkspaces } =
    await import('./workspace.svelte')
  await loadWorkspaces()
  await loadWorkspaces()
  expect(api.listWorkspaces).toHaveBeenCalledTimes(1)
  invalidateWorkspaces()
  await loadWorkspaces()
  expect(api.listWorkspaces).toHaveBeenCalledTimes(2)
})

it('a reconnect replaces the fallback a locked start left', async () => {
  const { loadWorkspaces, workspace } = await import('./workspace.svelte')
  const { reportResponse } = await import('./connection.svelte')
  api.listWorkspaces.mockImplementationOnce(() =>
    Promise.reject(new Error('401')),
  )
  reportResponse(401)
  await loadWorkspaces()
  expect(workspace.names).toEqual(['default'])
  reportResponse(200)
  await vi.waitFor(() => expect(workspace.names).toEqual(['default', 'studio']))
  expect(api.listWorkspaces).toHaveBeenCalledTimes(2)
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
  const { loadWorkspaces, workspace } = await import('./workspace.svelte')
  await loadWorkspaces()
  expect(location.hash).toBe('#/ws/default/overview')
  await vi.waitFor(() => expect(workspace.current).toBe('default'))
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

it('a shared route with a stale last-used name resets to default silently', async () => {
  localStorage.setItem('dw-workspace', 'gone')
  location.hash = '#/shared/prompts'
  await import('./router.svelte')
  const { loadWorkspaces, workspace } = await import('./workspace.svelte')
  expect(workspace.current).toBe('gone')
  await loadWorkspaces()
  // The URL never named the workspace, so nothing to say and nowhere to go:
  // the fallback is quietly replaced
  expect(workspace.current).toBe('default')
  expect(localStorage.getItem('dw-workspace')).toBeNull()
  expect(location.hash).toBe('#/shared/prompts')
  expect(errorToast).not.toHaveBeenCalled()
})

it('a missing workspace entered after the listing landed still redirects', async () => {
  location.hash = '#/ws/studio/gallery'
  await import('./router.svelte')
  const { loadWorkspaces } = await import('./workspace.svelte')
  await loadWorkspaces()
  expect(errorToast).not.toHaveBeenCalled()
  location.hash = '#/ws/typo/gallery'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  // Refetched once before it is called missing - see the next test
  await loadWorkspaces()
  expect(location.hash).toBe('#/ws/default/overview')
  expect(errorToast).toHaveBeenCalledWith(expect.stringContaining('typo'))
})

it('a workspace created after the listing landed is refetched, not called missing', async () => {
  location.hash = '#/ws/studio/gallery'
  await import('./router.svelte')
  const { loadWorkspaces, workspace } = await import('./workspace.svelte')
  await loadWorkspaces()
  // Created elsewhere (an agent's create_workspace) after this page loaded -
  // the running chip's job link then corrects the URL into it
  created.push('fresh')
  location.hash = '#/ws/fresh/jobs/abc'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  await loadWorkspaces()
  expect(workspace.names).toContain('fresh')
  expect(location.hash).toBe('#/ws/fresh/jobs/abc')
  expect(workspace.current).toBe('fresh')
  expect(errorToast).not.toHaveBeenCalled()
})

it('a failed listing does not read as every other workspace being gone', async () => {
  api.listWorkspaces.mockRejectedValueOnce(new Error('down'))
  location.hash = '#/ws/studio/gallery'
  await import('./router.svelte')
  const { loadWorkspaces, workspace } = await import('./workspace.svelte')
  await loadWorkspaces()
  expect(workspace.names).toEqual(['default'])
  location.hash = '#/ws/studio/jobs'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  expect(location.hash).toBe('#/ws/studio/jobs')
  expect(workspace.current).toBe('studio')
  expect(errorToast).not.toHaveBeenCalled()
})

it('a refetch that fails keeps the listing and still redirects the missing name', async () => {
  // Driven through applyRouteWorkspace rather than a hashchange: the routers
  // earlier tests imported still listen, and one would take the rejection.
  // With no router loaded, the replace it would hand over is a spy
  const { applyRouteWorkspace, loadWorkspaces, setRouteReplacer, workspace } =
    await import('./workspace.svelte')
  const replace = vi.fn()
  setRouteReplacer(replace)
  applyRouteWorkspace({
    kind: 'ws',
    workspace: 'studio',
    section: 'gallery',
    rest: [],
  })
  await loadWorkspaces()
  expect(workspace.names).toEqual(['default', 'studio'])
  // The refetch a name the cache lacks triggers hits a server hiccup
  api.listWorkspaces.mockRejectedValueOnce(new Error('down'))
  applyRouteWorkspace({
    kind: 'ws',
    workspace: 'typo',
    section: 'gallery',
    rest: [],
  })
  await loadWorkspaces()
  expect(workspace.names).toEqual(['default', 'studio'])
  expect(replace).toHaveBeenCalledWith('#/ws/default/overview')
  // `current` is the route's to set, and the spy lands none
  expect(workspace.current).toBe('typo')
  expect(errorToast).toHaveBeenCalledWith(expect.stringContaining('typo'))
})

it('a first listing that fails falls back to the default alone, until one lands', async () => {
  api.listWorkspaces.mockRejectedValueOnce(new Error('down'))
  const { invalidateWorkspaces, loadWorkspaces, workspace } =
    await import('./workspace.svelte')
  await loadWorkspaces()
  expect(workspace.names).toEqual(['default'])
  invalidateWorkspaces()
  await loadWorkspaces()
  expect(workspace.names).toEqual(['default', 'studio'])
})
