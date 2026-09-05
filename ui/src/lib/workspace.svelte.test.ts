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
      ],
    }),
  ),
)
vi.mock('./api', () => ({ api: { listWorkspaces: () => listWorkspaces() } }))

beforeEach(() => {
  listWorkspaces.mockClear()
  try {
    localStorage.clear()
  } catch {
    /* not needed here, but harmless if it throws */
  }
})

afterEach(() => {
  vi.resetModules()
})

it('shares one fetch across concurrent and repeated calls', async () => {
  // Fresh module per test: the cached promise is module-level state, and a
  // prior test's fetch must not satisfy this one
  const { loadWorkspaces, workspace } = await import('./workspace.svelte')
  await Promise.all([loadWorkspaces(), loadWorkspaces()])
  expect(listWorkspaces).toHaveBeenCalledTimes(1)
  expect(workspace.names).toEqual(['default'])
})

it('refetches only after invalidateWorkspaces()', async () => {
  const { invalidateWorkspaces, loadWorkspaces } = await import(
    './workspace.svelte'
  )
  await loadWorkspaces()
  await loadWorkspaces()
  expect(listWorkspaces).toHaveBeenCalledTimes(1)

  invalidateWorkspaces()
  await loadWorkspaces()
  expect(listWorkspaces).toHaveBeenCalledTimes(2)
})
