import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const createWorkspace = vi.hoisted(() => vi.fn())
const deleteWorkspace = vi.hoisted(() => vi.fn())
const confirm = vi.hoisted(() => vi.fn())
const errorToast = vi.hoisted(() => vi.fn())
vi.mock('./api', () => ({
  api: {
    createWorkspace: (n: string) => createWorkspace(n),
    deleteWorkspace: (n: string, ack?: boolean) => deleteWorkspace(n, ack),
    listWorkspaces: vi.fn(() =>
      Promise.resolve({
        workspace_root: '/ws',
        default: 'default',
        workspaces: [{ name: 'default', default: true }],
      }),
    ),
  },
}))
vi.mock('./confirm.svelte', () => ({
  confirmDialog: (...a: unknown[]) => confirm(...a),
}))
vi.mock('./toast', () => ({
  notify: { error: errorToast, success: vi.fn() },
}))

beforeEach(() => {
  createWorkspace.mockReset()
  deleteWorkspace.mockReset()
  confirm.mockReset()
  errorToast.mockReset()
  localStorage.clear()
  location.hash = '#/ws/default/overview'
})
afterEach(() => vi.resetModules())

it('refuses reserved and malformed names before asking the server', async () => {
  const { workspaceNameError } = await import('./workspaceActions')
  expect(workspaceNameError('assets')).toMatch(/reserved/)
  expect(workspaceNameError('common')).toMatch(/reserved/)
  expect(workspaceNameError('')).toMatch(/name/)
  expect(workspaceNameError('a/b')).toMatch(/letters/)
  expect(workspaceNameError('studio-2')).toBeNull()
})

it('creates, then navigates to the new overview', async () => {
  createWorkspace.mockResolvedValue({ name: 'studio' })
  const { createWorkspaceAndGo } = await import('./workspaceActions')
  expect(await createWorkspaceAndGo('studio')).toBeNull()
  expect(createWorkspace).toHaveBeenCalledWith('studio')
  expect(location.hash).toBe('#/ws/studio/overview')
})

it('returns the server detail on failure and stays put', async () => {
  createWorkspace.mockRejectedValue(new Error('already exists'))
  const { createWorkspaceAndGo } = await import('./workspaceActions')
  expect(await createWorkspaceAndGo('studio')).toBe('already exists')
  expect(location.hash).toBe('#/ws/default/overview')
})

it('asks with the server detail, then deletes acknowledged and goes to default', async () => {
  deleteWorkspace
    .mockRejectedValueOnce(new Error('holds 3 files'))
    .mockResolvedValueOnce({ name: 'studio', deleted: true })
  confirm.mockResolvedValue(true)
  location.hash = '#/ws/studio/overview'
  // The router is what applies the hash to `workspace.current`, which is
  // the state the delete runs from in the app
  await import('./router.svelte')
  const { deleteWorkspaceWithConfirm } = await import('./workspaceActions')
  expect(await deleteWorkspaceWithConfirm('studio')).toBe(true)
  expect(confirm).toHaveBeenCalledWith(
    expect.stringContaining('holds 3 files'),
    expect.anything(),
  )
  expect(deleteWorkspace).toHaveBeenLastCalledWith('studio', true)
  expect(location.hash).toBe('#/ws/default/overview')
  // The listing is reloaded after the delete, while the route still named
  // the deleted workspace: that must not read as an unknown-workspace error
  expect(errorToast).not.toHaveBeenCalled()
})

it('a declined confirm deletes nothing', async () => {
  deleteWorkspace.mockRejectedValueOnce(new Error('holds 3 files'))
  confirm.mockResolvedValue(false)
  const { deleteWorkspaceWithConfirm } = await import('./workspaceActions')
  expect(await deleteWorkspaceWithConfirm('studio')).toBe(false)
  expect(deleteWorkspace).toHaveBeenCalledTimes(1)
})
