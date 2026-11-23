import { afterEach, beforeEach, expect, it, vi } from 'vitest'
// Read by tests/test_ui_twins.py too, so the two rules cannot drift apart
import cases from '../../../tests/fixtures/workspace_names.json'

const createWorkspace = vi.hoisted(() => vi.fn())
const deleteWorkspace = vi.hoisted(() => vi.fn())
const confirm = vi.hoisted(() => vi.fn())
const errorToast = vi.hoisted(() => vi.fn())
const failureToast = vi.hoisted(() => vi.fn())
// Modules are reset between tests, so the class is read fresh each time
const apiError = async (message: string, status: number) =>
  new (await import('./apiErrors')).ApiError(message, status)
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
  notify: { error: errorToast, success: vi.fn(), failure: failureToast },
}))

beforeEach(() => {
  createWorkspace.mockReset()
  deleteWorkspace.mockReset()
  confirm.mockReset()
  errorToast.mockReset()
  failureToast.mockReset()
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
  createWorkspace.mockRejectedValue(await apiError('already exists', 409))
  const { createWorkspaceAndGo } = await import('./workspaceActions')
  expect(await createWorkspaceAndGo('studio')).toBe('already exists')
  expect(location.hash).toBe('#/ws/default/overview')
})

it('asks with the server detail, then deletes acknowledged and goes to default', async () => {
  deleteWorkspace
    .mockRejectedValueOnce(await apiError('holds 3 files', 409))
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
  deleteWorkspace.mockRejectedValueOnce(await apiError('holds 3 files', 409))
  confirm.mockResolvedValue(false)
  const { deleteWorkspaceWithConfirm } = await import('./workspaceActions')
  expect(await deleteWorkspaceWithConfirm('studio')).toBe(false)
  expect(deleteWorkspace).toHaveBeenCalledTimes(1)
})

it.each(cases)('decides $name as the engine does', async ({ name, valid }) => {
  const { workspaceNameError } = await import('./workspaceActions')
  expect(workspaceNameError(name) === null).toBe(valid)
})

it('an unreachable server is reported, not put in a confirm dialog', async () => {
  deleteWorkspace.mockRejectedValueOnce(new TypeError('Failed to fetch'))
  const { deleteWorkspaceWithConfirm } = await import('./workspaceActions')
  expect(await deleteWorkspaceWithConfirm('studio')).toBe(false)
  expect(confirm).not.toHaveBeenCalled()
  expect(failureToast).toHaveBeenCalled()
})

it('a failed create returns the described title, not the raw message', async () => {
  createWorkspace.mockRejectedValueOnce(new TypeError('Failed to fetch'))
  const { createWorkspaceAndGo } = await import('./workspaceActions')
  expect(await createWorkspaceAndGo('studio')).toBe("Can't reach the server")
})
