import { afterEach, beforeEach, expect, it, vi } from 'vitest'
// Read by tests/test_ui_twins.py too, so the two rules cannot drift apart
import cases from '../../../tests/fixtures/workspace_names.json'
import { api } from './test/apiMock'

const confirm = vi.hoisted(() => vi.fn())
const errorToast = vi.hoisted(() => vi.fn())
const failureToast = vi.hoisted(() => vi.fn())
// Modules are reset between tests, so the class is read fresh each time
const apiError = async (message: string, status: number) =>
  new (await import('./apiErrors')).ApiError(message, status)
vi.mock('./api', async () => (await import('./test/apiMock')).apiModule())
const { createWorkspace, deleteWorkspace, gallery } = api
const listed = () =>
  Promise.resolve({
    workspace_root: '/ws',
    default: 'default',
    workspaces: [{ name: 'default', default: true }],
  })
vi.mock('./confirm.svelte', () => ({
  confirmDialog: (...a: unknown[]) => confirm(...a),
}))
vi.mock('./toast', () => ({
  notify: { error: errorToast, success: vi.fn(), failure: failureToast },
}))

beforeEach(() => {
  createWorkspace.mockReset()
  deleteWorkspace.mockReset()
  gallery.mockReset()
  gallery.mockResolvedValue({ files: [] })
  api.listWorkspaces.mockImplementation(listed)
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

/** The server's unacknowledged refusal: what the workspace holds */
const holds = async (contents: Record<string, unknown>) =>
  new (await import('./apiErrors')).ApiError(
    "Deleting workspace 'studio' removes these files permanently.",
    409,
    undefined,
    { message: 'removes these files permanently', contents },
  )

it("asks with the counts from the server's refusal and deletes only on yes", async () => {
  deleteWorkspace
    .mockRejectedValueOnce(
      await holds({
        workflows: { files: 3, bytes: 1024 * 1024 },
        assets: { files: 0, bytes: 0 },
        outputs: { files: 5, bytes: 1024 * 1024 },
      }),
    )
    .mockResolvedValueOnce({ name: 'studio', deleted: true })
  gallery.mockResolvedValue({
    files: [{ name: 'a.png', starred: true }, { name: 'b.png' }],
  })
  confirm.mockResolvedValue(true)
  location.hash = '#/ws/studio/overview'
  // The router is what applies the hash to `workspace.current`, which is
  // the state the delete runs from in the app
  await import('./router.svelte')
  const { deleteWorkspaceWithConfirm } = await import('./workspaceActions')
  expect(await deleteWorkspaceWithConfirm('studio')).toBe(true)
  expect(confirm).toHaveBeenCalledWith(
    'This permanently removes 5 outputs (1 starred) and 3 workflows ' +
      '(2.0 MB). The prompt library and the shared asset library are kept.',
    expect.objectContaining({
      title: 'Delete workspace "studio"?',
      confirmLabel: 'Delete workspace',
      danger: true,
    }),
  )
  expect(deleteWorkspace).toHaveBeenLastCalledWith('studio', true)
  expect(location.hash).toBe('#/ws/default/overview')
  // The listing is reloaded after the delete, while the route still named
  // the deleted workspace: that must not read as an unknown-workspace error
  expect(errorToast).not.toHaveBeenCalled()

  // Declined, nothing past the unacknowledged ask is sent
  deleteWorkspace.mockReset()
  deleteWorkspace.mockRejectedValueOnce(await holds({}))
  confirm.mockResolvedValue(false)
  expect(await deleteWorkspaceWithConfirm('studio')).toBe(false)
  expect(deleteWorkspace).toHaveBeenCalledTimes(1)
  // the first try carries no acknowledgement
  expect(deleteWorkspace).toHaveBeenCalledWith('studio')
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
  deleteWorkspace.mockRejectedValueOnce(
    new (await import('./apiErrors')).NetworkError('Failed to fetch'),
  )
  const { deleteWorkspaceWithConfirm } = await import('./workspaceActions')
  expect(await deleteWorkspaceWithConfirm('studio')).toBe(false)
  expect(confirm).not.toHaveBeenCalled()
  expect(failureToast).toHaveBeenCalled()
})

it('a failed create returns the described title, not the raw message', async () => {
  createWorkspace.mockRejectedValueOnce(
    new (await import('./apiErrors')).NetworkError('Failed to fetch'),
  )
  const { createWorkspaceAndGo } = await import('./workspaceActions')
  expect(await createWorkspaceAndGo('studio')).toBe("Can't reach the server")
})

it('reads the contents off the payload, as send builds the refusal', async () => {
  const { ApiError, errorDetail, errorPayload } = await import('./apiErrors')
  const payload = {
    detail: {
      message: "Deleting workspace 'studio' removes these files permanently.",
      contents: { outputs: { files: 3, bytes: 5 * 1024 * 1024 } },
    },
  }
  deleteWorkspace
    .mockRejectedValueOnce(
      new ApiError(
        errorDetail(payload, 'refused'),
        409,
        undefined,
        errorPayload(payload),
      ),
    )
    .mockResolvedValueOnce({ name: 'studio', deleted: true })
  confirm.mockResolvedValue(true)
  const { deleteWorkspaceWithConfirm } = await import('./workspaceActions')
  await deleteWorkspaceWithConfirm('studio')
  expect(confirm).toHaveBeenCalledWith(
    expect.stringContaining('removes 3 outputs (5.0 MB).'),
    expect.anything(),
  )
})

it('a gallery that cannot be read still asks, without the count', async () => {
  deleteWorkspace
    .mockRejectedValueOnce(await holds({ outputs: { files: 2, bytes: 1 } }))
    .mockResolvedValueOnce({ name: 'studio', deleted: true })
  gallery.mockRejectedValueOnce(
    new (await import('./apiErrors')).NetworkError('Failed to fetch'),
  )
  confirm.mockResolvedValue(true)
  const { deleteWorkspaceWithConfirm } = await import('./workspaceActions')
  expect(await deleteWorkspaceWithConfirm('studio')).toBe(true)
  expect(confirm).toHaveBeenCalledWith(
    expect.stringContaining("The starred count couldn't be read."),
    expect.anything(),
  )
  expect(failureToast).not.toHaveBeenCalled()
  expect(deleteWorkspace).toHaveBeenCalledTimes(2)
})
