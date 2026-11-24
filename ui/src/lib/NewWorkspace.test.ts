import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import NewWorkspace from './NewWorkspace.svelte'
import { ApiError, describeError } from './apiErrors'
import { invalidateWorkspaces } from './workspace.svelte'

const createWorkspace = vi.hoisted(() => vi.fn())
vi.mock('./api', () => ({
  api: {
    listWorkspaces: () =>
      Promise.resolve({
        workspace_root: '/ws',
        default: 'default',
        workspaces: [
          { name: 'default', default: true },
          { name: 'fresh', default: false },
        ],
      }),
    createWorkspace: (n: string) => createWorkspace(n),
  },
}))
vi.mock('./toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

let onclose: ReturnType<typeof vi.fn<() => void>>
beforeEach(() => {
  createWorkspace.mockReset()
  invalidateWorkspaces()
  location.hash = '#/ws/default/gallery'
  onclose = vi.fn<() => void>()
})
afterEach(cleanup)

const field = () =>
  screen.getByRole('textbox', {
    name: 'new workspace name',
  }) as HTMLInputElement

async function typed(name: string) {
  render(NewWorkspace, { onclose })
  await fireEvent.input(field(), { target: { value: name } })
  return field()
}

it('opens with focus in the field', async () => {
  render(NewWorkspace, { onclose })
  await waitFor(() => expect(document.activeElement).toBe(field()))
})

it('Enter creates once, however often pressed while busy', async () => {
  let finish: (v: unknown) => void = () => {}
  createWorkspace.mockReturnValue(new Promise((r) => (finish = r)))
  const input = await typed('fresh')
  await fireEvent.keyDown(input, { key: 'Enter' })
  await fireEvent.keyDown(input, { key: 'Enter' })
  await fireEvent.keyDown(input, { key: 'Enter' })
  expect(createWorkspace).toHaveBeenCalledTimes(1)
  finish({ name: 'fresh' })
  await waitFor(() => expect(onclose).toHaveBeenCalled())
  expect(createWorkspace).toHaveBeenCalledTimes(1)
})

it('a refused name is said beside the field and the form stays open', async () => {
  const input = await typed('assets')
  await fireEvent.keyDown(input, { key: 'Enter' })
  // the local rule answers before any round trip
  await waitFor(() =>
    expect(screen.getByText('assets is reserved')).toBeTruthy(),
  )
  expect(createWorkspace).not.toHaveBeenCalled()
  expect(onclose).not.toHaveBeenCalled()
  // the draft stays, to be corrected
  expect(field().value).toBe('assets')
})

it("a name the server refuses says the server's reason, and keeps the draft", async () => {
  const refusal = new ApiError('workspace exists: taken', 409)
  createWorkspace.mockRejectedValue(refusal)
  const input = await typed('taken')
  await fireEvent.keyDown(input, { key: 'Enter' })
  await waitFor(() =>
    expect(screen.getByText(describeError(refusal).title)).toBeTruthy(),
  )
  expect(onclose).not.toHaveBeenCalled()
  expect(field().value).toBe('taken')
  expect(location.hash).toBe('#/ws/default/gallery')
})

it('a refusal after Create puts focus back in the field', async () => {
  const input = await typed('assets')
  const create = screen.getByRole('button', { name: 'Create' })
  create.focus()
  await fireEvent.click(create)
  await waitFor(() => expect(screen.getByText(/reserved/)).toBeTruthy())
  await waitFor(() => expect(document.activeElement).toBe(input))
})

it('Escape on a button closes it too', async () => {
  render(NewWorkspace, { onclose })
  await fireEvent.keyDown(screen.getByRole('button', { name: 'Cancel' }), {
    key: 'Escape',
  })
  expect(onclose).toHaveBeenCalledTimes(1)
})

it('Escape closes it', async () => {
  const input = await typed('half-typed')
  await fireEvent.keyDown(input, { key: 'Escape' })
  expect(onclose).toHaveBeenCalledTimes(1)
  expect(createWorkspace).not.toHaveBeenCalled()
})

it("success closes it and lands on the new workspace's overview", async () => {
  createWorkspace.mockResolvedValue({ name: 'fresh' })
  const input = await typed('fresh')
  await fireEvent.keyDown(input, { key: 'Enter' })
  await waitFor(() => expect(location.hash).toBe('#/ws/fresh/overview'))
  expect(onclose).toHaveBeenCalledTimes(1)
})

it('the name is trimmed before it is created', async () => {
  createWorkspace.mockResolvedValue({ name: 'fresh' })
  const input = await typed('  fresh ')
  await fireEvent.keyDown(input, { key: 'Enter' })
  await waitFor(() => expect(createWorkspace).toHaveBeenCalledWith('fresh'))
})

it('is aria-busy while creating', async () => {
  let finish: (v: unknown) => void = () => {}
  createWorkspace.mockReturnValue(new Promise((r) => (finish = r)))
  const input = await typed('fresh')
  const form = input.closest('form')!
  expect(form.getAttribute('aria-busy')).toBe('false')
  await fireEvent.keyDown(input, { key: 'Enter' })
  await waitFor(() => expect(form.getAttribute('aria-busy')).toBe('true'))
  finish({ name: 'fresh' })
  await waitFor(() => expect(onclose).toHaveBeenCalled())
})

it('Create creates', async () => {
  createWorkspace.mockResolvedValue({ name: 'fresh' })
  await typed('fresh')
  await fireEvent.click(screen.getByRole('button', { name: 'Create' }))
  await waitFor(() => expect(createWorkspace).toHaveBeenCalledWith('fresh'))
  await waitFor(() => expect(location.hash).toBe('#/ws/fresh/overview'))
})

it('Cancel closes', async () => {
  await typed('half-typed')
  await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  expect(onclose).toHaveBeenCalledTimes(1)
  expect(createWorkspace).not.toHaveBeenCalled()
})

it('a click away closes an empty field', async () => {
  render(NewWorkspace, { onclose })
  await fireEvent.focusOut(field(), { relatedTarget: document.body })
  expect(onclose).toHaveBeenCalledTimes(1)
})

it('a click away keeps a typed name', async () => {
  const input = await typed('half-typed')
  await fireEvent.focusOut(input, { relatedTarget: document.body })
  expect(onclose).not.toHaveBeenCalled()
  expect(field().value).toBe('half-typed')
})

it("focus moving to the form's own buttons does not close it", async () => {
  render(NewWorkspace, { onclose })
  await fireEvent.focusOut(field(), {
    relatedTarget: screen.getByRole('button', { name: 'Cancel' }),
  })
  expect(onclose).not.toHaveBeenCalled()
})
