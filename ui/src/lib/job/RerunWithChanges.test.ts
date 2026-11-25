import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import RerunWithChanges from './RerunWithChanges.svelte'
import { ApiError } from '../api'
import { api } from '../test/apiMock'
import { go } from '../router.svelte'
import type { JobDetail } from '../types'

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
api.listAssets.mockImplementation(() => new Promise(() => {}))
api.gallery.mockImplementation(() => new Promise(() => {}))
vi.mock('../router.svelte', () => ({ go: vi.fn() }))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

const job = (over: Partial<JobDetail> = {}): JobDetail =>
  ({
    id: 'orig',
    workflow: 'tiny',
    workflow_name: 'tiny',
    workspace: 'default',
    status: 'succeeded',
    arguments: { prompt: 'a cat', steps: 30 },
    ...over,
  }) as JobDetail

const declared = { prompt: 'a dog', steps: 20, seed: 7, size: 'small' }

const presets = {
  size: {
    small: { width: 512, height: 512 },
    large: { width: 768, height: 768 },
    huge: { width: 1024, height: 1024 },
  },
}

beforeEach(() => {
  api.rerunJob.mockReset()
  api.rerunJob.mockResolvedValue({ id: 'next' } as JobDetail)
  api.getWorkflow.mockReset()
  api.getWorkflow.mockResolvedValue({
    definition: { variables: declared, presets } as never,
    origin: 'workspace',
    writable: true,
    etag: null,
  })
  api.listWorkflows.mockReset()
  api.listWorkflows.mockResolvedValue({ details: {} } as never)
  vi.mocked(go).mockReset()
})
afterEach(cleanup)

const field = (name: string) =>
  document.getElementById(`rerun-var-${name}`) as HTMLInputElement

async function openForm() {
  await fireEvent.click(screen.getByText('Run again with changes…'))
  await waitFor(() => expect(field('prompt')).toBeTruthy())
}

const queueButton = () =>
  screen.getByRole('button', { name: /^Queue/ }) as HTMLButtonElement

it("opens holding the job's values", async () => {
  render(RerunWithChanges, { job: job(), declared })
  await openForm()
  expect(field('prompt').value).toBe('a cat')
  expect(field('steps').value).toBe('30')
  expect(field('seed').value).toBe('7')
  expect(field('size').value).toBe('small')
  expect(screen.getByText(/Each field holds what this job used/)).toBeTruthy()
})

it('nothing changed and no new seed keeps Queue disabled', async () => {
  render(RerunWithChanges, { job: job(), declared })
  await openForm()
  expect(queueButton().disabled).toBe(true)
  expect(screen.getByText('nothing changed')).toBeTruthy()
})

it('keeps typed edits and New seed when the job refreshes while open', async () => {
  const { rerender } = render(RerunWithChanges, { job: job(), declared })
  await openForm()
  await fireEvent.input(field('prompt'), { target: { value: 'a fox' } })
  await fireEvent.click(screen.getByLabelText('New seed'))
  await rerender({
    job: job({ arguments: { prompt: 'a cat', steps: 31 } }),
    declared: { ...declared, extra: 1 },
  })
  expect(field('prompt').value).toBe('a fox')
  expect((screen.getByLabelText('New seed') as HTMLInputElement).checked).toBe(
    true,
  )
})

it('sends only what changed, typed', async () => {
  render(RerunWithChanges, { job: job(), declared })
  await openForm()
  await fireEvent.input(field('prompt'), { target: { value: 'a fox' } })
  await fireEvent.input(field('seed'), { target: { value: '9' } })
  // retyped as the run had it: a number, and a string with padding
  await fireEvent.input(field('steps'), { target: { value: '30' } })
  await fireEvent.input(field('size'), { target: { value: ' small ' } })
  expect(queueButton().disabled).toBe(false)
  await fireEvent.click(queueButton())
  await waitFor(() =>
    expect(go).toHaveBeenCalledWith('ws', 'default', 'jobs', 'next'),
  )
  expect(api.rerunJob).toHaveBeenCalledWith('orig', {
    newSeed: false,
    arguments: { prompt: 'a fox', seed: 9 },
  })
})

it('New seed alone queues a rerun with a new seed', async () => {
  render(RerunWithChanges, { job: job(), declared })
  await openForm()
  await fireEvent.click(screen.getByLabelText('New seed'))
  expect(queueButton().disabled).toBe(false)
  await fireEvent.click(queueButton())
  await waitFor(() => expect(go).toHaveBeenCalled())
  expect(api.rerunJob).toHaveBeenCalledWith('orig', {
    newSeed: true,
    arguments: {},
  })
})

it('a double click sends one request', async () => {
  render(RerunWithChanges, { job: job(), declared })
  await openForm()
  await fireEvent.input(field('prompt'), { target: { value: 'a fox' } })
  // Both clicks land before the page renders the first one's busy state
  queueButton().click()
  queueButton().click()
  await waitFor(() => expect(go).toHaveBeenCalled())
  expect(api.rerunJob).toHaveBeenCalledTimes(1)
})

it('sends nothing when the only typed value equals the run', async () => {
  render(RerunWithChanges, { job: job(), declared })
  await openForm()
  await fireEvent.input(field('steps'), { target: { value: '30' } })
  expect(queueButton().disabled).toBe(true)
  await fireEvent.click(queueButton())
  expect(api.rerunJob).not.toHaveBeenCalled()
})

it('shows a server refusal in the modal and stays open', async () => {
  api.rerunJob.mockRejectedValue(
    new ApiError("Unknown preset value 'giant' for size", 400),
  )
  render(RerunWithChanges, { job: job(), declared })
  await openForm()
  await fireEvent.input(field('size'), { target: { value: 'giant' } })
  await fireEvent.click(queueButton())
  await waitFor(() =>
    expect(screen.getByRole('alert').textContent).toContain(
      "Unknown preset value 'giant'",
    ),
  )
  expect(field('size')).toBeTruthy()
  expect(go).not.toHaveBeenCalled()
})

const promotionButtons = () =>
  [...document.querySelectorAll('[data-promotion]')] as HTMLButtonElement[]

it('offers one promotion per other value of a declared selector', async () => {
  render(RerunWithChanges, { job: job(), declared })
  await waitFor(() => expect(promotionButtons()).toHaveLength(2))
  expect(promotionButtons().map((b) => b.textContent)).toEqual([
    'size: large',
    'size: huge',
  ])
  expect(api.getWorkflow).toHaveBeenCalledWith('tiny')
  expect(promotionButtons()[0].title).toContain('as it is now')
  await fireEvent.click(promotionButtons()[0])
  await waitFor(() => expect(go).toHaveBeenCalled())
  expect(api.rerunJob).toHaveBeenCalledWith('orig', {
    arguments: { size: 'large' },
  })
})

it('skips the selected value when the run passed it explicitly', async () => {
  render(RerunWithChanges, {
    job: job({ arguments: { size: 'large' } }),
    declared,
  })
  await waitFor(() => expect(promotionButtons()).toHaveLength(2))
  expect(promotionButtons().map((b) => b.textContent)).toEqual([
    'size: small',
    'size: huge',
  ])
})

it('offers none for a selector the run does not declare', async () => {
  const undeclared: Record<string, unknown> = { ...declared }
  delete undeclared.size
  render(RerunWithChanges, { job: job(), declared: undeclared })
  await waitFor(() => expect(api.getWorkflow).toHaveBeenCalled())
  await Promise.resolve()
  expect(promotionButtons()).toHaveLength(0)
})

it('offers none when the workflow cannot be fetched by name', async () => {
  api.getWorkflow.mockRejectedValue(new ApiError('not found', 404))
  render(RerunWithChanges, { job: job(), declared })
  await waitFor(() => expect(api.getWorkflow).toHaveBeenCalled())
  await Promise.resolve()
  expect(promotionButtons()).toHaveLength(0)
  // the form is still there
  await openForm()
})

it('does not look up an inline workflow', async () => {
  render(RerunWithChanges, { job: job({ workflow_name: null }), declared })
  await openForm()
  expect(api.getWorkflow).not.toHaveBeenCalled()
  expect(promotionButtons()).toHaveLength(0)
})

it('notes explicit arguments the bundle would set', async () => {
  render(RerunWithChanges, {
    job: job({ arguments: { width: 640, prompt: 'a cat' } }),
    declared,
  })
  await waitFor(() => expect(promotionButtons()).toHaveLength(2))
  expect(promotionButtons()[0].title).toContain(
    'keeps width=640 set explicitly',
  )
})

it('an image variable can be swapped through the picker', async () => {
  api.listWorkflows.mockResolvedValue({
    workflows: [],
    details: { tiny: { inputs: { image: 'image' } } },
    libraries: [],
    shadowed: [],
  } as never)
  render(RerunWithChanges, {
    job: job({ arguments: { image: 'asset:cast/a.png' } }),
    declared: { image: 'asset:cast/a.png' },
  })
  await fireEvent.click(screen.getByText('Run again with changes…'))
  await waitFor(() =>
    expect(screen.getByRole('button', { name: /^Choose/ })).toBeTruthy(),
  )
  expect(api.listWorkflows).toHaveBeenCalledWith({ inputs: true })
  expect(screen.getByTitle(/^Upload an image/)).toBeTruthy()
  await fireEvent.input(field('image'), {
    target: { value: 'asset:cast/b.png' },
  })
  expect(queueButton().disabled).toBe(false)
})
