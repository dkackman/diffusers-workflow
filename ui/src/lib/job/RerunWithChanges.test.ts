import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import RerunWithChanges from './RerunWithChanges.svelte'
import { ApiError, api } from '../api'
import { go } from '../router.svelte'
import type { JobDetail } from '../types'

vi.mock('../api', async (original) => {
  const real = await original<typeof import('../api')>()
  return {
    ApiError: real.ApiError,
    api: { rerunJob: vi.fn(), getWorkflow: vi.fn() },
  }
})
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
  vi.mocked(api.rerunJob).mockReset()
  vi.mocked(api.rerunJob).mockResolvedValue({ id: 'next' } as JobDetail)
  vi.mocked(api.getWorkflow).mockReset()
  vi.mocked(api.getWorkflow).mockResolvedValue({
    definition: { variables: declared, presets } as never,
    origin: 'workspace',
    writable: true,
  })
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

it('lists every declared variable with the run value as its placeholder', async () => {
  render(RerunWithChanges, { job: job(), declared })
  await openForm()
  expect(field('prompt').placeholder).toBe('a cat')
  expect(field('steps').placeholder).toBe('30')
  expect(field('seed').placeholder).toBe('7')
  expect(field('size').placeholder).toBe('small')
  expect(field('prompt').value).toBe('')
})

it('disables Queue until something changed', async () => {
  render(RerunWithChanges, { job: job(), declared })
  await openForm()
  expect(queueButton().disabled).toBe(true)
  expect(screen.getByText('nothing changed')).toBeTruthy()
})

it('sends only the changed keys and opens the new job', async () => {
  render(RerunWithChanges, { job: job(), declared })
  await openForm()
  await fireEvent.input(field('prompt'), { target: { value: 'a fox' } })
  // retyped as the run had it: a number, and a string with padding
  await fireEvent.input(field('steps'), { target: { value: '30' } })
  await fireEvent.input(field('size'), { target: { value: ' small ' } })
  expect(queueButton().disabled).toBe(false)
  await fireEvent.click(queueButton())
  await waitFor(() =>
    expect(go).toHaveBeenCalledWith('ws', 'default', 'jobs', 'next'),
  )
  expect(api.rerunJob).toHaveBeenCalledWith('orig', {
    arguments: { prompt: 'a fox' },
  })
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
  vi.mocked(api.rerunJob).mockRejectedValue(
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
  vi.mocked(api.getWorkflow).mockRejectedValue(new ApiError('not found', 404))
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
