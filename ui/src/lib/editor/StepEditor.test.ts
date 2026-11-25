import { afterEach, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { api } from '../test/apiMock'
import { VARIABLE } from '../references'
import StepEditor from './StepEditor.svelte'
import type { PipelineDescription, WorkflowStep } from '../types'

// The LoRA editor asks the catalog about a literal base repo; keep it offline
vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
api.listLoras.mockResolvedValue({ libraries: [], loras: [] })

afterEach(() => cleanup())

// component_type '' keeps the class-description fetch from firing, so the
// step renders from its own definition alone
const step = () => ({
  name: 'generate',
  pipeline: {
    configuration: { component_type: '' },
    from_pretrained_arguments: {
      model_name: 'org/model',
      torch_dtype: 'torch.float8_e4m3fn',
    },
    arguments: {},
  },
  result: { content_type: 'audio/x-flac' },
})

it('shows a dtype and a content type it does not list as selected', () => {
  const { container } = render(StepEditor, {
    step: step(),
    workflow: {},
    index: 0,
    count: 1,
    onremove: () => {},
    onmove: () => {},
  })
  expect((screen.getByLabelText('dtype') as HTMLSelectElement).value).toBe(
    'torch.float8_e4m3fn',
  )
  expect(
    (container.querySelector('select[id^="result-"]') as HTMLSelectElement)
      .value,
  ).toBe('audio/x-flac')
})

// H3 sets only a shift on the pipeline's own schedulers - no configuration
// block - and the full view used to throw on it, leaving the compact digest
it('opens full on a step whose schedulers set only a shift', () => {
  const h3 = step()
  Object.assign(h3.pipeline, {
    scheduler: { shift: 'variable:video_shift' },
    audio_scheduler: { shift: 3 },
  })
  render(StepEditor, {
    step: h3,
    workflow: {},
    index: 0,
    count: 1,
    mode: 'full',
    onremove: () => {},
    onmove: () => {},
  })
  expect((screen.getByLabelText('model') as HTMLInputElement).value).toBe(
    'org/model',
  )
  expect(
    (document.getElementById('schedulershift-0') as HTMLInputElement).value,
  ).toBe('variable:video_shift')
  expect(
    (document.getElementById('audio_schedulershift-0') as HTMLInputElement)
      .value,
  ).toBe('3')
})

// templates give the cache as a reference string (`variable:cache`); the
// full view used to set `.type` on it and throw
it('opens full on a step whose cache is a variable reference', () => {
  const ref = VARIABLE + 'cache'
  const s = step()
  Object.assign(s.pipeline.configuration, { cache: ref })
  render(StepEditor, {
    step: s,
    workflow: {},
    index: 0,
    count: 1,
    mode: 'full',
    onremove: () => {},
    onmove: () => {},
  })
  expect(
    (screen.getByPlaceholderText(/a reference/) as HTMLInputElement).value,
  ).toBe(ref)
  expect((s.pipeline.configuration as { cache?: string }).cache).toBe(ref)
})

// Opening the full view is display only. A step with no torch_dtype (the
// minimax templates) used to get the first dtype option bound back in, which
// marked the workflow unsaved with no edit (#1496)
it('leaves a step with no torch_dtype unchanged on opening full', async () => {
  const s = step()
  Object.assign(s.pipeline.configuration, { cache: VARIABLE + 'cache' })
  s.pipeline.from_pretrained_arguments = {
    model_name: 'MiniMaxAI/MiniMax-H3',
    workflow: 'ref2va',
  } as unknown as typeof s.pipeline.from_pretrained_arguments
  const before = JSON.stringify(s)
  render(StepEditor, {
    step: s,
    workflow: {},
    index: 0,
    count: 1,
    mode: 'full',
    onremove: () => {},
    onmove: () => {},
  })
  await new Promise((r) => setTimeout(r, 50))
  expect(JSON.stringify(s)).toBe(before)
})

// The step's media arguments take the results of the steps before it
it('offers an image argument the results of the steps before it', async () => {
  const s = step()
  s.name = 'use'
  Object.assign(s.pipeline, { arguments: { image: '' } })
  const workflow = {
    steps: [
      { name: 'still', result: { content_type: 'image/png' } },
      s,
      { name: 'later', result: { content_type: 'image/png' } },
    ],
  }
  render(StepEditor, {
    step: s,
    index: 1,
    workflow,
    count: 3,
    mode: 'full',
    onremove: () => {},
    onmove: () => {},
  })
  const input = screen.getByPlaceholderText('type, drop or choose an image')
  await fireEvent.focus(input)
  await fireEvent.input(input, { target: { value: '' } })
  const text = (await screen.findAllByRole('option'))
    .map((o) => o.textContent)
    .join('\n')
  expect(text).toContain('previous_result:still')
  expect(text).not.toContain('previous_result:later')
})

it('a step removed during the pause never reads its sub-workflow', async () => {
  vi.useFakeTimers()
  try {
    api.getWorkflow.mockResolvedValue({ definition: { variables: { a: 'b' } } })
    const sub = {
      name: 'sub',
      workflow: { path: 'child.json', arguments: {} },
    }
    const { unmount } = render(StepEditor, {
      step: sub,
      workflow: {},
      index: 0,
      count: 1,
      onremove: () => {},
      onmove: () => {},
    })
    await vi.advanceTimersByTimeAsync(100)
    unmount()
    await vi.advanceTimersByTimeAsync(1000)
    expect(api.getWorkflow).not.toHaveBeenCalled()
  } finally {
    vi.useRealTimers()
  }
})

const full = (s: WorkflowStep) =>
  render(StepEditor, {
    step: s,
    workflow: {},
    index: 0,
    count: 1,
    mode: 'full',
    onremove: () => {},
    onmove: () => {},
  })

// `arguments` is optional in a definition, and the full view used to throw
// reading a key of it
it('opens full on a step with no arguments, and writes nothing', async () => {
  const s: WorkflowStep = {
    name: 'bare',
    pipeline: { configuration: { component_type: '' } },
  }
  const before = JSON.stringify(s)
  full(s)
  await new Promise((r) => setTimeout(r, 50))
  expect(screen.getByLabelText('model')).toBeInTheDocument()
  expect(JSON.stringify(s)).toBe(before)
})

// Typing into a block the definition does not have yet used to write to a
// throwaway object, so the edit was lost
it('keeps what is typed into a block the step did not have', async () => {
  const s: WorkflowStep = { name: 'bare', pipeline: {} }
  full(s)
  await fireEvent.input(screen.getByLabelText('model'), {
    target: { value: 'org/model' },
  })
  expect(s.pipeline?.from_pretrained_arguments).toEqual({
    model_name: 'org/model',
  })
  await fireEvent.change(screen.getByLabelText('attention backend'), {
    target: { value: 'flash' },
  })
  expect(s.pipeline?.configuration?.attention_backend).toBe('flash')
})

it('creates the arguments block with the first argument added', async () => {
  api.describeClass.mockResolvedValue({
    name: 'Probe',
    summary: '',
    parameters: [],
    accepts_kwargs: true,
  } as unknown as PipelineDescription)
  const s: WorkflowStep = {
    name: 'bare',
    pipeline: { configuration: { component_type: 'Probe' } },
  }
  full(s)
  const [add] = await screen.findAllByPlaceholderText('add custom argument…')
  await fireEvent.input(add, { target: { value: 'steps' } })
  await fireEvent.keyDown(add, { key: 'Enter' })
  expect(JSON.stringify(s.pipeline)).toContain('"steps":""')
})

// The schema allows a task that takes `inputs`, and a sub-workflow step with
// only a `path`: neither has an arguments object
it.each<[string, WorkflowStep]>([
  [
    'a task with inputs',
    { name: 'bare', task: { command: 'gather_images', inputs: ['a.png'] } },
  ],
  [
    'a sub-workflow with only a path',
    { name: 'bare', workflow: { path: 'c.json' } },
  ],
])('opens full on %s, and writes nothing', async (_, s) => {
  api.getWorkflow.mockResolvedValue({ definition: {} })
  const before = JSON.stringify(s)
  full(s)
  await new Promise((r) => setTimeout(r, 400))
  expect(screen.getByText('arguments')).toBeInTheDocument()
  expect(JSON.stringify(s)).toBe(before)
})

it("creates a task's arguments block with the first argument added", async () => {
  const s: WorkflowStep = {
    name: 'bare',
    task: { command: 'gather_images', inputs: ['a.png'] },
  }
  full(s)
  const add = await screen.findByPlaceholderText('add custom argument…')
  await fireEvent.input(add, { target: { value: 'urls' } })
  await fireEvent.keyDown(add, { key: 'Enter' })
  expect(s.task?.arguments).toEqual({ urls: '' })
})

it("creates a sub-workflow's arguments block with the first argument added", async () => {
  api.getWorkflow.mockResolvedValue({ definition: {} })
  const s: WorkflowStep = { name: 'bare', workflow: { path: 'c.json' } }
  full(s)
  const name = screen.getByPlaceholderText('new argument name…')
  await fireEvent.input(name, { target: { value: 'prompt' } })
  await fireEvent.click(screen.getByLabelText('add this argument'))
  expect(s.workflow?.arguments).toEqual({ prompt: '' })
})
