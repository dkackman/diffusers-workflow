import { afterEach, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/svelte'
import { VARIABLE } from '../references'
import StepEditor from './StepEditor.svelte'

// The LoRA editor asks the catalog about a literal base repo; keep it offline
vi.mock('../api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api')>()
  return {
    ...actual,
    api: {
      ...actual.api,
      listLoras: () => Promise.resolve({ libraries: [], loras: [] }),
    },
  }
})

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
    index: 0,
    count: 1,
    mode: 'full',
    onremove: () => {},
    onmove: () => {},
  })
  expect(
    (screen.getByPlaceholderText(/a reference/) as HTMLInputElement).value,
  ).toBe(ref)
  expect((s.pipeline.configuration as { cache?: string }).cache).toBe(
    ref,
  )
})
