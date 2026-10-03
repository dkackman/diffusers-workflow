import { afterEach, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/svelte'
import StepEditor from './StepEditor.svelte'

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
