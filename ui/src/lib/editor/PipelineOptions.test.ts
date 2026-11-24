import { afterEach, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import PipelineOptionsHarness from './PipelineOptionsHarness.svelte'
import PipelineOptions from './PipelineOptions.svelte'
import { api } from '../api'
import type { PipelineDescription } from '../types'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

it("names a scheduler's interchangeable classes once its name settles", async () => {
  const describe = vi.spyOn(api, 'describeClass').mockResolvedValue({
    parameters: [],
    accepts_kwargs: true,
    compatibles: ['EulerProbeScheduler', 'HeunProbeScheduler'],
  } as unknown as PipelineDescription)
  render(PipelineOptions, {
    pipeline: {
      scheduler: { configuration: { scheduler_type: 'DPMProbeScheduler' } },
    },
    index: 0,
    openSection: '',
    references: [],
  })
  expect(
    await screen.findByText(
      'interchangeable with: EulerProbeScheduler, HeunProbeScheduler',
    ),
  ).toBeInTheDocument()
  expect(describe).toHaveBeenCalledWith('DPMProbeScheduler', 'init')
})

const sdDescription = {
  name: 'StableDiffusionPipeline',
  summary: '',
  accepts_kwargs: false,
  parameters: [
    { name: 'vae', annotation: "<class 'diffusers.AutoencoderKL'>" },
    { name: 'unet', annotation: "<class 'diffusers.UNet2DConditionModel'>" },
    { name: 'requires_safety_checker', annotation: "<class 'bool'>" },
  ],
} as unknown as PipelineDescription

it('offers the slots StableDiffusionPipeline takes', async () => {
  const describe = vi
    .spyOn(api, 'describeClass')
    .mockResolvedValue(sdDescription)
  render(PipelineOptions, {
    pipeline: {
      configuration: { component_type: 'StableDiffusionPipeline' },
      vae: { configuration: { component_type: 'AutoencoderKL' } },
    },
    index: 0,
    openSection: 'components',
    references: [],
  })
  await vi.waitFor(() =>
    expect(describe).toHaveBeenCalledWith('StableDiffusionPipeline', 'init'),
  )
  const select = screen.getByRole('combobox', { name: '' })
  await vi.waitFor(() =>
    expect([...select.querySelectorAll('option')].map((o) => o.value)).toEqual([
      '',
      'unet',
    ]),
  )
})

it('a slot the pipeline lacks is marked', async () => {
  vi.spyOn(api, 'describeClass').mockResolvedValue(sdDescription)
  render(PipelineOptions, {
    pipeline: {
      configuration: { component_type: 'StableDiffusionPipeline' },
      transformer: { configuration: { component_type: 'SomeTransformer' } },
    },
    index: 0,
    openSection: 'components',
    references: [],
  })
  expect(
    await screen.findByText('not a component of StableDiffusionPipeline'),
  ).toHaveClass('warn')
})

it('a constructor slot added from the pipeline shows, removes and leaves the select', async () => {
  vi.spyOn(api, 'describeClass').mockResolvedValue({
    ...sdDescription,
    parameters: [
      ...sdDescription.parameters,
      {
        name: 'safety_checker',
        annotation: "<class 'diffusers.StableDiffusionSafetyChecker'>",
      },
    ],
  } as unknown as PipelineDescription)
  render(PipelineOptionsHarness, {
    initial: { configuration: { component_type: 'SafetyProbePipeline' } },
  })
  const select = screen.getByRole('combobox', { name: '' })
  await vi.waitFor(() =>
    expect(
      [...select.querySelectorAll('option')].map((o) => o.value),
    ).toContain('safety_checker'),
  )
  await fireEvent.change(select, { target: { value: 'safety_checker' } })
  await fireEvent.click(screen.getByRole('button', { name: 'add' }))
  expect(screen.getByText('safety_checker')).toBeInTheDocument()
  expect(
    [...select.querySelectorAll('option')].map((o) => o.value),
  ).not.toContain('safety_checker')
  await fireEvent.click(screen.getByRole('button', { name: /remove/i }))
  expect([...select.querySelectorAll('option')].map((o) => o.value)).toContain(
    'safety_checker',
  )
})

it('shows a component key outside the known slots', () => {
  vi.spyOn(api, 'describeClass').mockResolvedValue(null as never)
  render(PipelineOptions, {
    pipeline: {
      configuration: { component_type: 'P' },
      safety_checker: {
        configuration: { component_type: 'X' },
        from_pretrained_arguments: { model_name: 'm' },
      },
      seed: 1,
    },
    index: 0,
    openSection: 'components',
    references: [],
  })
  expect(screen.getByText('safety_checker')).toBeInTheDocument()
})
