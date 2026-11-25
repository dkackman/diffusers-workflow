import { afterEach, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { tick } from 'svelte'
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
  const select = screen.getByRole('combobox', { name: 'component to add' })
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
  const select = screen.getByRole('combobox', { name: 'component to add' })
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

it('keeps a fold the person opened when the class lookup answers', async () => {
  let answer: (d: PipelineDescription) => void = () => {}
  const describe = vi
    .spyOn(api, 'describeClass')
    .mockReturnValue(
      new Promise<PipelineDescription>((resolve) => (answer = resolve)),
    )
  render(PipelineOptions, {
    pipeline: { configuration: { component_type: 'SlowProbePipeline' } },
    index: 0,
    openSection: '',
    references: [],
  })
  await vi.waitFor(() => expect(describe).toHaveBeenCalled())
  expect(screen.getByText('reading the pipeline…')).toBeInTheDocument()
  const fold = screen
    .getByText('LoRAs')
    .closest('details') as HTMLDetailsElement
  fold.open = true
  answer(sdDescription)
  expect(await screen.findByText('add component…')).toBeInTheDocument()
  expect(fold.open).toBe(true)
})

const cacheType = () => screen.getByLabelText('cache type') as HTMLSelectElement

it('shows a cache type the list does not hold', () => {
  render(PipelineOptionsHarness, {
    initial: { configuration: { cache: { type: 'brand_new_cache' } } },
  })
  expect(cacheType().value).toBe('brand_new_cache')
})

it('writes no cache type into a cache that has none', async () => {
  const cache: Record<string, unknown> = { threshold: 0.1 }
  const initial = { configuration: { cache } }
  render(PipelineOptions, {
    pipeline: initial,
    index: 0,
    openSection: 'acceleration',
    references: [],
  })
  await tick()
  expect(cacheType().value).toBe('')
  expect(cache).toEqual({ threshold: 0.1 })
})
