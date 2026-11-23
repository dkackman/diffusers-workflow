import { afterEach, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/svelte'
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
