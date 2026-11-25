import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it } from 'vitest'
import JobProgress from './JobProgress.svelte'

afterEach(cleanup)

function show(etaSeconds: number | null) {
  render(JobProgress, {
    steps: ['gen'],
    finishedSteps: [],
    listStep: 'gen',
    running: true,
    progress: {
      phase: 'generating',
      label: null,
      denoise: { step: 2, total_steps: 10 },
    },
    etaSeconds,
  })
}

it('reads the ETA as about <duration> left', () => {
  show(180)
  expect(screen.getByText(/about 3 min left/)).toBeTruthy()
})

it('the estimate names the step', () => {
  show(180)
  expect(screen.getByText(/about 3 min left in this step/)).toBeTruthy()
})

it('says under 1 min for a short ETA and nothing without one', () => {
  show(20)
  expect(screen.getByText(/under 1 min left in this step/)).toBeTruthy()
})
