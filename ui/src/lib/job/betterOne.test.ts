import { expect, it } from 'vitest'
import { betterOneArguments } from './betterOne'
import type { JobDetail } from '../types'

const job = (over: Partial<JobDetail> = {}) =>
  ({
    status: 'succeeded',
    workflow: 'text-to-image',
    workflow_name: 'templates/text-to-image',
    arguments: { prompt: 'a cat', width: 512, height: 512 },
    ...over,
  }) as JobDetail

it('carries the prompt and seed, and no size or tier', () => {
  expect(betterOneArguments(job(), { prompt: 'default' }, 42)).toEqual({
    prompt: 'a cat',
    seed: 42,
  })
})

it('a prompt the job did not set is the default it ran', () => {
  expect(
    betterOneArguments(job({ arguments: {} }), { prompt: 'a lighthouse' }, 7),
  ).toEqual({ prompt: 'a lighthouse', seed: 7 })
})

it('an unknown seed is left for the template to draw', () => {
  expect(betterOneArguments(job(), null, undefined)).toEqual({
    prompt: 'a cat',
  })
})

it('only a succeeded job of the first-run template', () => {
  for (const status of ['failed', 'running', 'queued', 'cancelled'] as const)
    expect(betterOneArguments(job({ status }), null, 1)).toBeNull()
  expect(
    betterOneArguments(job({ workflow_name: 'models/z-image' }), null, 1),
  ).toBeNull()
  // a job recorded with no catalog name falls back to its workflow
  expect(
    betterOneArguments(
      job({ workflow_name: null, workflow: 'templates/text-to-image' }),
      null,
      1,
    ),
  ).toEqual({ prompt: 'a cat', seed: 1 })
})
