import { expect, it } from 'vitest'
import { activeStall, stepProgress } from '../progress'
import { activeMember, chainSegments, nodeDurations } from '../runstate'
import type { JobEvent } from '../types'
import { RunView } from './runView.svelte'

const stream = [
  { event: 'workflow_start', steps: ['a@one', 'b'], seed: 7, at: 0 },
  { event: 'run_start', version: 3, at: 0 },
  { event: 'step_start', step: 'a@one', at: 1 },
  { event: 'phase', phase: 'loading', detail: 'acme/model', at: 1 },
  { event: 'download_progress', repo_id: 'acme/model', at: 2 },
  { event: 'download_progress', repo_id: 'acme/model', at: 3 },
  { event: 'log', message: 'hello', at: 3 },
  { event: 'phase', phase: 'generating', detail: 'segment 1/2', at: 4 },
  { event: 'pipeline_step', step: 1, total_steps: 4, at: 5 },
  { event: 'pipeline_step', step: 2, total_steps: 4, at: 6 },
  { event: 'download_progress', repo_id: 'acme/vae', at: 7 },
  { event: 'warning', kind: 'phase_stall', at: 8 },
  { event: 'phase', phase: 'decoding', at: 9 },
  {
    event: 'step_end',
    step: 'a@one',
    output_kinds: { 'a.png': 'image' },
    at: 10,
  },
  { event: 'cancel_pending', at: 10 },
  { event: 'step_start', step: 'b', at: 11 },
  { event: 'pipeline_step', step: 1, total_steps: null, at: 12 },
] as unknown as JobEvent[]

it('reads the same, one event at a time, as the whole stream would', () => {
  const run = new RunView()
  stream.forEach((event, i) => {
    run.apply(event, i)
    const all = stream.slice(0, i + 1)
    expect(stepProgress(run.window)).toEqual(stepProgress(all))
    expect(activeStall([run.last!])).toEqual(activeStall(all))
    expect(chainSegments(run.marks)).toEqual(chainSegments(all))
    expect(nodeDurations(run.marks)).toEqual(nodeDurations(all))
    expect(activeMember(run.marks)).toEqual(activeMember(all))
  })
  expect(run.count).toBe(stream.length)
  expect(run.workflowStart?.seed).toBe(7)
  expect(run.runStart?.version).toBe(3)
  expect(run.stepStart?.step).toBe('b')
  expect(run.logs).toHaveLength(1)
  expect(run.logs[0]).toMatch(/\+.*hello$/)
  expect(run.cancelRequested).toBe(true)
  expect(run.kinds).toEqual({ 'a.png': 'image' })
})

it('holds no more for a long denoise loop', () => {
  const run = new RunView()
  run.apply({ event: 'step_start', step: 'a' } as unknown as JobEvent, 0)
  for (let step = 1; step <= 500; step++)
    run.apply({ event: 'pipeline_step', step } as unknown as JobEvent, step)
  expect(run.marks).toHaveLength(1)
  expect(run.window).toHaveLength(2)
  expect(run.stepTimes.length).toBeLessThanOrEqual(7)
})
