import type { JobEvent } from './types'

/** What a phase event means in the UI. The engine's vocabulary is small on
 * purpose - a phase says what the run is waiting on, not what any library
 * is doing internally. */
const PHASE_LABELS: Record<string, string> = {
  loading: 'loading',
  cached: 'models cached',
  generating: 'generating',
  decoding: 'decoding',
  saving: 'saving',
  task: 'running',
}

export interface StepProgress {
  /** The step's latest phase, or null before it reports one. */
  phase: string | null
  /** Ready-to-render text for that phase, detail included. */
  label: string | null
  /** The latest denoise counter, or null when the step has none yet. */
  denoise: { step: number; total_steps: number | null } | null
}

/** The progress of the step currently running.
 *
 * Everything here is scoped to the events since the last step_start:
 * a denoise counter belongs to the loop that emitted it, and carrying the
 * previous step's last one forward renders a full bar over a step that has
 * not started generating yet. */
export function stepProgress(events: JobEvent[]): StepProgress {
  let start = -1
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i].event === 'step_start') {
      start = i
      break
    }
  }

  let phase: JobEvent | null = null
  let denoise: JobEvent | null = null
  for (let i = events.length - 1; i > start; i--) {
    const event = events[i]
    if (!phase && event.event === 'phase') phase = event
    if (!denoise && event.event === 'pipeline_step') denoise = event
    if (phase && denoise) break
  }

  return {
    phase: (phase?.phase as string) ?? null,
    label: phase ? phaseLabel(phase) : null,
    denoise: denoise
      ? {
          step: denoise.step as number,
          total_steps: (denoise.total_steps as number | null) ?? null,
        }
      : null,
  }
}

/** "loading acme/model" - the detail is what makes a phase worth showing. */
export function phaseLabel(event: JobEvent): string {
  const phase = event.phase as string
  const label = PHASE_LABELS[phase] ?? phase
  const detail = event.detail as string | null | undefined
  return detail ? `${label} ${detail}` : label
}

/** Seconds left in the running denoise loop, at the pace of its last six
 * steps, or null until three steps have arrived or when it has no total.
 *
 * `stepTimes` are the arrival clocks (ms) of its pipeline_step events. */
export function estimateEta(
  stepTimes: number[],
  denoise: StepProgress['denoise'],
): number | null {
  if (!denoise?.total_steps || stepTimes.length < 3) return null
  const window = stepTimes.slice(-6)
  const perStep = (window[window.length - 1] - window[0]) / (window.length - 1)
  const remaining = denoise.total_steps - denoise.step
  if (remaining <= 0 || perStep <= 0) return null
  return Math.round((remaining * perStep) / 1000)
}

/** The arrival clocks after `event`: a denoise step is recorded at `now`,
 * and a new step, iteration or status starts them over - the gap since the
 * previous loop's last step includes a model load, and would inflate the
 * ETA. Any other event leaves them as they were. */
export function nextStepTimes(
  times: number[],
  event: JobEvent,
  now: number,
): number[] {
  if (event.event === 'pipeline_step') return [...times.slice(-6), now]
  if (
    event.event === 'step_start' ||
    event.event === 'iteration_start' ||
    event.event === 'job_status'
  )
    return []
  return times
}
