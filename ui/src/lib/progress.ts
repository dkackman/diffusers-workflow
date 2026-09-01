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
