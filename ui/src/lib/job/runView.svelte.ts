// What the job page reads off a run's event stream, taken in one event at a
// time: the stream grows without bound (a denoise step each), so nothing
// here keeps all of it or reads it again.
import { formatDuration } from '../format'
import { nextStepTimes, progressWindow } from '../progress'
import type { JobEvent } from '../types'

/** The events the runstate.ts and results.ts readings are made from: where
 * steps start and end, and a chained step's segments */
const isMark = (event: JobEvent) =>
  event.event === 'step_start' ||
  event.event === 'step_end' ||
  (event.event === 'phase' && event.phase === 'generating')

// A line led by its clock (seconds since the job started)
const logLine = (event: JobEvent) =>
  typeof event.at === 'number'
    ? `${('+' + formatDuration(event.at)).padStart(8)}  ${event.message}`
    : (event.message as string)

/** One job's run so far. A new job, or a stream read again from the start,
 * takes a new one. */
export class RunView {
  /** How many events have arrived */
  count = $state(0)
  /** The latest event, which is all a stall is read from */
  last = $state.raw<JobEvent | null>(null)
  marks = $state.raw<JobEvent[]>([])
  /** What the current step's progress is read from (progressWindow) */
  window = $state.raw<JobEvent[]>([])
  stepStart = $state.raw<JobEvent | null>(null)
  /** The first workflow_start and run_start */
  workflowStart = $state.raw<JobEvent | null>(null)
  runStart = $state.raw<JobEvent | null>(null)
  logs = $state<string[]>([])
  /** Arrival clocks for pipeline_step events, for the ETA estimate */
  stepTimes = $state.raw<number[]>([])
  cancelRequested = $state(false)
  /** The output kinds step_end events have named, by file */
  kinds = $state.raw<Record<string, string | null>>({})

  apply(event: JobEvent, now: number) {
    this.count += 1
    this.last = event
    if (isMark(event)) this.marks = [...this.marks, event]
    this.window = progressWindow(this.window, event)
    this.stepTimes = nextStepTimes(this.stepTimes, event, now)
    if (event.output_kinds)
      this.kinds = {
        ...this.kinds,
        ...(event.output_kinds as Record<string, string | null>),
      }
    this.note(event)
  }

  // The events that matter for themselves
  private note(event: JobEvent) {
    if (event.event === 'step_start') this.stepStart = event
    else if (event.event === 'workflow_start') this.workflowStart ??= event
    else if (event.event === 'run_start') this.runStart ??= event
    else if (event.event === 'log') this.logs.push(logLine(event))
    else if (event.event === 'cancel_pending') this.cancelRequested = true
  }
}
