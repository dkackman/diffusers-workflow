// The job page's record and its live event stream, kept true across an
// outage (JB-1). The record is a resource watching `jobs` and `gallery` (a
// run note and the run's outputs move `gallery`, not `jobs`); a 404 drops
// what it held, so the page says the job is gone rather than showing it
// running forever. The stream opens once the record answers (a historical
// job has no event log to stream); when it drops, the record is read again,
// and a job that has ended or is gone stops it. EventSource repairs a
// dropped connection itself, resuming by seq, but gives up on an HTTP error,
// so an epoch rise reopens the stream of a job still running from the last
// seq it delivered. Call both during component init.
import { untrack } from 'svelte'
import { TERMINAL_STATUSES, api, streamJobEvents } from '../api'
import { ApiError } from '../apiErrors'
import { connection } from '../connection.svelte'
import { resource, type Resource } from '../resource.svelte'
import type { JobDetail, JobEvent } from '../types'

const isGone = (e: unknown) => e instanceof ApiError && e.status === 404

/** The record of job `jobId()`, held only while it is that job's and the
 * server has not said it is gone */
export function jobRecord(jobId: () => string): Resource<JobDetail> {
  // Gone is per job: latched by a 404 for that id and lifted only when a
  // record for it lands, so neither the next job's read nor a Retry of
  // this one is judged by the last job's answer
  let goneId = $state<string | undefined>(undefined)
  const read = resource(
    async () => {
      const id = jobId()
      try {
        const job = await api.getJob(id)
        if (goneId === id) goneId = undefined
        return job
      } catch (e) {
        if (isGone(e)) goneId = id
        throw e
      }
    },
    { watch: ['jobs', 'gallery'] },
  )
  const gone = $derived(goneId === jobId())
  const held = $derived(
    !gone && read.data?.id === jobId() ? read.data : undefined,
  )
  return {
    get status() {
      return gone ? 'error' : read.status
    },
    get data() {
      return held
    },
    get error() {
      return read.error
    },
    get stale() {
      return read.stale
    },
    reload: read.reload,
  }
}

const ended = (job: JobDetail | undefined) =>
  !job || job.historical === true || TERMINAL_STATUSES.includes(job.status)

/** One job's stream at a time: `open` resumes from the last seq it
 * delivered, `close` ends it for good, `reset` forgets the last job */
function jobStream(
  record: Resource<JobDetail>,
  onEvent: (event: JobEvent) => void,
) {
  let stop: (() => void) | null = null
  let lastSeq = -1
  let generation = 0
  const state = { opened: false, done: false }

  function halt() {
    stop?.()
    stop = null
  }
  function close() {
    halt()
    state.done = true
  }
  async function dropped(asked: number) {
    // EventSource retries every few seconds through an outage; the first
    // drop's read marked the connection offline, and the epoch rise on
    // recovery reads the record again and reopens the stream
    if (connection.state === 'offline') return
    await record.reload()
    if (asked === generation && ended(record.data)) close()
  }
  function open(id: string) {
    halt()
    const asked = generation
    stop = streamJobEvents(
      id,
      lastSeq,
      (event) => {
        lastSeq = Math.max(lastSeq, event.seq)
        onEvent(event)
      },
      () => {
        stop = null
        state.done = true
        void record.reload()
      },
      () => void dropped(asked),
    )
  }
  function reset() {
    generation++
    halt()
    lastSeq = -1
    state.opened = false
    state.done = false
  }
  return { state, open, reset }
}

/** Stream job `jobId()`'s events into `onEvent`, from the first, once
 * `record` answers; `onReset` clears what the last job streamed */
export function followJob(
  jobId: () => string,
  record: Resource<JobDetail>,
  onEvent: (event: JobEvent) => void,
  onReset: () => void,
): void {
  const stream = jobStream(record, onEvent)
  const { state } = stream
  $effect(() => {
    jobId()
    untrack(onReset)
    return stream.reset
  })
  $effect(() => {
    const job = record.data
    if (!job || state.opened) return
    state.opened = true
    if (!job.historical) untrack(() => stream.open(job.id))
  })
  let seen = untrack(() => connection.epoch)
  $effect(() => {
    const epoch = connection.epoch
    if (epoch === seen) return
    seen = epoch
    untrack(() => {
      const job = record.data
      if (state.opened && !state.done && !ended(job)) stream.open(job!.id)
    })
  })
}
