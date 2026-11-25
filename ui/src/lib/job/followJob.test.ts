// The job page's record and stream across a drop (JB-1), driven without a
// page: the record is the real resource, the server and its stream are not.
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { api } from '../test/apiMock'
import { ApiError } from '../apiErrors'
import { connection } from '../connection.svelte'
import { inEffectRoot } from '../test/effectRoot.svelte'
import type { JobEvent } from '../types'
import { followJob, jobRecord } from './followJob.svelte'

type Stream = {
  id: string
  from: number
  event: (e: JobEvent) => void
  end: () => void
  drop: () => void
  stop: ReturnType<typeof vi.fn>
}
const streams = vi.hoisted(() => [] as unknown[]) as Stream[]
vi.mock('../api', async () =>
  (await import('../test/apiMock')).apiModule({
    streamJobEvents: (
      id: string,
      from: number,
      event: Stream['event'],
      end: Stream['end'],
      drop?: Stream['drop'],
    ) => {
      const stop = vi.fn()
      streams.push({ id, from, event, end, drop: drop!, stop })
      return stop
    },
  }),
)

const job = (over: object = {}) => ({
  id: 'j1',
  status: 'running',
  historical: false,
  ...over,
})
const settle = () => vi.advanceTimersByTimeAsync(10)

let stop: () => void
async function follow() {
  const onEvent = vi.fn()
  const onReset = vi.fn()
  const root = inEffectRoot(() => {
    const record = jobRecord(() => 'j1')
    followJob(() => 'j1', record, onEvent, onReset)
    return record
  })
  stop = root.stop
  await settle()
  root.flush()
  return { record: root.made, onEvent, onReset }
}

beforeEach(() => {
  vi.useFakeTimers()
  streams.length = 0
  connection.state = 'ok'
  api.getJob.mockReset().mockResolvedValue(job())
})
afterEach(() => {
  stop()
  vi.useRealTimers()
})

it('streams from the first event once the record answers', async () => {
  const { onEvent, onReset } = await follow()
  expect(onReset).toHaveBeenCalledTimes(1)
  expect(streams).toHaveLength(1)
  expect(streams[0]).toMatchObject({ id: 'j1', from: -1 })
  streams[0].event({ seq: 3 } as JobEvent)
  expect(onEvent).toHaveBeenCalledWith({ seq: 3 })
  stop()
  expect(streams[0].stop).toHaveBeenCalled()
  stop = () => {}
})

it('opens no stream for a historical job', async () => {
  api.getJob.mockResolvedValue(job({ historical: true, status: 'succeeded' }))
  await follow()
  expect(streams).toHaveLength(0)
})

it('an outage that lifts reopens from the last seq delivered', async () => {
  await follow()
  streams[0].event({ seq: 7 } as JobEvent)
  connection.epoch++
  await settle()
  expect(streams).toHaveLength(2)
  expect(streams[1].from).toBe(7)
})

it('a stream that ended is not reopened', async () => {
  await follow()
  streams[0].end()
  await settle()
  connection.epoch++
  await settle()
  expect(streams).toHaveLength(1)
})

it('a drop on a job since finished reads it again and stops for good', async () => {
  const { record } = await follow()
  api.getJob.mockResolvedValue(job({ status: 'succeeded' }))
  streams[0].drop()
  await settle()
  expect(record.data?.status).toBe('succeeded')
  connection.epoch++
  await settle()
  expect(streams).toHaveLength(1)
})

it('a drop while offline reads nothing', async () => {
  await follow()
  connection.state = 'offline'
  api.getJob.mockClear()
  streams[0].drop()
  await settle()
  expect(api.getJob).not.toHaveBeenCalled()
})

it('a 404 drops the record it held', async () => {
  const { record } = await follow()
  expect(record.data?.id).toBe('j1')
  api.getJob.mockRejectedValue(new ApiError('no such job', 404))
  await record.reload()
  expect(record.data).toBeUndefined()
  expect(record.status).toBe('error')
})
