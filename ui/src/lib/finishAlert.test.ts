import {
  type MockInstance,
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from 'vitest'
import type { HealthInfo, JobDetail } from './types'
import { alertMode, followJobs, resetFollow } from './finishAlert'
import { adoptJob, adoptedIds } from './jobWatch'
import { ApiError } from './apiErrors'
import { api } from './api'
import { notify } from './toast'
import { checkNow } from './statusPoll'
import { connection } from './connection.svelte'

let news: MockInstance<typeof notify.news>
const goTo = vi.hoisted(() => vi.fn())
vi.mock('./router.svelte', () => ({ go: goTo }))
const hz = (ids: string[]) =>
  ({
    current_job: ids[0] ?? null,
    workers: ids.map((id) => ({ current_job: id })),
  }) as unknown as HealthInfo
const done = (id: string) =>
  ({
    id,
    workspace: 'w1',
    workflow: 'a/b.json',
    status: 'succeeded',
  }) as unknown as JobDetail

describe('followJobs', () => {
  beforeEach(() => {
    news = vi.spyOn(notify, 'news').mockImplementation(() => {})
  })
  afterEach(() => {
    news.mockClear()
    goTo.mockClear()
    vi.restoreAllMocks()
    resetFollow()
  })
  async function finish(onItsPage: (id: string) => boolean) {
    vi.spyOn(api, 'getJob').mockImplementation(async (id) => done(id))
    await followJobs(hz(['j1']), { onItsPage, alertsOn: () => false })
    await followJobs(hz([]), { onItsPage, alertsOn: () => false })
  }
  it('announces with an Open action that goes to the job', async () => {
    await finish(() => false)
    expect(news).toHaveBeenCalledTimes(1)
    news.mock.calls[0][1].onClick()
    expect(goTo).toHaveBeenCalledWith('ws', 'w1', 'jobs', 'j1')
    expect(news.mock.calls[0][1].label).toBe('Open')
  })
  it("does not announce on the job's own page", async () => {
    await finish((id) => id === 'j1')
    expect(news).not.toHaveBeenCalled()
  })
  it('announces an adopted job that ended between polls', async () => {
    vi.spyOn(api, 'getJob').mockImplementation(async (id) => done(id))
    await followJobs(hz([]), { onItsPage: () => false, alertsOn: () => false })
    adoptJob('quick')
    await followJobs(hz([]), { onItsPage: () => false, alertsOn: () => false })
    await followJobs(hz([]), { onItsPage: () => false, alertsOn: () => false })
    expect(news).toHaveBeenCalledTimes(1)
  })
  it('a status poll reaching connection.health drives the announce', async () => {
    vi.spyOn(api, 'getJob').mockImplementation(async (id) => done(id))
    vi.spyOn(api, 'memory').mockResolvedValue({} as never)
    const health = vi.spyOn(api, 'health')
    const ctx = { onItsPage: () => false, alertsOn: () => false }
    // What App's effect does with each reading the poll lands
    const effect = () => followJobs(connection.health, ctx)
    health.mockResolvedValue(hz(['j2']))
    await checkNow()
    await effect()
    health.mockResolvedValue(hz([]))
    await checkNow()
    await effect()
    expect(news).toHaveBeenCalledTimes(1)
  })
})

describe('adopted jobs end once and stay bounded', () => {
  const ctx = { onItsPage: () => false, alertsOn: () => false }
  const queuedHealth = (ids: string[], queued: number) =>
    ({ ...hz(ids), queued }) as HealthInfo
  beforeEach(() => {
    news = vi.spyOn(notify, 'news').mockImplementation(() => {})
  })
  afterEach(() => {
    vi.restoreAllMocks()
    resetFollow()
  })
  it('adopted, queued, then running, announces exactly once', async () => {
    const get = vi
      .spyOn(api, 'getJob')
      .mockImplementation(async (id) => done(id))
    await followJobs(hz([]), ctx)
    adoptJob('q')
    await followJobs(queuedHealth([], 1), ctx)
    await followJobs(queuedHealth(['q'], 0), ctx)
    await followJobs(hz([]), ctx)
    await followJobs(hz([]), ctx)
    expect(news).toHaveBeenCalledTimes(1)
    expect(get.mock.calls.filter(([id]) => id === 'q')).toHaveLength(2)
  })
  it('a queued adopted job is not re-read on every poll', async () => {
    const get = vi
      .spyOn(api, 'getJob')
      .mockImplementation(async (id) => done(id))
    await followJobs(hz([]), ctx)
    adoptJob('q')
    for (let i = 0; i < 4; i++) await followJobs(queuedHealth([], 2), ctx)
    expect(get).not.toHaveBeenCalled()
    expect(adoptedIds().has('q')).toBe(true)
  })
  it('a 404 drops the id with no further reads', async () => {
    const get = vi
      .spyOn(api, 'getJob')
      .mockRejectedValue(new ApiError('gone', 404, 'not_found'))
    await followJobs(hz([]), ctx)
    adoptJob('gone')
    for (let i = 0; i < 3; i++) await followJobs(hz([]), ctx)
    expect(get).toHaveBeenCalledTimes(1)
    expect(news).not.toHaveBeenCalled()
  })
  it('a transient error keeps the id for the next poll', async () => {
    const get = vi.spyOn(api, 'getJob').mockRejectedValueOnce(new Error('net'))
    get.mockImplementation(async (id) => done(id))
    await followJobs(hz([]), ctx)
    adoptJob('t')
    await followJobs(hz([]), ctx)
    await followJobs(hz([]), ctx)
    expect(news).toHaveBeenCalledTimes(1)
  })
  it('two overlapping readings announce once', async () => {
    vi.spyOn(api, 'getJob').mockImplementation(async (id) => {
      await new Promise((r) => setTimeout(r, 5))
      return done(id)
    })
    await followJobs(hz([]), ctx)
    adoptJob('o')
    await Promise.all([followJobs(hz([]), ctx), followJobs(hz([]), ctx)])
    expect(news).toHaveBeenCalledTimes(1)
  })
  it('a reading that overlaps a slow title read announces once', async () => {
    vi.spyOn(api, 'getJob').mockImplementation(async (id) => {
      if (id === 'r') await new Promise((r) => setTimeout(r, 20))
      return done(id)
    })
    await followJobs(hz(['r']), ctx)
    adoptJob('o')
    const first = followJobs(hz(['r']), ctx)
    const second = followJobs(hz(['r']), ctx)
    await Promise.all([first, second])
    expect(news).toHaveBeenCalledTimes(1)
  })
  it('a held adopted job that finished is announced on the sixth reading', async () => {
    const get = vi
      .spyOn(api, 'getJob')
      .mockImplementation(async (id) => done(id))
    await followJobs(hz([]), ctx)
    adoptJob('q')
    for (let i = 0; i < 4; i++) await followJobs(queuedHealth([], 1), ctx)
    expect(get).not.toHaveBeenCalled()
    await followJobs(queuedHealth([], 1), ctx)
    expect(news).toHaveBeenCalledTimes(1)
    expect(adoptedIds().has('q')).toBe(false)
  })
  it('a recheck that finds the job still queued keeps it adopted', async () => {
    vi.spyOn(api, 'getJob').mockImplementation(
      async (id) => ({ ...done(id), status: 'queued' }) as JobDetail,
    )
    await followJobs(hz([]), ctx)
    adoptJob('q')
    for (let i = 0; i < 5; i++) await followJobs(queuedHealth([], 1), ctx)
    expect(news).not.toHaveBeenCalled()
    expect(adoptedIds().has('q')).toBe(true)
  })
  it('the adopted set is bounded, oldest dropped first', () => {
    for (let i = 0; i < 60; i++) adoptJob(`j${i}`)
    expect(adoptedIds().size).toBe(50)
    expect(adoptedIds().has('j0')).toBe(false)
    expect(adoptedIds().has('j59')).toBe(true)
  })
})

describe('alertMode', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('uses desktop notifications on a secure origin', () => {
    vi.stubGlobal('isSecureContext', true)
    vi.stubGlobal('Notification', function () {})
    expect(alertMode()).toBe('notification')
  })
  it('falls back to a chime over plain http from another machine', () => {
    vi.stubGlobal('isSecureContext', false)
    vi.stubGlobal('Notification', function () {})
    expect(alertMode()).toBe('chime')
  })
  it('falls back to a chime where the API does not exist', () => {
    vi.stubGlobal('isSecureContext', true)
    vi.stubGlobal('Notification', undefined)
    expect(alertMode()).toBe('chime')
  })
})
