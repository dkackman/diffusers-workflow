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
import { adoptJob } from './jobWatch'
import { api } from './api'
import { notify } from './toast'
import { checkNow } from './statusPoll'
import { connection } from './connection.svelte'
import app from '../App.svelte?raw'

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
    await followJobs(hz(['j1']), { onItsPage })
    await followJobs(hz([]), { onItsPage })
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
    await followJobs(hz([]), { onItsPage: () => false })
    adoptJob('quick')
    await followJobs(hz([]), { onItsPage: () => false })
    await followJobs(hz([]), { onItsPage: () => false })
    expect(news).toHaveBeenCalledTimes(1)
  })
  it('a status poll reaching connection.health drives the announce', async () => {
    vi.spyOn(api, 'getJob').mockImplementation(async (id) => done(id))
    vi.spyOn(api, 'memory').mockResolvedValue({} as never)
    const health = vi.spyOn(api, 'health')
    // What App's effect does with each reading the poll lands
    const effect = () =>
      followJobs(connection.health, { onItsPage: () => false })
    health.mockResolvedValue(hz(['j2']))
    await checkNow()
    await effect()
    health.mockResolvedValue(hz([]))
    await checkNow()
    await effect()
    expect(news).toHaveBeenCalledTimes(1)
    expect(app).toMatch(/followJobs\(\s*info/)
    expect(app).toContain('connection.health')
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
