import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { LinkedRequest } from './linkedRequest.svelte'
import { entry, item, request } from './fixtures'
import type { LedgerNote, LedgerRequest } from '../types'

const api = vi.hoisted(() => ({ getRequest: vi.fn() }))
vi.mock('../api', async () => {
  const { ApiError } = await import('../apiErrors')
  return { ApiError, api }
})
const notify = vi.hoisted(() => ({ news: vi.fn() }))
vi.mock('../toast', () => ({ notify }))

beforeEach(() => {
  api.getRequest.mockReset()
  notify.news.mockReset()
  location.hash = '#/ws/studio/jobs/j1'
})
afterEach(() => (document.body.innerHTML = ''))

const at = (minute: number) =>
  `2026-01-01T00:${String(minute).padStart(2, '0')}:00+00:00`
const note: LedgerNote = { id: 'n1', text: 'warmer', created: at(5), at: at(5) }
const waiting = (revision = 3): LedgerRequest =>
  request([item('s3', { notes: [note] })], { revision })
const answered = (): LedgerRequest =>
  request([item('s3', { notes: [note] })], {
    revision: 4,
    agent_wrote_at: at(8),
    history: [entry(4, 'agent', 'warmer grade on take 2')],
  })

it('reads the request, and says nothing on the first read', async () => {
  api.getRequest.mockResolvedValue(answered())
  const linked = new LinkedRequest('studio', 'r1')
  await linked.read()
  expect(api.getRequest).toHaveBeenCalledWith('studio', 'r1')
  expect(linked.request?.revision).toBe(4)
  expect(notify.news).not.toHaveBeenCalled()
})

it('tells once when a waiting note is answered between two reads', async () => {
  const linked = new LinkedRequest('studio', 'r1')
  api.getRequest.mockResolvedValueOnce(waiting())
  await linked.read()
  api.getRequest.mockResolvedValue(answered())
  await linked.read()
  await linked.read()
  expect(notify.news).toHaveBeenCalledTimes(1)
  expect(notify.news.mock.calls[0][0]).toBe(
    'The agent updated Launch posters after your note: “warmer grade on take 2”',
  )
})

it('goes to the note when this page shows it, else to the request page', async () => {
  const linked = new LinkedRequest('studio', 'r1')
  api.getRequest.mockResolvedValueOnce(waiting())
  await linked.read()
  api.getRequest.mockResolvedValue(answered())
  await linked.read()
  const { onClick } = notify.news.mock.calls[0][1]
  onClick()
  expect(location.hash).toBe('#/ws/studio/requests/r1')
  location.hash = '#/ws/studio/jobs/j1'
  document.body.innerHTML = '<li id="request-note-n1" tabindex="-1"></li>'
  onClick()
  expect(document.activeElement?.id).toBe('request-note-n1')
  expect(location.hash).toBe('#/ws/studio/jobs/j1')
})

it('is gone on a 404 and keeps what it holds on any other failure', async () => {
  const { ApiError } = await import('../apiErrors')
  const linked = new LinkedRequest('studio', 'r1')
  api.getRequest.mockResolvedValueOnce(waiting())
  await linked.read()
  api.getRequest.mockRejectedValueOnce(new ApiError('boom', 500))
  await linked.read()
  expect(linked.gone).toBe(false)
  expect(linked.request?.revision).toBe(3)
  api.getRequest.mockRejectedValueOnce(new ApiError('gone', 404))
  await linked.read()
  expect(linked.gone).toBe(true)
})

it('drops a reply overtaken by a later read', async () => {
  const linked = new LinkedRequest('studio', 'r1')
  let answer: (r: LedgerRequest) => void = () => {}
  api.getRequest.mockReturnValueOnce(new Promise((r) => (answer = r)))
  const first = linked.read()
  api.getRequest.mockResolvedValueOnce(answered())
  await linked.read()
  answer(waiting(5))
  await first
  expect(linked.request?.revision).toBe(4)
})
