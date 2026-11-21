import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { LISTENING_POLL_MS, NotesChannel, listening } from './listening.svelte'

const notesChannel = vi.hoisted(() => vi.fn())
vi.mock('../api', () => ({ api: { notesChannel } }))

let hidden = false
beforeEach(() => {
  vi.useFakeTimers()
  hidden = false
  vi.spyOn(document, 'visibilityState', 'get').mockImplementation(() =>
    hidden ? 'hidden' : 'visible',
  )
  notesChannel.mockReset()
  notesChannel.mockResolvedValue({ listening: true, last_seen: 'now' })
})
afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

const settle = () => vi.advanceTimersByTimeAsync(0)

it('is one shared answer per workspace', () => {
  expect(listening('studio')).toBe(listening('studio'))
  expect(listening('studio')).not.toBe(listening('other'))
})

it('asks while held, every poll, and stops when the last holder lets go', async () => {
  const channel = new NotesChannel('studio')
  const first = channel.hold()
  const second = channel.hold()
  await settle()
  expect(channel.listening).toBe(true)
  expect(notesChannel).toHaveBeenCalledTimes(1)
  expect(notesChannel).toHaveBeenCalledWith('studio')
  await vi.advanceTimersByTimeAsync(LISTENING_POLL_MS)
  expect(notesChannel).toHaveBeenCalledTimes(2)
  first()
  first() // a second release of the same hold counts once
  await vi.advanceTimersByTimeAsync(LISTENING_POLL_MS)
  expect(notesChannel).toHaveBeenCalledTimes(3)
  second()
  await vi.advanceTimersByTimeAsync(LISTENING_POLL_MS * 3)
  expect(notesChannel).toHaveBeenCalledTimes(3)
})

it('pauses while the tab is hidden and asks again when it is shown', async () => {
  const channel = new NotesChannel('studio')
  const release = channel.hold()
  await settle()
  hidden = true
  document.dispatchEvent(new Event('visibilitychange'))
  await vi.advanceTimersByTimeAsync(LISTENING_POLL_MS * 3)
  expect(notesChannel).toHaveBeenCalledTimes(1)
  hidden = false
  document.dispatchEvent(new Event('visibilitychange'))
  await settle()
  expect(notesChannel).toHaveBeenCalledTimes(2)
  await vi.advanceTimersByTimeAsync(LISTENING_POLL_MS)
  expect(notesChannel).toHaveBeenCalledTimes(3)
  release()
})

it('a failed read counts as no session listening', async () => {
  const channel = new NotesChannel('studio')
  expect(await channel.refresh()).toBe(true)
  notesChannel.mockRejectedValue(new TypeError('Failed to fetch'))
  expect(await channel.refresh()).toBe(false)
  expect(channel.listening).toBe(false)
})
