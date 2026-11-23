import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { notify, OFFLINE_ACTION_REMEDY, UNDO_MS } from './toast'
import { ApiError, NetworkError } from './apiErrors'
import { copyText } from './clipboard'

const sonner = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  dismiss: vi.fn(),
}))
vi.mock('svelte-sonner', () => ({ toast: sonner }))
vi.mock('./clipboard', () => ({ copyText: vi.fn() }))

beforeEach(() => vi.resetAllMocks())

it('an undo toast carries the way back for eight seconds', () => {
  const back = vi.fn()
  notify.undo('Dropped Shot 3', back, 'undo-1')
  expect(sonner.success).toHaveBeenCalledOnce()
  const [message, options] = sonner.success.mock.calls[0]
  expect(message).toBe('Dropped Shot 3')
  expect(options.duration).toBe(UNDO_MS)
  expect(options.id).toBe('undo-1')
  expect(UNDO_MS).toBe(8000)
  expect(options.action.label).toBe('Undo')
  options.action.onClick()
  expect(back).toHaveBeenCalledOnce()
})

it('an error stays until dismissed', () => {
  notify.error('no', 'id-1')
  expect(sonner.error).toHaveBeenCalledWith('no', {
    duration: Number.POSITIVE_INFINITY,
    id: 'id-1',
  })
})

describe('a failure toast', () => {
  const copy = vi.mocked(copyText)
  const press = () => {
    const [, options] = sonner.error.mock.calls.at(-1)!
    const event = { preventDefault: vi.fn() }
    options.action.onClick(event)
    return event
  }

  it('says what failed, why, and what to do, and stays', () => {
    notify.failure(
      'delete x',
      new ApiError('boom\nTraceback (most recent call last): x', 500),
    )
    const [message, options] = sonner.error.mock.calls[0]
    expect(message).toBe("Couldn't delete x: The server hit an error")
    expect(options.description).toBe(
      'Try again. The server log has the detail.',
    )
    expect(options.duration).toBe(Number.POSITIVE_INFINITY)
    expect(options.action.label).toBe('Copy details')
  })

  it('offers Copy details only when there is a detail', () => {
    notify.failure('delete x', new ApiError('', 500))
    expect(sonner.error.mock.calls[0][1].action).toBeUndefined()
  })

  it('an offline action is told to try again, not that it reloads', () => {
    notify.failure('delete x', new NetworkError('Failed to fetch'))
    const [message, options] = sonner.error.mock.calls[0]
    expect(message).toBe("Couldn't delete x: Can't reach the server")
    expect(options.description).toBe(OFFLINE_ACTION_REMEDY)
    expect(options.description).not.toContain('reloads')
  })

  it('Copy details keeps the toast and copies the detail', async () => {
    copy.mockResolvedValue(true)
    notify.failure(
      'delete x',
      new ApiError('boom\nTraceback (most recent call last): x', 500),
    )
    const event = press()
    expect(event.preventDefault).toHaveBeenCalled()
    expect(copy).toHaveBeenCalledWith(
      'boom\nTraceback (most recent call last): x',
    )
    await vi.waitFor(() =>
      expect(sonner.success).toHaveBeenCalledWith('Copied'),
    )
  })

  it('a copy that fails puts the detail in the same toast', async () => {
    copy.mockResolvedValue(false)
    sonner.error.mockReturnValue('t1')
    notify.failure(
      'delete x',
      new ApiError('boom\nTraceback (most recent call last): x', 500),
    )
    press()
    await vi.waitFor(() => expect(sonner.error).toHaveBeenCalledTimes(2))
    const [, options] = sonner.error.mock.calls[1]
    expect(options.id).toBe('t1')
    expect(options.description).toContain(
      'boom\nTraceback (most recent call last): x',
    )
    expect(sonner.success).not.toHaveBeenCalled()
  })
})

describe('copyText', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('falls back to the textarea where there is no clipboard API', async () => {
    const real =
      await vi.importActual<typeof import('./clipboard')>('./clipboard')
    vi.stubGlobal('navigator', { clipboard: undefined })
    const exec = vi.fn(() => true)
    document.execCommand = exec
    expect(await real.copyText('hello')).toBe(true)
    expect(exec).toHaveBeenCalledWith('copy')
  })

  it('falls back when the clipboard API refuses', async () => {
    const real =
      await vi.importActual<typeof import('./clipboard')>('./clipboard')
    const writeText = vi.fn(() => Promise.reject(new Error('denied')))
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    document.execCommand = vi.fn(() => false)
    expect(await real.copyText('hello')).toBe(false)
    expect(writeText).toHaveBeenCalledWith('hello')
  })
})
