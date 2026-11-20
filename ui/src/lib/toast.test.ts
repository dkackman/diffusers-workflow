import { beforeEach, expect, it, vi } from 'vitest'
import { notify, UNDO_MS } from './toast'

const sonner = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  dismiss: vi.fn(),
}))
vi.mock('svelte-sonner', () => ({ toast: sonner }))

beforeEach(() => vi.clearAllMocks())

it('an undo toast carries the way back for eight seconds', () => {
  const back = vi.fn()
  notify.undo('Dropped Shot 3', back)
  expect(sonner.success).toHaveBeenCalledOnce()
  const [message, options] = sonner.success.mock.calls[0]
  expect(message).toBe('Dropped Shot 3')
  expect(options.duration).toBe(UNDO_MS)
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
