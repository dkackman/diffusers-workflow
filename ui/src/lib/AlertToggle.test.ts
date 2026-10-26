import { afterEach, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, waitFor } from '@testing-library/svelte'

const enableAlerts = vi.fn<() => Promise<boolean>>()
const error = vi.fn()
vi.mock('./finishAlert', () => ({
  alertMode: () => 'chime',
  enableAlerts: () => enableAlerts(),
}))
vi.mock('./toast', () => ({ notify: { error: (m: string) => error(m) } }))

import AlertToggle from './AlertToggle.svelte'
import { alerts } from './alerts.svelte'
import { storageGet } from './storage'

afterEach(() => {
  cleanup()
  alerts.on = false
  enableAlerts.mockReset()
  error.mockReset()
  localStorage.clear()
})

const button = () => document.querySelector('button')!

it('turns alerts on once the browser allows them, and remembers it', async () => {
  enableAlerts.mockResolvedValue(true)
  render(AlertToggle)
  expect(button().getAttribute('aria-pressed')).toBe('false')
  await fireEvent.click(button())
  await waitFor(() => expect(alerts.on).toBe(true))
  expect(button().getAttribute('aria-pressed')).toBe('true')
  expect(button().getAttribute('aria-label')).toBe('stop job-finished alerts')
  expect(storageGet('alert-finished', null)).toBe(true)
})

it('stays off and says why when the browser refuses', async () => {
  enableAlerts.mockResolvedValue(false)
  render(AlertToggle)
  await fireEvent.click(button())
  await waitFor(() => expect(error).toHaveBeenCalledOnce())
  expect(alerts.on).toBe(false)
  expect(button().getAttribute('title')).toMatch(/^chime when a job finishes/)
})

it('turns off without asking', async () => {
  alerts.on = true
  render(AlertToggle)
  await fireEvent.click(button())
  await waitFor(() => expect(alerts.on).toBe(false))
  expect(enableAlerts).not.toHaveBeenCalled()
  expect(storageGet('alert-finished', null)).toBe(false)
})
