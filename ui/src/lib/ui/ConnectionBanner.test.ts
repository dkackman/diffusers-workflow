import { afterEach, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { tick } from 'svelte'

const checkNow = vi.hoisted(() => vi.fn(() => Promise.resolve()))
vi.mock('../statusPoll', () => ({ checkNow }))

import ConnectionBanner from './ConnectionBanner.svelte'
import { connection } from '../connection.svelte'

afterEach(() => {
  cleanup()
  connection.state = 'unknown'
})

it('offline shows Retry now and locked shows Enter token', async () => {
  const ontoken = vi.fn()
  connection.state = 'ok'
  const { container } = render(ConnectionBanner, { props: { ontoken } })
  expect(container.textContent?.trim()).toBe('')

  connection.state = 'offline'
  await tick()
  expect(
    screen.getByText("Can't reach the server. Retrying every few seconds."),
  ).toBeTruthy()
  await fireEvent.click(screen.getByRole('button', { name: 'Retry now' }))
  expect(checkNow).toHaveBeenCalledTimes(1)
  expect(screen.queryByRole('button', { name: 'Enter token' })).toBeNull()

  connection.state = 'locked'
  await tick()
  expect(screen.getByText('This server needs an API token.')).toBeTruthy()
  await fireEvent.click(screen.getByRole('button', { name: 'Enter token' }))
  expect(ontoken).toHaveBeenCalledTimes(1)
  expect(screen.queryByRole('button', { name: 'Retry now' })).toBeNull()

  connection.state = 'unknown'
  await tick()
  expect(container.textContent?.trim()).toBe('')
})
