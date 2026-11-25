import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'

const success = vi.hoisted(() => vi.fn())
vi.mock('./toast', () => ({ notify: { success, error: vi.fn() } }))

import TokenPopover from './TokenPopover.svelte'
import { overlayOpen } from './ui/layers.svelte'
import { connection } from './connection.svelte'

// jsdom has no layout: Bits never positions the popover, so it stays
// visibility:hidden, and an accessible-name query cannot see a hidden
// element. These find it by its content attribute and read its label;
// the e2e suite (a real browser) covers that it shows and is named, and
// that a press outside closes it (Bits' dismiss layer does not see a
// synthetic jsdom press as its own).
const popover = () =>
  document.querySelector<HTMLElement>('[data-popover-content]')

afterEach(() => {
  cleanup()
  document.body.innerHTML = ''
})

function trigger() {
  const button = document.createElement('button')
  button.textContent = 'API token'
  document.body.appendChild(button)
  button.focus()
  return button
}

it('is a non-modal dialog with focus inside, holding a layer', async () => {
  render(TokenPopover, { props: { open: true, anchor: trigger() } })
  await waitFor(() => expect(popover()).not.toBeNull())
  const pop = popover()!
  expect(pop.getAttribute('role')).toBe('dialog')
  expect(pop.getAttribute('aria-label')).toBe('API token')
  expect(pop.getAttribute('aria-modal')).not.toBe('true')
  await waitFor(() => expect(pop.contains(document.activeElement)).toBe(true))
  expect(overlayOpen()).toBe(true)
})

it('closes on Escape', async () => {
  render(TokenPopover, { props: { open: true, anchor: trigger() } })
  await waitFor(() => expect(popover()).not.toBeNull())
  await fireEvent.keyDown(popover()!, { key: 'Escape' })
  await waitFor(() => expect(popover()).toBeNull())
})

it('leaves a press on its own trigger to the trigger', async () => {
  // The trigger toggles; were the popover to close itself on that press
  // too, the trigger's toggle would reopen it
  const button = trigger()
  render(TokenPopover, { props: { open: true, anchor: button } })
  await waitFor(() => expect(popover()).not.toBeNull())
  await fireEvent.pointerDown(button)
  await fireEvent.pointerUp(button)
  await new Promise((r) => setTimeout(r, 50))
  expect(popover()).not.toBeNull()
})

describe('saving a token', () => {
  function answer(status: number) {
    vi.stubGlobal('fetch', () =>
      Promise.resolve({
        ok: status < 400,
        status,
        statusText: 'x',
        headers: new Headers(),
        json: () => Promise.resolve({ detail: 'Invalid or missing API token' }),
      }),
    )
  }
  afterEach(() => {
    vi.unstubAllGlobals()
    success.mockClear()
    connection.state = 'unknown'
  })

  async function enter(token: string) {
    const input = popover()!.querySelector('input')!
    await fireEvent.input(input, { target: { value: token } })
    await fireEvent.click(screen.getByText('Save'))
  }

  it('an accepted token raises the epoch and closes', async () => {
    connection.state = 'locked'
    const before = connection.epoch
    answer(200)
    render(TokenPopover, { props: { open: true, anchor: trigger() } })
    await waitFor(() => expect(popover()).not.toBeNull())
    await enter('good')
    await waitFor(() => expect(popover()).toBeNull())
    expect(connection.state).toBe('ok')
    expect(connection.epoch).toBe(before + 1)
    expect(success).toHaveBeenCalledWith('Token accepted')
  })

  it('a refused token stays locked and says so', async () => {
    connection.state = 'locked'
    const before = connection.epoch
    answer(401)
    render(TokenPopover, { props: { open: true, anchor: trigger() } })
    await waitFor(() => expect(popover()).not.toBeNull())
    await enter('bad')
    await waitFor(() =>
      expect(popover()!.textContent).toContain('The server refused that token'),
    )
    expect(popover()!.textContent).not.toContain('Saved')
    expect(success).not.toHaveBeenCalled()
    expect(connection.state).toBe('locked')
    expect(connection.epoch).toBe(before)
  })

  it('Enter while a check is pending sends one check', async () => {
    let calls = 0
    let reply: (r: unknown) => void = () => {}
    vi.stubGlobal('fetch', () => {
      calls += 1
      return new Promise((r) => (reply = r))
    })
    render(TokenPopover, { props: { open: true, anchor: trigger() } })
    await waitFor(() => expect(popover()).not.toBeNull())
    const input = popover()!.querySelector('input')!
    await fireEvent.input(input, { target: { value: 'tok' } })
    await fireEvent.keyDown(input, { key: 'Enter' })
    await fireEvent.keyDown(input, { key: 'Enter' })
    expect(calls).toBe(1)
    reply({ ok: true, status: 200, headers: new Headers(), json: () => ({}) })
    await waitFor(() => expect(success).toHaveBeenCalledTimes(1))
  })

  it('a reopened popover forgets the last refusal', async () => {
    answer(401)
    const view = render(TokenPopover, {
      props: { open: true, anchor: trigger() },
    })
    await waitFor(() => expect(popover()).not.toBeNull())
    await enter('bad')
    await waitFor(() =>
      expect(popover()!.textContent).toContain('The server refused that token'),
    )
    await view.rerender({ open: false })
    await waitFor(() => expect(popover()).toBeNull())
    await view.rerender({ open: true })
    await waitFor(() => expect(popover()).not.toBeNull())
    expect(popover()!.textContent).not.toContain('The server refused')
  })

  it('an unreachable server says so inline', async () => {
    vi.stubGlobal('fetch', () => Promise.reject(new TypeError('Failed')))
    render(TokenPopover, { props: { open: true, anchor: trigger() } })
    await waitFor(() => expect(popover()).not.toBeNull())
    await enter('any')
    await waitFor(() =>
      expect(popover()!.textContent).toContain("Can't reach the server"),
    )
    expect(success).not.toHaveBeenCalled()
  })
})
