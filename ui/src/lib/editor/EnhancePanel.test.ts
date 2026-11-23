import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import EnhancePanel from './EnhancePanel.svelte'
import { Enhancer } from '../enhancer.svelte'

vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))
afterEach(cleanup)

it('a failed enhancement leads with plain words, the engine text folded behind Details', () => {
  const enhancer = new Enhancer()
  const raw = 'Traceback (most recent call last): KeyError: preset'
  enhancer.error = "Couldn't start the enhancement: The server hit an error"
  enhancer.errorDetail = raw
  render(EnhancePanel, { props: { enhancer, onuse: vi.fn() } })
  const alert = screen.getByRole('alert')
  expect(alert.querySelector('p')?.textContent).toBe(
    "Couldn't start the enhancement: The server hit an error",
  )
  expect(alert.querySelector('details')?.open).toBe(false)
  expect(alert.querySelector('details pre')?.textContent).toBe(raw)
})
