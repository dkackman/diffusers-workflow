import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import ReferenceChip from './ReferenceChip.svelte'
import { ASSET, OUTPUT } from '../references'
import { copyText } from '../clipboard'

vi.mock('../clipboard', () => ({
  copyText: vi.fn(() => Promise.resolve(true)),
}))
vi.mock('../toast', () => ({ notify: { error: vi.fn() } }))

const TITLE = 'Copy the reference a workflow argument takes'

afterEach(() => {
  cleanup()
  vi.mocked(copyText).mockClear()
})

it('copies output:<name>', async () => {
  render(ReferenceChip, { props: { prefix: OUTPUT, name: 'demo/r1/a.png' } })
  expect(screen.getByText(`${OUTPUT}demo/r1/a.png`).tagName).toBe('CODE')
  await fireEvent.click(screen.getByRole('button', { name: TITLE }))
  expect(copyText).toHaveBeenCalledWith(`${OUTPUT}demo/r1/a.png`)
})

it('copies asset:<name>', async () => {
  render(ReferenceChip, { props: { prefix: ASSET, name: 'faces/hero.png' } })
  expect(screen.getByText(`${ASSET}faces/hero.png`).tagName).toBe('CODE')
  await fireEvent.click(screen.getByRole('button', { name: TITLE }))
  expect(copyText).toHaveBeenCalledWith(`${ASSET}faces/hero.png`)
})
