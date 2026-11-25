import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import RemoteChanged from './RemoteChanged.svelte'

afterEach(cleanup)

it('says what changed, in the spec copy', () => {
  render(RemoteChanged, { label: 'hero', ontheirs: vi.fn(), onmine: vi.fn() })
  expect(
    screen.getByText(
      'hero changed since you opened it, in another tab or by an agent',
    ),
  ).toBeTruthy()
})

it('Keep mine is the filled button; Load theirs, which discards the edits, is quiet', () => {
  render(RemoteChanged, { label: 'hero', ontheirs: vi.fn(), onmine: vi.fn() })
  const mine = screen.getByRole('button', { name: 'Keep mine' })
  const theirs = screen.getByRole('button', { name: 'Load theirs' })
  expect(mine.classList.contains('quiet')).toBe(false)
  expect(mine.classList.contains('bare')).toBe(false)
  expect([...theirs.classList]).toEqual(
    expect.arrayContaining(['quiet', 'danger']),
  )
})

it('each button answers its own way', async () => {
  const ontheirs = vi.fn()
  const onmine = vi.fn()
  render(RemoteChanged, { label: 'hero', ontheirs, onmine })
  await fireEvent.click(screen.getByRole('button', { name: 'Keep mine' }))
  expect(onmine).toHaveBeenCalledTimes(1)
  await fireEvent.click(screen.getByRole('button', { name: 'Load theirs' }))
  expect(ontheirs).toHaveBeenCalledTimes(1)
})
