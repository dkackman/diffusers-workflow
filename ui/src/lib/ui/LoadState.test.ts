import { createRawSnippet } from 'svelte'
import { afterEach, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render } from '@testing-library/svelte'
import LoadState from './LoadState.svelte'
import type { Resource } from '../resource.svelte'

afterEach(cleanup)

const rows = createRawSnippet((data: () => string[]) => ({
  render: () =>
    `<ul>${data()
      .map((row) => `<li>${row}</li>`)
      .join('')}</ul>`,
}))
const icon = createRawSnippet(() => ({ render: () => '<span>icon</span>' }))
const emptyText = createRawSnippet(() => ({
  render: () => '<span>No jobs yet</span>',
}))

function fake(state: Partial<Resource<string[]>>): Resource<string[]> {
  return {
    status: 'ready',
    data: undefined,
    error: undefined,
    stale: false,
    reload: vi.fn(() => Promise.resolve()),
    ...state,
  }
}

const show = (of: Resource<string[]>, more: Record<string, unknown> = {}) =>
  render(LoadState<string[]>, {
    props: {
      of,
      label: 'jobs',
      empty: (of.data ?? []).length === 0,
      emptyIcon: icon,
      emptyText,
      children: rows,
      ...more,
    },
  })

it('loading is a skeleton, not empty', () => {
  const { container, queryByText, getByRole } = show(
    fake({ status: 'loading' }),
  )
  expect(getByRole('status')).not.toBeNull()
  expect(queryByText('No jobs yet')).toBeNull()
  expect(container.querySelector('ul')).toBeNull()
})

it('an error with no data offers Retry', async () => {
  const of = fake({ status: 'error', error: new TypeError('Failed to fetch') })
  const { container, getByRole, queryByText } = show(of)
  expect(container.textContent).toContain(
    "Couldn't load jobs: Can't reach the server",
  )
  expect(container.textContent).toContain('Check that dw.serve is running.')
  expect(queryByText('No jobs yet')).toBeNull()
  await fireEvent.click(getByRole('button', { name: 'Retry' }))
  expect(of.reload).toHaveBeenCalledTimes(1)
})

it('filtered empty offers Clear filters', async () => {
  const onclear = vi.fn()
  const { container, getByRole, queryByText } = show(fake({ data: [] }), {
    filtered: true,
    onclear,
  })
  expect(container.textContent).toContain('Nothing matches those filters.')
  expect(queryByText('No jobs yet')).toBeNull()
  await fireEvent.click(getByRole('button', { name: 'Clear filters' }))
  expect(onclear).toHaveBeenCalledTimes(1)
})

it('empty with no filter is the Empty state', () => {
  const { queryByText } = show(fake({ data: [] }))
  expect(queryByText('No jobs yet')).not.toBeNull()
  expect(queryByText('icon')).not.toBeNull()
})

it('a failed refresh keeps the rows and says so', async () => {
  const of = fake({
    data: ['one', 'two'],
    stale: true,
    error: new TypeError('Failed to fetch'),
  })
  const { container, getByRole, getAllByRole } = show(of)
  expect(getAllByRole('listitem')).toHaveLength(2)
  expect(container.textContent).toContain(
    "Couldn't refresh jobs: Can't reach the server",
  )
  await fireEvent.click(getByRole('button', { name: 'Retry' }))
  expect(of.reload).toHaveBeenCalledTimes(1)
})

it('ready with rows shows only the rows', () => {
  const { container, queryByRole } = show(fake({ data: ['one'] }))
  expect(container.textContent).not.toContain("Couldn't")
  expect(queryByRole('button')).toBeNull()
})

it('quiet: a secondary panel shows nothing loading or failed', () => {
  for (const state of [
    fake({ status: 'loading' }),
    fake({ status: 'error', error: new TypeError('Failed to fetch') }),
  ]) {
    const { container, queryByRole } = show(state, {
      quiet: true,
      empty: false,
    })
    expect(queryByRole('status')).toBeNull()
    expect(queryByRole('alert')).toBeNull()
    expect(container.textContent?.trim()).toBe('')
    cleanup()
  }
})

it('quiet: a failed refresh keeps the data without a line', () => {
  const { container, queryByRole } = show(
    fake({
      data: ['a'],
      stale: true,
      error: new TypeError('Failed to fetch'),
    }),
    { quiet: true },
  )
  expect(queryByRole('alert')).toBeNull()
  expect(container.querySelectorAll('li')).toHaveLength(1)
})
