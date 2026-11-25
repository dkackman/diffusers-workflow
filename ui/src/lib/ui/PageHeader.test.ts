import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import { tick } from 'svelte'
import { afterEach, expect, it, vi } from 'vitest'
import PageHeader from './PageHeader.svelte'
import { setHashQuery } from '../router.svelte'
import { pageFrame } from '../pageFrame.svelte'
import { api } from '../test/apiMock'

vi.mock('../api', async () =>
  (await import('../test/apiMock')).apiModule({
    TERMINAL_STATUSES: ['succeeded', 'failed', 'cancelled'],
  }),
)
api.listWorkspaces.mockImplementation(() => new Promise(() => {}))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

afterEach(() => {
  cleanup()
  location.hash = ''
})

const heading = (name: string) => screen.getByRole('heading', { name })

it('sets the tab title from its title', async () => {
  render(PageHeader, { props: { title: 'Gallery' } })
  await tick()
  expect(pageFrame.title).toBe('Gallery')
  expect(document.title).toBe('Gallery · diffusers-workflow')
})

it('PageHeader focuses its h1 on a path change, never on a query change or the first render', async () => {
  render(PageHeader, { props: { title: 'Jobs' } })
  await tick()
  await tick()
  expect(document.activeElement).not.toBe(heading('Jobs'))

  // typing in a filter writes the query: no navigation, no focus
  setHashQuery({ q: 'flux' })
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  await tick()
  await tick()
  expect(document.activeElement).not.toBe(heading('Jobs'))

  // a new path is a navigation
  location.hash = '#/ws/studio/jobs'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  await waitFor(() => expect(document.activeElement).toBe(heading('Jobs')))
  expect(heading('Jobs')).toHaveAttribute('tabindex', '-1')
})

it('PageHeader keeps focus where it is on a real same-path hashchange, forward or Back', async () => {
  render(PageHeader, { props: { title: 'Jobs' } })
  location.hash = '#/ws/other/jobs'
  await waitFor(() => expect(document.activeElement).toBe(heading('Jobs')))
  ;(document.activeElement as HTMLElement).blur()

  const changed = () =>
    new Promise<void>((r) =>
      window.addEventListener('hashchange', () => r(), { once: true }),
    )
  // a hand-edited query: the hash really changes, the path does not
  let done = changed()
  location.hash = '#/ws/other/jobs?status=failed'
  await done
  await tick()
  await tick()
  expect(document.activeElement).not.toBe(heading('Jobs'))

  done = changed()
  location.hash = '#/ws/other/jobs?status=running'
  await done
  await tick()
  await tick()
  expect(document.activeElement).not.toBe(heading('Jobs'))

  // Back to the first query: still the same path
  done = changed()
  history.back()
  await done
  await tick()
  await tick()
  expect(document.activeElement).not.toBe(heading('Jobs'))
})

it('renders the count with its noun', () => {
  render(PageHeader, {
    props: { title: 'Gallery', count: { n: 3, noun: 'file' } },
  })
  expect(screen.getByText('3 files')).toBeInTheDocument()
  cleanup()
  render(PageHeader, {
    props: { title: 'Gallery', count: { n: 1, noun: 'file' } },
  })
  expect(screen.getByText('1 file')).toBeInTheDocument()
  cleanup()
  render(PageHeader, {
    props: {
      title: 'Assets',
      count: { n: 2, noun: 'library', plural: 'libraries' },
    },
  })
  expect(screen.getByText('2 libraries')).toBeInTheDocument()
})

it('renders one back-link form', () => {
  render(PageHeader, {
    props: {
      title: 'Compare',
      back: { href: '#/ws/studio/gallery', label: 'Gallery' },
    },
  })
  const back = screen.getByRole('link', { name: '← Gallery' })
  expect(back).toHaveAttribute('href', '#/ws/studio/gallery')
  expect(document.querySelectorAll('.pagehead a').length).toBe(1)
})

it('a nested header is an h2 that never names the tab or takes focus', async () => {
  pageFrame.title = 'Status'
  render(PageHeader, { props: { title: 'All jobs', nested: true } })
  location.hash = '#/server/status-nested'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  await tick()
  await tick()
  expect(
    screen.getByRole('heading', { level: 2, name: 'All jobs' }),
  ).toBeTruthy()
  expect(screen.queryByRole('heading', { level: 1 })).toBeNull()
  expect(pageFrame.title).toBe('Status')
  expect(document.activeElement).toBe(document.body)
})

it('renders n of m', () => {
  const { container, rerender } = render(PageHeader, {
    props: { title: 'Gallery', count: { n: 3, of: 12, noun: 'file' } },
  })
  expect(container.querySelector('.count')?.textContent).toBe('3 of 12 files')
  // the same number twice says it once
  void rerender({ title: 'Gallery', count: { n: 12, of: 12, noun: 'file' } })
  return tick().then(() =>
    expect(container.querySelector('.count')?.textContent).toBe('12 files'),
  )
})
