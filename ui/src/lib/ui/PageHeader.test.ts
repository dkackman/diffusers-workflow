import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import { tick } from 'svelte'
import { afterEach, expect, it, vi } from 'vitest'
import PageHeader from './PageHeader.svelte'
import { setHashQuery } from '../router.svelte'
import { pageFrame } from '../pageFrame.svelte'

vi.mock('../api', () => ({
  api: { listWorkspaces: vi.fn(() => new Promise(() => {})) },
  TERMINAL_STATUSES: ['succeeded', 'failed', 'cancelled'],
}))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

afterEach(cleanup)

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
