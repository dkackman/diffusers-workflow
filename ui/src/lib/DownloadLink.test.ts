import { render } from '@testing-library/svelte'
import { expect, it } from 'vitest'
import DownloadLink from './DownloadLink.svelte'

it('renders a download link with the given href and default label', () => {
  const { container } = render(DownloadLink, { href: '/api/foo/download' })
  const link = container.querySelector('a')!
  expect(link.getAttribute('href')).toBe('/api/foo/download')
  expect(link.hasAttribute('download')).toBe(true)
  expect(link.getAttribute('aria-label')).toBe('Download')
  expect(link.getAttribute('title')).toBe('Download')
})

it('uses a custom label for aria-label and title when given', () => {
  const { container } = render(DownloadLink, {
    href: '/api/bar/download',
    label: 'Download workflow',
  })
  const link = container.querySelector('a')!
  expect(link.getAttribute('aria-label')).toBe('Download workflow')
  expect(link.getAttribute('title')).toBe('Download workflow')
})
