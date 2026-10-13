import { cleanup, render } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import GalleryDetail from './GalleryDetail.svelte'
import type { GalleryFile } from '../types'

vi.mock('../api', () => ({
  api: { outputDownloadUrl: (name: string) => `/download/${name}` },
}))

afterEach(cleanup)

const file: GalleryFile = {
  name: 'wf/run-1/final/a.png',
  folder: 'wf',
  subfolder: 'final',
  run_id: 'run-1',
  version: 1,
  url: '/outputs/wf/run-1/final/a.png',
  kind: 'image',
  size: 10,
  mtime: 1,
  label: 'a.png',
  role: 'final',
}

const props = {
  file,
  metadata: { model_name: 'org/model', seed: 7 },
  metadataLoading: false,
  sourceJob: null,
  role: 'final' as const,
  inputs: [
    { reference: 'asset:face.png', kind: 'asset' as const },
    { reference: 'output:wf/run-0/b.png', kind: 'output' as const },
  ],
  onremove: () => {},
  onclose: () => {},
}

it('orders the panel role, Settings, Made from, Location', () => {
  const { container } = render(GalleryDetail, props)
  const meta = container.querySelector('.meta') as HTMLElement
  const order = [...meta.children].map(
    (el) =>
      (el.querySelector('.role-tag') && 'role') ||
      (el.matches('details') && el.querySelector('summary')?.textContent) ||
      el.querySelector('.muted')?.textContent,
  )
  expect(order).toEqual(['role', 'Settings', 'Made from', 'Location'])
  // Settings starts collapsed
  expect((meta.querySelector('details') as HTMLDetailsElement).open).toBe(false)
  expect(meta.querySelector('.location code')?.textContent).toBe(
    'wf/run-1/final/',
  )
})

it('lists each run input as a link', () => {
  const { container } = render(GalleryDetail, props)
  const links = [...container.querySelectorAll('.madefrom a')]
  expect(links.map((a) => a.textContent)).toEqual([
    'face.png',
    'wf/run-0/b.png',
  ])
  expect(links[1].getAttribute('href')).toContain('wf/run-0/b.png')
})

it('omits Made from when the run read nothing', () => {
  const { container } = render(GalleryDetail, { ...props, inputs: [] })
  expect(container.querySelector('.madefrom')).toBeNull()
})
