import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
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
      (el.classList.contains('madefrom') && 'Made from') ||
      (el.querySelector('.role-tag') && 'role') ||
      (el.matches('details') && el.querySelector('summary')?.textContent) ||
      el.querySelector('.muted')?.textContent,
  )
  expect(order).toEqual(['role', 'Note', 'Settings', 'Made from', 'Location'])
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
  expect(
    [...container.querySelectorAll('.madefrom .role-tag')].map(
      (t) => (t as HTMLElement).dataset.role,
    ),
  ).toEqual(['input', 'input'])
  expect(links[1].getAttribute('href')).toContain('wf/run-0/b.png')
})

it('omits Made from when the run read nothing', () => {
  const { container } = render(GalleryDetail, { ...props, inputs: [] })
  expect(container.querySelector('.madefrom')).toBeNull()
})

it('shows an inherited run note as the run note and offers to note the file', async () => {
  render(GalleryDetail, {
    ...props,
    note: 'for the client',
    runNote: 'for the client',
    noteSource: 'run' as const,
  })
  expect(screen.getByText('Run note')).toBeTruthy()
  expect(screen.getByText('for the client')).toBeTruthy()
  await fireEvent.click(screen.getByRole('button', { name: 'Note this file' }))
  expect(screen.getByText('File note')).toBeTruthy()
  expect(screen.queryByText('for the client')).toBeNull()
})

it("shows the file's own note", () => {
  render(GalleryDetail, {
    ...props,
    note: 'keeper',
    noteSource: 'file' as const,
  })
  expect(screen.getByText('keeper')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Note this file' })).toBeNull()
})
