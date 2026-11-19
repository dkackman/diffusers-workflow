import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import GalleryDetail from './GalleryDetail.svelte'
import type { GalleryFile } from '../types'

const setNote = vi.hoisted(() => vi.fn())
const assess = vi.hoisted(() => vi.fn<(name: string) => Promise<unknown>>())
vi.mock('../api', () => ({
  setNote: (t: string, n: string | null) => setNote(t, n),
  api: {
    outputDownloadUrl: (name: string) => `/download/${name}`,
    assess: (name: string) => assess(name),
  },
}))

afterEach(() => {
  cleanup()
  assess.mockReset()
})

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
  expect(order).toEqual([
    'role',
    'Note',
    'Findings',
    'Settings',
    'Made from',
    'Location',
  ])
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

it("labels a saved file note as the file's own, and a cleared one returns the run note", async () => {
  const run = {
    ...props,
    note: 'for the client',
    runNote: 'for the client',
    noteSource: 'run' as const,
  }
  const spy = setNote
  const onnotesaved = vi.fn(async () => {})
  const { rerender } = render(GalleryDetail, { ...run, onnotesaved })
  await fireEvent.click(screen.getByRole('button', { name: 'Note this file' }))
  spy.mockResolvedValue({
    target: file.name,
    note: 'mine',
    note_source: 'file',
  })
  onnotesaved.mockImplementation(async () => {
    await rerender({ ...run, note: 'mine', noteSource: 'file', onnotesaved })
  })
  await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
  await fireEvent.input(screen.getByRole('textbox'), {
    target: { value: 'mine' },
  })
  await fireEvent.click(screen.getByRole('button', { name: /save/i }))
  await waitFor(() => expect(screen.getByText('File note')).toBeTruthy())
  expect(screen.getByText('mine')).toBeTruthy()

  spy.mockResolvedValue({
    target: file.name,
    note: 'for the client',
    note_source: 'run',
  })
  onnotesaved.mockImplementation(async () => {
    await rerender({ ...run, onnotesaved })
  })
  await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
  await fireEvent.input(screen.getByRole('textbox'), { target: { value: '' } })
  await fireEvent.click(screen.getByRole('button', { name: /save/i }))
  await waitFor(() => expect(screen.getByText('Run note')).toBeTruthy())
  expect(screen.getByText('for the client')).toBeTruthy()
})

const finding = (
  severity: 'warn' | 'info',
  says: string,
  at: number | null = null,
) => ({
  rule: `${severity}_rule`,
  severity,
  at,
  value: 1,
  threshold: 0,
  says,
})

const video: GalleryFile = {
  ...file,
  name: 'wf/run-1/final/clip.mp4',
  url: '/outputs/wf/run-1/final/clip.mp4',
  kind: 'video',
  label: 'clip.mp4',
}

const reply = (findings: ReturnType<typeof finding>[]) => ({
  name: video.name,
  source: 'output',
  kind: 'video',
  findings,
  rules_applied: ['near_silent', 'shot_dead_air'],
  rules_skipped: [],
  not_applicable: { analyze_seams: 'no shot boundaries recorded' },
})

const findingsRegion = () => screen.getByRole('region', { name: 'Findings' })

it("lists the metadata's findings warn first", () => {
  render(GalleryDetail, {
    ...props,
    findings: [finding('info', 'quiet note'), finding('warn', 'too loud')],
  })
  const rows = [...findingsRegion().querySelectorAll('li')]
  expect(rows.map((r) => r.querySelector('.severity')?.textContent)).toEqual([
    'warn',
    'info',
  ])
  expect(rows[0].textContent).toContain('too loud')
})

it('says No findings when there are none', () => {
  render(GalleryDetail, props)
  expect(findingsRegion().textContent).toContain('No findings')
})

it("disables an image's Assess and says why", () => {
  render(GalleryDetail, props)
  const button = screen.getByRole('button', { name: 'Assess' })
  expect((button as HTMLButtonElement).disabled).toBe(true)
  expect(button.getAttribute('title')).toBe(
    'a still has no shots, seams or soundtrack to measure',
  )
  expect(assess).not.toHaveBeenCalled()
})

it("Assess shows the reply's findings, their time and what did not apply", async () => {
  assess.mockResolvedValue(
    reply([finding('info', 'picture jumps', 64.25), finding('warn', 'silent')]),
  )
  render(GalleryDetail, { ...props, file: video })
  await fireEvent.click(screen.getByRole('button', { name: 'Assess' }))
  expect(assess).toHaveBeenCalledWith(video.name)
  await waitFor(() =>
    expect(findingsRegion().querySelectorAll('li')).toHaveLength(2),
  )
  const rows = [...findingsRegion().querySelectorAll('li')]
  expect(rows[0].textContent).toContain('silent')
  expect(rows[1].querySelector('.at')?.textContent).toBe('1:04.3')
  expect(findingsRegion().textContent).toContain(
    'analyze_seams not applicable (no shot boundaries recorded)',
  )
})

it('shows Assessing… while the request runs', async () => {
  assess.mockImplementation(() => new Promise(() => {}))
  render(GalleryDetail, { ...props, file: video })
  await fireEvent.click(screen.getByRole('button', { name: 'Assess' }))
  const busy = await screen.findByRole('button', { name: /Assessing/ })
  expect((busy as HTMLButtonElement).disabled).toBe(true)
})

it('a failed Assess keeps the earlier list and shows the error', async () => {
  assess.mockResolvedValueOnce(reply([finding('warn', 'silent')]))
  render(GalleryDetail, { ...props, file: video })
  await fireEvent.click(screen.getByRole('button', { name: 'Assess' }))
  await waitFor(() => expect(findingsRegion().textContent).toContain('silent'))
  assess.mockRejectedValueOnce(new Error('cannot read this file'))
  await fireEvent.click(screen.getByRole('button', { name: 'Assess' }))
  expect((await screen.findByRole('alert')).textContent).toContain(
    'cannot read this file',
  )
  expect(findingsRegion().textContent).toContain('silent')
})
