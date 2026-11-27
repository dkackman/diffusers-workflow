import {
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import PreferenceCard from './PreferenceCard.svelte'
import { resetApi } from '../test/apiMock'

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
vi.mock('../confirm.svelte', () => ({ confirmDialog: vi.fn() }))

const entry = (over = {}) => ({
  source: 'shipped' as const,
  etag: '"k1"',
  ladder: {
    draft: ['models/z-image', 'models/z-image-sdnq'],
    best: ['models/flux-dev'],
  },
  resolutions: {
    draft: { target: 'models/z-image', skipped: [] },
    best: { target: 'models/flux-dev', skipped: [] },
  },
  used_by: ['templates/a', 'templates/b'],
  ...over,
})
const props = (e: ReturnType<typeof entry> | null) => ({
  key: 'text_to_image',
  entry: e,
  defaultTier: 'best',
  catalog: [],
  onsaved: vi.fn(),
  onreset: vi.fn(),
  onreload: vi.fn(),
})
const tier = (name: string) =>
  document.querySelector<HTMLElement>(`.tier[data-tier="${name}"]`)!

beforeEach(resetApi)
afterEach(cleanup)

it('a ready tier says ready, and the header names the key', () => {
  render(PreferenceCard, props(entry()))
  expect(within(tier('draft')).getByText('ready')).toBeTruthy()
  expect(screen.getByText('Text to image')).toBeTruthy()
  expect(screen.getByText('as shipped')).toBeTruthy()
})

it('a pick that downloads first links to the downloaded models', () => {
  const e = entry()
  e.resolutions.best = {
    target: 'models/flux-dev',
    skipped: [],
    notes: ['will download 41.2 GB'],
  } as never
  render(PreferenceCard, props(e))
  const link = within(tier('best')).getByRole('link', {
    name: 'downloads 41.2 GB before its first job',
  })
  expect(link.getAttribute('href')).toBe('#/server/models/downloaded')
})

it('a skipped target is struck with its reason', () => {
  const e = entry()
  e.resolutions.draft = {
    target: 'models/z-image',
    skipped: [{ workflow: 'models/minimax-h3-image', reason: 'cuda-only' }],
  } as never
  render(PreferenceCard, props(e))
  const struck = tier('draft').querySelector('s')!
  expect(struck.textContent).toBe('models/minimax-h3-image')
  expect(tier('draft').textContent).toContain('needs CUDA')
})

it('the models after the pick are fallback, not checked here', () => {
  render(PreferenceCard, props(entry()))
  expect(tier('draft').textContent).toContain(
    'then models/z-image-sdnq (fallback, not checked here)',
  )
  expect(tier('best').textContent).not.toContain('fallback')
})

it('the default tier carries the default tag', () => {
  render(PreferenceCard, props(entry()))
  expect(within(tier('best')).getByText('default')).toBeTruthy()
  expect(within(tier('draft')).queryByText('default')).toBeNull()
})

it('used by N workflows expands to each name', async () => {
  render(PreferenceCard, props(entry()))
  const summary = screen.getByText('used by 2 workflows')
  await fireEvent.click(summary)
  expect(screen.getByText('templates/a')).toBeTruthy()
  expect(screen.getByText('templates/b')).toBeTruthy()
})

it('a tier with no fit reads Nothing fits this machine inside the card', () => {
  const error =
    "Preference 'k' tier best names a target not in the catalog: 'gone/x'"
  const e = entry()
  e.resolutions.best = { error } as never
  render(PreferenceCard, props(e))
  const card = screen.getByRole('region', { name: 'preference text_to_image' })
  expect(
    within(card).getByText(/Nothing fits this machine/).textContent,
  ).toContain(error)
})

it('Reset to shipped shows only for the server own list; Edit opens the editor', async () => {
  const { unmount } = render(PreferenceCard, props(entry()))
  expect(screen.queryByRole('button', { name: 'Reset to shipped' })).toBeNull()
  unmount()
  render(PreferenceCard, props(entry({ source: 'server' })))
  expect(screen.getByText('changed on this server')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Reset to shipped' })).toBeTruthy()
  await fireEvent.click(screen.getByRole('button', { name: 'Edit' }))
  expect(screen.getByRole('button', { name: 'Save' })).toBeTruthy()
})
