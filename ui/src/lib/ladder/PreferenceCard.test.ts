import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import PreferenceCard from './PreferenceCard.svelte'
import { addLeaveGuard } from '../router.svelte'
import { api, resetApi } from '../test/apiMock'

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
vi.mock('../confirm.svelte', () => ({ confirmDialog: vi.fn() }))
const removeGuard = vi.hoisted(() => vi.fn())
vi.mock('../router.svelte', async (original) => ({
  ...(await original<typeof import('../router.svelte')>()),
  addLeaveGuard: vi.fn(() => removeGuard),
}))

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
const props = (e: ReturnType<typeof entry> | null, over = {}) => ({
  key: 'text_to_image',
  entry: e,
  defaultTier: 'best',
  catalog: [],
  onsaved: vi.fn(),
  onreset: vi.fn(),
  onreload: vi.fn(),
  ...over,
})
const tier = (name: string) =>
  document.querySelector<HTMLElement>(`.tier[data-tier="${name}"]`)!

beforeEach(() => {
  resetApi()
  vi.mocked(addLeaveGuard).mockClear()
  removeGuard.mockClear()
  api.checkPreference.mockResolvedValue({ tiers: {} })
})
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

it('used by N workflows expands to each name, linked where it lives', async () => {
  render(
    PreferenceCard,
    props(entry(), { readOnly: (name: string) => name === 'templates/a' }),
  )
  const summary = screen.getByText('used by 2 workflows')
  await fireEvent.click(summary)
  // a shipped (read-only) workflow opens under Examples, as WorkflowsPage links it
  expect(
    screen.getByRole('link', { name: 'templates/a' }).getAttribute('href'),
  ).toBe('#/shared/examples/templates/a')
  // one in a writable library opens in the workspace
  expect(
    screen.getByRole('link', { name: 'templates/b' }).getAttribute('href'),
  ).toMatch(/^#\/ws\/[^/]+\/workflows\/templates\/b$/)
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
  // nothing changed yet: the editor closes rather than saves
  expect(screen.getByRole('button', { name: 'Close editor' })).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Save' })).toBeNull()
})

const edit = async () =>
  fireEvent.click(screen.getByRole('button', { name: 'Edit' }))

it('Edit then Save sends the draft with its etag, then reads the page again', async () => {
  const saved = {
    key: 'text_to_image',
    ...entry({ source: 'server', etag: '"k2"' }),
  }
  api.setPreference.mockResolvedValue(saved)
  const p = props(entry())
  render(PreferenceCard, p)
  await edit()
  await fireEvent.click(
    screen.getByRole('button', { name: 'Remove models/z-image-sdnq' }),
  )
  expect(
    screen.getByText(/Jobs already queued keep the model they were given/),
  ).toBeTruthy()
  await fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  await waitFor(() => expect(p.onreload).toHaveBeenCalled())
  expect(api.setPreference).toHaveBeenCalledWith(
    'text_to_image',
    { draft: ['models/z-image'], best: ['models/flux-dev'] },
    '"k1"',
  )
  expect(api.setPreference.mock.invocationCallOrder[0]).toBeLessThan(
    p.onreload.mock.invocationCallOrder[0],
  )
  expect(p.onsaved).toHaveBeenCalledWith(saved)
  // saved: the card reads again, out of the editor
  expect(screen.queryByRole('button', { name: 'Save' })).toBeNull()
})

it('a dirty draft holds a leave guard, and Discard lets it go', async () => {
  render(PreferenceCard, props(entry()))
  await edit()
  expect(addLeaveGuard).not.toHaveBeenCalled()
  await fireEvent.click(
    screen.getByRole('button', { name: 'Remove models/z-image-sdnq' }),
  )
  expect(addLeaveGuard).toHaveBeenCalledTimes(1)
  await fireEvent.click(screen.getByRole('button', { name: 'Discard' }))
  expect(removeGuard).toHaveBeenCalledTimes(1)
  expect(screen.queryByRole('button', { name: 'Save' })).toBeNull()
  expect(tier('draft').textContent).toContain('models/z-image-sdnq')
})

it('Edit waits while another card holds the open edit', () => {
  render(PreferenceCard, props(entry(), { locked: true }))
  const button = screen.getByRole<HTMLButtonElement>('button', { name: 'Edit' })
  expect(button.disabled).toBe(true)
  expect(button.title).toBe('Save or discard the open edit first')
})

it('a target the dry run refuses holds Save', async () => {
  api.checkPreference.mockResolvedValue({
    tiers: {
      draft: {
        target: 'models/z-image',
        skipped: [],
        target_errors: { 'gone/x': 'not in the catalog' },
      },
    },
  })
  render(PreferenceCard, props(entry()))
  await edit()
  await fireEvent.click(
    screen.getByRole('button', { name: 'Remove models/z-image-sdnq' }),
  )
  await waitFor(() => expect(api.checkPreference).toHaveBeenCalled())
  await waitFor(() =>
    expect(
      screen.getByRole<HTMLButtonElement>('button', { name: 'Save' }).disabled,
    ).toBe(true),
  )
  expect(api.checkPreference).toHaveBeenLastCalledWith('text_to_image', {
    ladder: { draft: ['models/z-image'], best: ['models/flux-dev'] },
    full: false,
    known_blocked: [],
  })
})

it("a dry run's shape refusal shows and holds Save", async () => {
  const { ApiError } = await import('../apiErrors')
  api.checkPreference.mockRejectedValue(
    new ApiError("tier draft: 'x' is not a target", 422),
  )
  render(PreferenceCard, props(entry()))
  await edit()
  await fireEvent.click(
    screen.getByRole('button', { name: 'Remove models/z-image-sdnq' }),
  )
  expect((await screen.findByRole('alert')).textContent).toContain(
    "Couldn't check text_to_image",
  )
  expect(
    screen.getByRole<HTMLButtonElement>('button', { name: 'Save' }).disabled,
  ).toBe(true)
})

// A write made elsewhere while this one was being edited: the edit stays,
// and saving it is checked against what it started from, so the server
// answers 412 rather than the edit silently replacing the other write
it('an edited draft keeps its own etag when a read moves the key', async () => {
  api.setPreference.mockResolvedValue({
    key: 'text_to_image',
    ...entry({ etag: '"k3"' }),
  })
  const first = entry({ ladder: { standard: ['a', 'b'] } })
  const { rerender } = render(PreferenceCard, props(first))
  await edit()
  await fireEvent.click(screen.getByRole('button', { name: 'Remove a' }))
  await rerender(
    props(entry({ etag: '"k2"', ladder: { standard: ['c'] } }), {
      editing: true,
    }),
  )
  expect(screen.queryByText('c')).toBeNull()
  await fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  await waitFor(() =>
    expect(api.setPreference).toHaveBeenCalledWith(
      'text_to_image',
      { standard: ['b'] },
      '"k1"',
    ),
  )
})

it('an unedited draft follows the read', async () => {
  const { rerender } = render(
    PreferenceCard,
    props(entry({ ladder: { standard: ['a'] } })),
  )
  await edit()
  await rerender(
    props(entry({ etag: '"k2"', ladder: { standard: ['c'] } }), {
      editing: true,
    }),
  )
  expect(await screen.findByText('c')).toBeTruthy()
})

it('Reset drops an edit along with the server ladder', async () => {
  const { confirmDialog } = await import('../confirm.svelte')
  vi.mocked(confirmDialog).mockResolvedValueOnce(true)
  api.resetPreference.mockResolvedValue({
    key: 'text_to_image',
    removed: false,
    source: 'shipped',
    etag: '"s1"',
    ladder: { standard: ['shipped-one'] },
  })
  const p = props(entry({ source: 'server', ladder: { standard: ['a'] } }))
  render(PreferenceCard, p)
  await edit()
  await fireEvent.click(screen.getByRole('button', { name: 'Remove a' }))
  await fireEvent.click(
    screen.getByRole('button', { name: 'Reset to shipped' }),
  )
  await waitFor(() => expect(p.onreset).toHaveBeenCalled())
  expect(api.resetPreference).toHaveBeenCalledWith('text_to_image', '"k1"')
  expect(screen.queryByRole('button', { name: 'Save' })).toBeNull()
})
