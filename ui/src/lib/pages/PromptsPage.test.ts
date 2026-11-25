import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import PromptsPage from './PromptsPage.svelte'
import '../router.svelte'
import { DEFAULT_WORKSPACE, workspace } from '../workspace.svelte'
import { feed } from '../test/feedMock'

const detail = (origin: string, writable: boolean) => ({
  description: '',
  intended_model: '',
  tags: [] as string[],
  text: 'a prompt',
  origin,
  writable,
})

const listPrompts = vi.hoisted(() =>
  vi.fn(() =>
    Promise.resolve({
      libraries: [
        { origin: 'workspace', root: '/srv/prompts', writable: true },
        { origin: 'examples', root: '/ex/prompts', writable: false },
      ],
      shadowed: [],
      prompts: ['mine', 'theirs'],
      details: {},
    } as unknown),
  ),
)
vi.mock('../api', () => ({
  api: { listPrompts: () => listPrompts() },
}))

vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))

const answer = (prompts: string[]) => ({
  libraries: [{ origin: 'workspace', root: '/srv/prompts', writable: true }],
  shadowed: [],
  prompts,
  details: Object.fromEntries(
    prompts.map((p) => [p, detail('workspace', true)]),
  ),
})

beforeEach(() => {
  location.hash = '#/ws/default/prompts'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  listPrompts.mockResolvedValue({
    libraries: [
      { origin: 'workspace', root: '/srv/prompts', writable: true },
      { origin: 'examples', root: '/ex/prompts', writable: false },
    ],
    shadowed: [],
    prompts: ['mine', 'theirs'],
    details: {
      mine: detail('workspace', true),
      theirs: detail('examples', false),
    },
  })
})
afterEach(() => {
  cleanup()
  listPrompts.mockClear()
  workspace.current = DEFAULT_WORKSPACE
})
const settle = async () => {
  for (let i = 0; i < 3; i++) await new Promise((r) => setTimeout(r, 10))
}

it('badges a prompt by the origin on its own detail', async () => {
  render(PromptsPage)
  await waitFor(() => expect(screen.getByText('theirs')).toBeTruthy())
  expect(
    screen.getByTitle('read-only: from the examples library').textContent,
  ).toContain('examples')
  // The workspace's own prompt carries no badge
  expect(screen.queryAllByTitle(/read-only/)).toHaveLength(1)
})

it('names the writable workspace root as where prompts are read from', async () => {
  render(PromptsPage)
  await waitFor(() => expect(screen.getByText('/srv/prompts')).toBeTruthy())
})

it('loads once on mount, and again quietly when prompts move', async () => {
  render(PromptsPage)
  await waitFor(() => expect(screen.getByText('theirs')).toBeTruthy())
  await settle()
  expect(listPrompts).toHaveBeenCalledTimes(1)

  listPrompts.mockResolvedValue(answer(['mine', 'theirs', 'fresh']))
  feed.move('default', 'prompts')
  // No blanking while it asks
  expect(screen.getByText('mine')).toBeTruthy()
  await waitFor(() => expect(screen.getByText('fresh')).toBeTruthy())
  expect(listPrompts).toHaveBeenCalledTimes(2)
})

it('reloads the new workspace prompts on a workspace switch', async () => {
  render(PromptsPage)
  await waitFor(() => expect(screen.getByText('theirs')).toBeTruthy())

  listPrompts.mockResolvedValue(answer(['elsewhere']))
  workspace.current = 'other'

  await waitFor(() => expect(screen.getByText('elsewhere')).toBeTruthy())
  expect(screen.queryByText('theirs')).toBeNull()
  expect(listPrompts).toHaveBeenCalledTimes(2)
})

it('a quiet reload that fails keeps the list', async () => {
  render(PromptsPage)
  await waitFor(() => expect(screen.getByText('theirs')).toBeTruthy())
  listPrompts.mockRejectedValue(new Error('boom'))
  feed.move('default', 'prompts')
  await waitFor(() => expect(listPrompts).toHaveBeenCalledTimes(2))
  await settle()
  expect(screen.getByText('theirs')).toBeTruthy()
  expect(screen.queryByText(/boom/)).toBeNull()
})

it('another kind moving, or another workspace, reloads nothing', async () => {
  render(PromptsPage)
  await waitFor(() => expect(screen.getByText('theirs')).toBeTruthy())
  feed.move('default', 'assets')
  feed.move('other', 'prompts')
  await settle()
  expect(listPrompts).toHaveBeenCalledTimes(1)
})

it('shows a skeleton while loading, not the empty state', async () => {
  listPrompts.mockReturnValue(new Promise(() => {}))
  render(PromptsPage)
  expect(screen.getByRole('status', { name: 'loading prompts' })).toBeTruthy()
  expect(screen.queryByText(/No prompts yet/)).toBeNull()
})

it('a failed first load offers Retry and Retry loads', async () => {
  listPrompts.mockRejectedValueOnce(new Error('boom'))
  render(PromptsPage)
  await screen.findByText(/Couldn't load prompts/)
  await fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await waitFor(() => expect(screen.getByText('theirs')).toBeTruthy())
  expect(listPrompts).toHaveBeenCalledTimes(2)
})

it('filtered to nothing offers Clear filters', async () => {
  render(PromptsPage)
  await waitFor(() => expect(screen.getByText('theirs')).toBeTruthy())
  await fireEvent.input(screen.getByPlaceholderText('filter…'), {
    target: { value: 'zzz' },
  })
  expect(screen.getByText('Nothing matches those filters.')).toBeTruthy()
  await fireEvent.click(screen.getByRole('button', { name: 'Clear filters' }))
  await waitFor(() => expect(screen.getByText('theirs')).toBeTruthy())
})

it('filtered to nothing still says where prompts are read from', async () => {
  render(PromptsPage)
  await waitFor(() => expect(screen.getByText('theirs')).toBeTruthy())
  await fireEvent.input(screen.getByPlaceholderText('filter…'), {
    target: { value: 'zzz' },
  })
  await screen.findByText('Nothing matches those filters.')
  expect(screen.getByText('/srv/prompts')).toBeTruthy()
})

it('an empty library with a filter typed says there are no prompts yet', async () => {
  listPrompts.mockResolvedValue(answer([]))
  render(PromptsPage)
  await fireEvent.input(screen.getByPlaceholderText('filter…'), {
    target: { value: 'zzz' },
  })
  await screen.findByText(/No prompts yet/)
  expect(screen.queryByText('Nothing matches those filters.')).toBeNull()
  expect(screen.getByText('/srv/prompts')).toBeTruthy()
})

it('a tag chip sets the filter on a press and clears it on a second press', async () => {
  listPrompts.mockResolvedValue({
    ...answer(['mine', 'theirs']),
    details: {
      mine: { ...detail('workspace', true), tags: ['anime'] },
      theirs: detail('workspace', true),
    },
  })
  render(PromptsPage)
  await waitFor(() => expect(screen.getByText('theirs')).toBeTruthy())
  const input = screen.getByPlaceholderText('filter…') as HTMLInputElement
  const chip = screen.getByRole('button', { name: 'anime' })

  await fireEvent.click(chip)
  expect(input.value).toBe('anime')
  expect(chip.getAttribute('aria-pressed')).toBe('true')

  await fireEvent.click(chip)
  expect(input.value).toBe('')
  expect(chip.getAttribute('aria-pressed')).toBe('false')
})

it('a quiet reload keeps focus, filter and selection', async () => {
  listPrompts.mockResolvedValue(answer(['food', 'fog', 'bar']))
  render(PromptsPage)
  await waitFor(() => expect(screen.getByText('fog')).toBeTruthy())
  const input = screen.getByPlaceholderText('filter…') as HTMLInputElement
  input.focus()
  await fireEvent.input(input, { target: { value: 'fo' } })
  await waitFor(() => expect(screen.queryByText('bar')).toBeNull())

  const next = answer(['food', 'fog', 'bar'])
  next.details.fog = { ...next.details.fog, text: 'a foggier prompt' }
  listPrompts.mockResolvedValue(next)
  feed.move('default', 'prompts')
  await screen.findByText('a foggier prompt')

  expect(document.activeElement).toBe(input)
  expect(input.value).toBe('fo')
  const marked = [...document.querySelectorAll('.card.row-changed')]
  expect(marked.map((c) => c.querySelector('.cardname')?.textContent)).toEqual([
    'fog',
  ])
})

it('the name has its own line, and the model chip sits with the tags', async () => {
  const long = 'a-very-long-prompt-name-that-would-have-been-cut-by-its-chip'
  listPrompts.mockResolvedValue({
    ...answer([long]),
    details: {
      [long]: {
        ...detail('workspace', true),
        intended_model: 'flux-dev',
        tags: ['anime'],
      },
    },
  })
  render(PromptsPage)
  const name = await screen.findByText(long)
  const chip = screen.getByRole('button', { name: 'flux-dev' })
  const tag = screen.getByRole('button', { name: 'anime' })
  expect(name.parentElement?.contains(chip)).toBe(false)
  expect(chip.parentElement).toBe(tag.parentElement)
  expect(chip.parentElement?.firstElementChild).toBe(chip)
  expect(name.getAttribute('title')).toBe(long)
})

const dated = (prompts: Record<string, number | undefined>) => ({
  libraries: [{ origin: 'workspace', root: '/srv/prompts', writable: true }],
  shadowed: [],
  prompts: Object.keys(prompts),
  details: Object.fromEntries(
    Object.entries(prompts).map(([p, mtime]) => [
      p,
      { ...detail('workspace', true), ...(mtime ? { mtime } : {}) },
    ]),
  ),
})
const cardOf = (name: string) =>
  screen.getByText(name, { selector: '.cardname' }).closest('.card')!

it('PromptsPage says when each prompt changed', async () => {
  const now = Math.floor(Date.now() / 1000)
  listPrompts.mockResolvedValue(dated({ fox: now - 2 * 3600, owl: undefined }))
  render(PromptsPage)
  await waitFor(() => expect(screen.getByText('fox')).toBeTruthy())
  const stamp = cardOf('fox').querySelector('time')!
  expect(stamp.textContent?.trim()).toBe('changed 2 h ago')
  expect(stamp.getAttribute('title')).toBeTruthy()
  expect(cardOf('owl').querySelector('time')).toBeNull()
})

it('the newest-first sort', async () => {
  const now = Math.floor(Date.now() / 1000)
  listPrompts.mockResolvedValue(
    dated({ ant: now - 86_400, bee: now - 60, cat: now - 3600 }),
  )
  render(PromptsPage)
  await waitFor(() => expect(screen.getByText('ant')).toBeTruthy())
  const names = () =>
    [...document.querySelectorAll('.cardname')].map((n) => n.textContent)
  expect(names()).toEqual(['ant', 'bee', 'cat'])
  await fireEvent.click(screen.getByRole('button', { name: 'Newest' }))
  await waitFor(() => expect(names()).toEqual(['bee', 'cat', 'ant']))
  expect(location.hash).toContain('sort=newest')
})
