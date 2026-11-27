import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import LadderCard from './LadderCard.svelte'
import { api, resetApi } from '../test/apiMock'

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
vi.mock('../confirm.svelte', () => ({ confirmDialog: vi.fn() }))

const entry = (etag: string, standard: string[]) => ({
  source: 'server' as const,
  etag,
  ladder: { standard },
})
const props = (e: ReturnType<typeof entry>) => ({
  key: 'text_to_image',
  entry: e,
  catalog: [],
  defaultTier: 'standard',
  onsaved: vi.fn(),
  onreset: vi.fn(),
  onreload: vi.fn(),
})

beforeEach(resetApi)
afterEach(cleanup)

// A write made elsewhere while this one was being edited: the edit stays,
// and saving it is checked against what it started from, so the server
// answers 412 rather than the edit silently replacing the other write
it('an edited draft keeps its own etag when a read moves the key', async () => {
  api.setPreference.mockResolvedValue({
    key: 'text_to_image',
    ...entry('"k3"', ['b']),
  })
  const { rerender } = render(LadderCard, props(entry('"k1"', ['a', 'b'])))
  await fireEvent.click(screen.getByRole('button', { name: 'Remove a' }))
  await rerender(props(entry('"k2"', ['c'])))
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
  const { rerender } = render(LadderCard, props(entry('"k1"', ['a'])))
  await rerender(props(entry('"k2"', ['c'])))
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
  render(LadderCard, props(entry('"k1"', ['a'])))
  await fireEvent.click(screen.getByRole('button', { name: 'Remove a' }))
  await fireEvent.click(
    screen.getByRole('button', { name: 'Reset to shipped' }),
  )
  expect(await screen.findByText('shipped-one')).toBeTruthy()
})

it('Add clears the field it took the name from', async () => {
  render(LadderCard, props(entry('"k1"', ['a'])))
  const field = screen.getByLabelText<HTMLInputElement>(
    'workflow to add to best',
  )
  await fireEvent.input(field, { target: { value: 'models/z-image' } })
  await fireEvent.click(screen.getAllByRole('button', { name: 'Add' })[2])
  expect(await screen.findByText('models/z-image')).toBeTruthy()
  expect(
    screen.getByLabelText<HTMLInputElement>('workflow to add to best').value,
  ).toBe('')
})
