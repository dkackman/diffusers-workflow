import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import PreferencesPage from './PreferencesPage.svelte'
import '../router.svelte'
import { ApiError } from '../apiErrors'
import { confirmDialog } from '../confirm.svelte'
import { api, resetApi } from '../test/apiMock'

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
vi.mock('../confirm.svelte', () => ({ confirmDialog: vi.fn() }))

const CUDA = { workflow: 'models/minimax-h3-image', devices: ['cuda'] }
const LADDER = {
  draft: ['templates/text-to-image'],
  standard: ['models/z-image', 'models/z-image-sdnq'],
  best: [CUDA, 'models/qwen-image-2.1'],
}
const RESOLUTIONS = {
  draft: { target: 'templates/text-to-image', skipped: [] },
  standard: { target: 'models/z-image', skipped: [] },
  best: {
    target: 'models/qwen-image-2.1',
    skipped: [{ workflow: 'models/minimax-h3-image', reason: 'cuda only' }],
  },
}

function read(source = 'shipped') {
  return {
    default_tier: 'standard',
    etag: '"tier1"',
    preferences: {
      text_to_image: {
        source,
        etag: '"k1"',
        ladder: LADDER,
        resolutions: RESOLUTIONS,
      },
    },
  }
}

beforeEach(() => {
  resetApi()
  vi.mocked(confirmDialog).mockReset()
  api.getPreferences.mockResolvedValue(read())
  api.listWorkflows.mockResolvedValue({
    workflows: ['models/z-image', 'models/flux-dev'],
  })
})
afterEach(cleanup)

async function card() {
  return screen.findByRole('region', { name: 'preference text_to_image' })
}
const tier = (el: HTMLElement, name: string) =>
  el.querySelector<HTMLElement>(`.tier[data-tier="${name}"]`)!
const pick = (el: HTMLElement, name: string) =>
  el.querySelector<HTMLElement>(`.rungs [data-tier="${name}"]`)!

it('lists a shipped key, its three tiers and the default tier', async () => {
  render(PreferencesPage)
  const el = await card()
  expect(api.getPreferences).toHaveBeenCalledWith({ resolve: true })
  expect(within(el).getByText('shipped')).toBeTruthy()
  expect(
    [...el.querySelectorAll('div.tier')].map((t) =>
      t.getAttribute('data-tier'),
    ),
  ).toEqual(['draft', 'standard', 'best'])
  expect(tier(el, 'best').textContent).toContain('models/minimax-h3-image')
  expect(tier(el, 'best').textContent).toContain('cuda only')
  expect(pick(el, 'best').textContent).toContain('models/qwen-image-2.1')
  expect(screen.getByLabelText<HTMLSelectElement>('Default tier').value).toBe(
    'standard',
  )
  // a shipped key has nothing to reset to
  expect(
    within(el).queryByRole('button', { name: 'Reset to shipped' }),
  ).toBeNull()
  // nothing edited yet: nothing to save
  expect(
    within(el).getByRole<HTMLButtonElement>('button', { name: 'Save' })
      .disabled,
  ).toBe(true)
})

it('saves a reordered tier with the etag and shows the picks it answers', async () => {
  api.setPreference.mockResolvedValue({
    key: 'text_to_image',
    source: 'server',
    etag: '"k2"',
    ladder: { ...LADDER, best: ['models/qwen-image-2.1', CUDA] },
    resolutions: {
      ...RESOLUTIONS,
      best: { target: 'models/qwen-image-2.1', skipped: [], notes: ['moved'] },
    },
  })
  render(PreferencesPage)
  const el = await card()
  await fireEvent.click(
    within(tier(el, 'best')).getByRole('button', {
      name: 'Move models/qwen-image-2.1 up',
    }),
  )
  await fireEvent.click(within(el).getByRole('button', { name: 'Save' }))
  await waitFor(() =>
    expect(api.setPreference).toHaveBeenCalledWith(
      'text_to_image',
      { ...LADDER, best: ['models/qwen-image-2.1', CUDA] },
      '"k1"',
    ),
  )
  // the answer's picks replace the read's: the skip is gone, the key is the server's
  await waitFor(() =>
    expect(pick(el, 'best').textContent).not.toContain('skipped'),
  )
  expect(within(el).getByText('server')).toBeTruthy()
  expect(
    within(el).getByRole('button', { name: 'Reset to shipped' }),
  ).toBeTruthy()
})

it('asks Reload or Overwrite on a 412, and Overwrite sends the new etag', async () => {
  api.setPreference
    .mockRejectedValueOnce(
      new ApiError(
        "Preference 'text_to_image' changed since it was read",
        412,
        'stale_write',
        {
          code: 'stale_write',
          etag: '"theirs"',
        },
      ),
    )
    .mockResolvedValueOnce({
      key: 'text_to_image',
      source: 'server',
      etag: '"k3"',
      ladder: { ...LADDER, standard: ['models/z-image'] },
      resolutions: RESOLUTIONS,
    })
  render(PreferencesPage)
  const el = await card()
  await fireEvent.click(
    within(tier(el, 'standard')).getByRole('button', {
      name: 'Remove models/z-image-sdnq',
    }),
  )
  await fireEvent.click(within(el).getByRole('button', { name: 'Save' }))
  const notice = await within(el).findByRole('alert')
  expect(notice.textContent).toContain(
    'text_to_image changed since you opened it',
  )
  // the held-back save has finished: Overwrite is a fresh write
  await waitFor(() =>
    expect(
      within(el).getByRole<HTMLButtonElement>('button', { name: 'Save' })
        .disabled,
    ).toBe(false),
  )
  expect(within(notice).getByRole('button', { name: 'Reload' })).toBeTruthy()
  await fireEvent.click(
    within(notice).getByRole('button', { name: 'Overwrite' }),
  )
  await waitFor(() =>
    expect(api.setPreference).toHaveBeenLastCalledWith(
      'text_to_image',
      { ...LADDER, standard: ['models/z-image'] },
      '"theirs"',
    ),
  )
  await waitFor(() => expect(within(el).queryByRole('alert')).toBeNull())
})

it('Reload on a 412 drops the edit and reads the server copy again', async () => {
  api.setPreference.mockRejectedValueOnce(
    new ApiError('changed', 412, 'stale_write', { etag: '"theirs"' }),
  )
  render(PreferencesPage)
  const el = await card()
  await fireEvent.click(
    within(tier(el, 'standard')).getByRole('button', {
      name: 'Remove models/z-image-sdnq',
    }),
  )
  await fireEvent.click(within(el).getByRole('button', { name: 'Save' }))
  await fireEvent.click(
    await within(el).findByRole('button', { name: 'Reload' }),
  )
  await waitFor(() => expect(api.getPreferences).toHaveBeenCalledTimes(2))
  expect(tier(el, 'standard').textContent).toContain('models/z-image-sdnq')
})

it("shows the server's 422 refusal beside the key", async () => {
  api.setPreference.mockRejectedValueOnce(
    new ApiError(
      "Preference 'text_to_image' tier draft: target 'models/flux-dev' takes no argument 'seed'",
      422,
    ),
  )
  render(PreferencesPage)
  const el = await card()
  const field = within(tier(el, 'draft')).getByLabelText(
    'workflow to add to draft',
  )
  await fireEvent.input(field, { target: { value: 'models/flux-dev' } })
  await fireEvent.click(
    within(tier(el, 'draft')).getByRole('button', { name: 'Add' }),
  )
  expect(tier(el, 'draft').textContent).toContain('models/flux-dev')
  await fireEvent.click(within(el).getByRole('button', { name: 'Save' }))
  const refusal = await within(el).findByRole('alert')
  expect(refusal.textContent).toContain("Couldn't save text_to_image")
  expect(refusal.textContent).toContain("target 'models/flux-dev'")
})

it('Reset to shipped sends DELETE with the etag and shows the key as shipped', async () => {
  api.getPreferences.mockResolvedValue(read('server'))
  vi.mocked(confirmDialog).mockResolvedValueOnce(true)
  api.resetPreference.mockResolvedValue({
    key: 'text_to_image',
    removed: false,
    source: 'shipped',
    etag: '"k4"',
    ladder: LADDER,
    resolutions: RESOLUTIONS,
  })
  render(PreferencesPage)
  const el = await card()
  await fireEvent.click(
    within(el).getByRole('button', { name: 'Reset to shipped' }),
  )
  await waitFor(() =>
    expect(api.resetPreference).toHaveBeenCalledWith('text_to_image', '"k1"'),
  )
  await waitFor(() => expect(within(el).getByText('shipped')).toBeTruthy())
  expect(
    within(el).queryByRole('button', { name: 'Reset to shipped' }),
  ).toBeNull()
})

it('refuses an invalid new key before sending, and adds a valid one unsaved', async () => {
  render(PreferencesPage)
  await card()
  const field = screen.getByLabelText('New key')
  await fireEvent.input(field, { target: { value: '../etc' } })
  await fireEvent.click(screen.getByRole('button', { name: 'Add key' }))
  expect((await screen.findByRole('alert')).textContent).toMatch(
    /starts with a letter/,
  )
  await fireEvent.input(field, { target: { value: 'text_to_image' } })
  await fireEvent.click(screen.getByRole('button', { name: 'Add key' }))
  expect((await screen.findByRole('alert')).textContent).toMatch(/already/)
  await fireEvent.input(field, { target: { value: 'image_edit' } })
  await fireEvent.click(screen.getByRole('button', { name: 'Add key' }))
  const added = await screen.findByRole('region', {
    name: 'preference image_edit',
  })
  expect(within(added).getByText('not saved')).toBeTruthy()
  // an empty ladder is not sent: the server refuses an empty rung
  expect(
    within(added).getByRole<HTMLButtonElement>('button', { name: 'Save' })
      .disabled,
  ).toBe(true)
  expect(api.setPreference).not.toHaveBeenCalled()
})

it('changing the default tier writes it with its etag', async () => {
  api.setDefaultTier.mockResolvedValue({
    default_tier: 'best',
    etag: '"tier2"',
  })
  render(PreferencesPage)
  await card()
  const select = screen.getByLabelText<HTMLSelectElement>('Default tier')
  await fireEvent.change(select, { target: { value: 'best' } })
  await waitFor(() =>
    expect(api.setDefaultTier).toHaveBeenCalledWith('best', '"tier1"'),
  )
})
