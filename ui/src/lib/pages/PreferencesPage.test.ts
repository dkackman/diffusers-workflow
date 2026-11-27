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
import { feed } from '../test/feedMock'

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
vi.mock('../confirm.svelte', () => ({ confirmDialog: vi.fn() }))
vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))
// Another tab or an agent wrote: every watcher reads again
const elsewhere = () => {
  for (const watch of [...feed.watches]) void watch.fn()
}

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
    skipped: [{ workflow: 'models/minimax-h3-image', reason: 'cuda-only' }],
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
  api.checkPreference.mockResolvedValue({ tiers: {} })
})
afterEach(cleanup)

async function card() {
  return screen.findByRole('region', { name: 'preference text_to_image' })
}
async function edit(el: HTMLElement) {
  await fireEvent.click(within(el).getByRole('button', { name: 'Edit' }))
  return screen.getByRole('region', { name: 'preference text_to_image' })
}
const tier = (el: HTMLElement, name: string) =>
  el.querySelector<HTMLElement>(`.tier[data-tier="${name}"]`)!

it('lists a shipped key, its three tiers and the default tier', async () => {
  render(PreferencesPage)
  const el = await card()
  expect(api.getPreferences).toHaveBeenCalledWith({
    resolve: true,
    full: true,
  })
  expect(within(el).getByText('as shipped')).toBeTruthy()
  expect(
    [...el.querySelectorAll('div.tier')].map((t) =>
      t.getAttribute('data-tier'),
    ),
  ).toEqual(['draft', 'standard', 'best'])
  expect(tier(el, 'best').textContent).toContain('models/minimax-h3-image')
  expect(tier(el, 'best').textContent).toContain('needs CUDA')
  expect(tier(el, 'best').textContent).toContain('models/qwen-image-2.1')
  expect(
    screen
      .getByRole('group', { name: "Requests that don't name a tier run at" })
      .querySelector('[aria-pressed="true"]')?.textContent,
  ).toBe('Standard')
  // a shipped key has nothing to reset to
  expect(
    within(el).queryByRole('button', { name: 'Reset to shipped' }),
  ).toBeNull()
})

it('saves a reordered tier with the etag and shows the picks it answers', async () => {
  const saved = {
    source: 'server',
    etag: '"k2"',
    ladder: { ...LADDER, best: ['models/qwen-image-2.1', CUDA] },
    resolutions: {
      ...RESOLUTIONS,
      best: { target: 'models/qwen-image-2.1', skipped: [], notes: ['moved'] },
    },
  }
  api.setPreference.mockResolvedValue({ key: 'text_to_image', ...saved })
  render(PreferencesPage)
  await card()
  // the read after the save: what the server now holds
  api.getPreferences.mockResolvedValue({
    ...read(),
    preferences: { text_to_image: saved },
  })
  const el = await edit(await card())
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
  const after = await screen.findByText('changed on this server')
  expect(after).toBeTruthy()
  const shown = await card()
  expect(shown.textContent).not.toContain('needs CUDA')
  expect(
    within(shown).getByRole('button', { name: 'Reset to shipped' }),
  ).toBeTruthy()
  // and the page reads its preferences again
  await waitFor(() => expect(api.getPreferences).toHaveBeenCalledTimes(2))
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
  const el = await edit(await card())
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
  await waitFor(() =>
    expect(screen.queryByRole('button', { name: 'Save' })).toBeNull(),
  )
})

it('Reload on a 412 drops the edit and reads the server copy again', async () => {
  api.setPreference.mockRejectedValueOnce(
    new ApiError('changed', 412, 'stale_write', { etag: '"theirs"' }),
  )
  render(PreferencesPage)
  const el = await edit(await card())
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
  const el = await edit(await card())
  await fireEvent.click(
    within(tier(el, 'draft')).getByRole('button', {
      name: 'Add a model to draft',
    }),
  )
  const pop = await waitFor(() => {
    const found = document.querySelector<HTMLElement>('[data-popover-content]')
    expect(found).not.toBeNull()
    return found!
  })
  await fireEvent.input(
    within(pop).getByLabelText('workflow to add to draft'),
    {
      target: { value: 'models/flux-dev' },
    },
  )
  await fireEvent.click(within(pop).getByText('Add'))
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
  await waitFor(() => expect(screen.getByText('as shipped')).toBeTruthy())
  expect(screen.queryByRole('button', { name: 'Reset to shipped' })).toBeNull()
})

async function newPreference(name: string, first: string) {
  await fireEvent.click(screen.getByRole('button', { name: 'New preference' }))
  const form = screen.getByRole('form', { name: 'New preference' })
  await fireEvent.input(within(form).getByLabelText('Name'), {
    target: { value: name },
  })
  await fireEvent.input(
    within(form).getByLabelText('First model for the standard tier'),
    { target: { value: first } },
  )
  await fireEvent.click(within(form).getByRole('button', { name: 'Create' }))
  return form
}

it('New preference makes an unsaved card in the editor, its first model in the default tier', async () => {
  render(PreferencesPage)
  await card()
  await fireEvent.click(screen.getByRole('button', { name: 'New preference' }))
  expect(
    screen.getByText(
      /A workflow uses it by naming preference:<name> as a step's workflow path\./,
    ),
  ).toBeTruthy()
  await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  await newPreference('image_edit', 'qa/a')
  const added = await screen.findByRole('region', {
    name: 'preference image_edit',
  })
  expect(within(added).getByText('not saved')).toBeTruthy()
  expect(tier(added, 'standard').textContent).toContain('qa/a')
  expect(
    within(added).getByRole<HTMLButtonElement>('button', { name: 'Save' })
      .disabled,
  ).toBe(false)
  // one card in edit at a time: the others wait for this one
  const other = await card()
  const edit = within(other).getByRole<HTMLButtonElement>('button', {
    name: 'Edit',
  })
  expect(edit.disabled).toBe(true)
  expect(api.setPreference).not.toHaveBeenCalled()
  // Discard drops a key never saved
  await fireEvent.click(within(added).getByRole('button', { name: 'Discard' }))
  expect(
    screen.queryByRole('region', { name: 'preference image_edit' }),
  ).toBeNull()
  expect(edit.disabled).toBe(false)
})

it('one preference in edit: the open New preference form locks every Edit, and an open edit locks New', async () => {
  render(PreferencesPage)
  const el = await card()
  await fireEvent.click(screen.getByRole('button', { name: 'New preference' }))
  const edit = within(el).getByRole<HTMLButtonElement>('button', {
    name: 'Edit',
  })
  expect(edit.disabled).toBe(true)
  expect(edit.title).toBe('Save or discard the open edit first')
  // closing the form frees the card; an open edit then locks New
  await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  const again = within(await card()).getByRole<HTMLButtonElement>('button', {
    name: 'Edit',
  })
  expect(again.disabled).toBe(false)
  await fireEvent.click(again)
  expect(
    screen.getByRole<HTMLButtonElement>('button', { name: 'New preference' })
      .disabled,
  ).toBe(true)
  expect(screen.queryByRole('form', { name: 'New preference' })).toBeNull()
})

it('New preference refuses a name the server would, before sending', async () => {
  render(PreferencesPage)
  await card()
  const form = await newPreference('../etc', 'qa/a')
  expect(within(form).getByRole('alert').textContent).toMatch(
    /starts with a letter/,
  )
  await fireEvent.input(within(form).getByLabelText('Name'), {
    target: { value: 'text_to_image' },
  })
  await fireEvent.click(within(form).getByRole('button', { name: 'Create' }))
  expect(within(form).getByRole('alert').textContent).toMatch(/already/)
  expect(screen.queryByRole('region', { name: 'preference ../etc' })).toBeNull()
})

it('changing the default tier writes it with its etag', async () => {
  api.setDefaultTier.mockResolvedValue({
    default_tier: 'best',
    etag: '"tier2"',
  })
  render(PreferencesPage)
  await card()
  await fireEvent.click(screen.getByRole('button', { name: 'Best' }))
  await waitFor(() =>
    expect(api.setDefaultTier).toHaveBeenCalledWith('best', '"tier1"'),
  )
})

it('the edit lock releases when the edited key is removed elsewhere', async () => {
  const both = read()
  const mine = {
    source: 'server',
    etag: '"q1"',
    ladder: { best: ['models/z-image'] },
    resolutions: {},
  }
  api.getPreferences.mockResolvedValue({
    ...both,
    preferences: { ...both.preferences, qa: mine },
  })
  render(PreferencesPage)
  const qa = await screen.findByRole('region', { name: 'preference qa' })
  await fireEvent.click(within(qa).getByRole('button', { name: 'Edit' }))
  const other = () =>
    within(
      screen.getByRole('region', { name: 'preference text_to_image' }),
    ).getByRole<HTMLButtonElement>('button', { name: 'Edit' })
  expect(other().disabled).toBe(true)
  // reset to shipped in another tab: a key only the server had is gone
  api.getPreferences.mockResolvedValue(both)
  elsewhere()
  await waitFor(() =>
    expect(screen.queryByRole('region', { name: 'preference qa' })).toBeNull(),
  )
  await waitFor(() => expect(other().disabled).toBe(false))
  expect(
    screen.getByRole<HTMLButtonElement>('button', { name: 'New preference' })
      .disabled,
  ).toBe(false)
})

it('Discard after the key changed elsewhere reopens on what the server holds', async () => {
  render(PreferencesPage)
  const el = await edit(await card())
  await fireEvent.click(
    within(tier(el, 'standard')).getByRole('button', {
      name: 'Remove models/z-image-sdnq',
    }),
  )
  const theirs = read()
  theirs.preferences.text_to_image.etag = '"k2"'
  theirs.preferences.text_to_image.ladder = {
    ...LADDER,
    standard: ['models/flux-dev'],
  }
  api.getPreferences.mockResolvedValue(theirs)
  elsewhere()
  await waitFor(() => expect(api.getPreferences).toHaveBeenCalledTimes(2))
  // the new read reaches the card before Discard is pressed
  await new Promise((settle) => setTimeout(settle, 20))
  await fireEvent.click(within(el).getByRole('button', { name: 'Discard' }))
  const again = await edit(await card())
  expect(tier(again, 'standard').textContent).toContain('models/flux-dev')
  expect(tier(again, 'standard').textContent).not.toContain('models/z-image')
})
