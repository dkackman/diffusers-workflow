import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
// Hoisted above the imports so the static import of the component below -
// itself hoisted - sees an initialized mock
import WorkflowsPage from './WorkflowsPage.svelte'
import { api } from '../test/apiMock'
import { feed } from '../test/feedMock'
import '../router.svelte'

type Detail = Record<string, unknown>

const listing = vi.hoisted(() => ({
  workflows: [] as string[],
  details: {} as Record<string, Detail>,
  cost_basis: 'curated' as string,
}))

vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))
vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
api.listWorkflows.mockImplementation(
  () =>
    new Promise((resolve) =>
      setTimeout(
        () =>
          resolve({
            libraries: [
              {
                origin: 'workspace',
                root: '/workspace/workflows',
                writable: true,
              },
            ],
            shadowed: [],
            workflows: listing.workflows,
            details: listing.details,
            cost_basis: listing.cost_basis,
          }),
        0,
      ),
    ),
)
// The proof thumbnails load beside the listing. These tests are about
// the catalog's text, so the gallery answers empty and every card
// renders in its no-output form.
api.gallery.mockImplementation(() => Promise.resolve({ files: [] }))

/** The card for a workflow, found by the leaf name it is titled with -
 * the anchor's own accessible name is every badge and count as well. */
const card = (leaf: string) =>
  screen.queryByText(leaf, { selector: '.cardname' })?.closest('a') ?? null

async function renderPage(first: string) {
  render(WorkflowsPage)
  await waitFor(() => expect(card(first)).toBeTruthy())
}

beforeEach(() => {
  location.hash = '#/ws/default/workflows'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  listing.workflows = ['templates/tti', 'templates/shot', 'models/flux-dev']
  listing.cost_basis = 'curated'
  listing.details = {
    'templates/tti': {
      kinds: ['image'],
      steps: 1,
      variables: 2,
      description: 'A still. Second sentence.',
      summary: 'A still.',
      shape: 'image',
      traits: [],
      cost: null,
    },
    'templates/shot': {
      kinds: ['video'],
      steps: 2,
      variables: 3,
      description: 'One shot with speech.',
      summary: 'One shot with speech.',
      shape: 'shot',
      traits: ['has-audio', 'image-conditioned'],
      cost: [{ device: 'cuda', name: 'RTX 4090', vram_gb: 22, minutes: 3 }],
      cost_drivers: { num_frames: 121, images: ['a.png', 'b.png'] },
    },
    'models/flux-dev': {
      kinds: ['image'],
      steps: 1,
      variables: 2,
      description: 'Flux dev config.',
      summary: 'Flux dev config.',
      shape: 'image',
      traits: ['image-conditioned'],
      configures: 'templates/tti',
      cost: [{ device: 'mps', vram_gb: 0.5, minutes: 0.5 }],
    },
  }
})

afterEach(() => {
  // Without this each render's DOM stays behind and the next test's
  // queries find two of everything
  cleanup()
  api.listWorkflows.mockClear()
  localStorage.clear()
})

it('narrows the list to one shape', async () => {
  await renderPage('tti')
  expect(card('shot')).toBeTruthy()

  // Shape is a row of buttons rather than a select, so the vocabulary is
  // visible without opening anything
  const shapes = screen.getByRole('group', { name: 'filter by shape' })
  await fireEvent.click(within(shapes).getByRole('button', { name: 'shot' }))

  await waitFor(() => expect(card('tti')).toBeNull())
  expect(card('shot')).toBeTruthy()
  expect(card('flux-dev')).toBeNull()

  // and pressing the active one again clears the filter
  await fireEvent.click(within(shapes).getByRole('button', { name: 'shot' }))
  await waitFor(() => expect(card('tti')).toBeTruthy())
})

it('ANDs two traits together rather than widening', async () => {
  await renderPage('tti')

  // image-conditioned alone keeps both the shot and the model config
  await fireEvent.click(
    screen.getByRole('button', { name: 'image-conditioned' }),
  )
  await waitFor(() => expect(card('tti')).toBeNull())
  expect(card('shot')).toBeTruthy()
  expect(card('flux-dev')).toBeTruthy()

  // adding has-audio leaves only the workflow carrying both
  await fireEvent.click(screen.getByRole('button', { name: 'has-audio' }))
  await waitFor(() => expect(card('flux-dev')).toBeNull())
  expect(card('shot')).toBeTruthy()
})

it('shows the measured cost and omits it when nothing was measured', async () => {
  await renderPage('tti')

  // The accelerator it was measured on moves to the card's tooltip - the
  // caption has room for the numbers, not for the hardware
  expect(screen.getByText('about 3 min · 22 GB VRAM')).toBeTruthy()
  // Under a minute reads as a bound rather than a rounded zero
  expect(screen.getByText('under 1 min · 0.5 GB VRAM')).toBeTruthy()
  // cost: null - the card says nothing rather than guessing
  expect(card('tti')?.textContent).not.toContain('min')
})

it("says whose figure the cost is and the card's settings it was measured at", async () => {
  await renderPage('tti')
  // The "at" values are the card's cost_drivers - preset-applied - in map
  // order, a list as its entry count
  expect(card('shot')!.getAttribute('title')).toContain(
    "a maintainer's figure, measured on RTX 4090 at num_frames=121, images=2 entries",
  )
  // No drivers: no "at" clause; the device names the card when it has no name
  const flux = card('flux-dev')!.getAttribute('title')!
  expect(flux).toContain("a maintainer's figure, measured on mps")
  expect(flux).not.toContain(' at ')
  // A null cost has no cost line at all
  expect(card('tti')!.getAttribute('title')).not.toContain('measured on')
  // and the caption is unchanged
  expect(screen.getByText('about 3 min · 22 GB VRAM')).toBeTruthy()
})

it('prints a basis it does not know and a hostile driver value literally', async () => {
  listing.cost_basis = 'estimated'
  listing.details['templates/shot'] = {
    ...listing.details['templates/shot'],
    cost_drivers: { prompt: '<img src=x onerror=alert(1)>' },
  }
  await renderPage('tti')
  expect(card('shot')!.getAttribute('title')).toContain(
    'estimated, measured on RTX 4090 at prompt=<img src=x onerror=alert(1)>',
  )
  expect(document.querySelector('img[src="x"]')).toBeNull()
})

it('keeps a fractional measurement rather than rounding it away', async () => {
  listing.workflows = ['templates/half']
  listing.details = {
    'templates/half': {
      kinds: ['image'],
      steps: 1,
      variables: 1,
      description: 'Ninety seconds.',
      summary: 'Ninety seconds.',
      shape: 'image',
      traits: [],
      cost: [{ device: 'cuda', vram_gb: 8, minutes: 1.5 }],
    },
  }
  await renderPage('half')
  expect(screen.getByText('about 1.5 min · 8 GB VRAM')).toBeTruthy()
})

it('shows the summary rather than the whole description', async () => {
  await renderPage('tti')
  expect(card('tti')?.textContent).toContain('A still.')
  expect(card('tti')?.textContent).not.toContain('Second sentence')
})

it('the examples view lists only read-only workflows and links under shared', async () => {
  listing.details['models/flux-dev'] = {
    ...listing.details['models/flux-dev'],
    origin: 'examples',
    writable: false,
  }
  listing.details['templates/tti'] = {
    ...listing.details['templates/tti'],
    origin: 'workspace',
    writable: true,
  }
  render(WorkflowsPage, { examples: true })
  await waitFor(() => expect(card('flux-dev')).toBeTruthy())
  expect(card('tti')).toBeNull()
  expect(card('flux-dev')!.getAttribute('href')).toBe(
    '#/shared/examples/models/flux-dev',
  )
  expect(screen.queryByTitle('new workflow')).toBeNull()
})

it('the examples view counts and offers chips for only the read-only population', async () => {
  // One read-only (image), two writable (image, shot) - the count and the
  // shape vocabulary must reflect only the one the examples view lists
  listing.details['models/flux-dev'] = {
    ...listing.details['models/flux-dev'],
    origin: 'examples',
    writable: false,
  }
  render(WorkflowsPage, { examples: true })
  await waitFor(() => expect(card('flux-dev')).toBeTruthy())
  expect(screen.getByText('1 example', { selector: '.count' })).toBeTruthy()
  const shapes = screen.getByRole('group', { name: 'filter by shape' })
  expect(within(shapes).queryByRole('button', { name: 'shot' })).toBeNull()
  expect(within(shapes).getByRole('button', { name: 'image' })).toBeTruthy()
})

it('with no examples says how to add them', async () => {
  render(WorkflowsPage, { examples: true })
  expect(await screen.findByText('No examples on this server.')).toBeTruthy()
  expect(
    screen.getByText(
      'Start dw.serve with --examples-dir <folder> to list some.',
    ),
  ).toBeTruthy()
})

it('the workspace view leaves the examples out and links to them', async () => {
  listing.details['models/flux-dev'] = {
    ...listing.details['models/flux-dev'],
    origin: 'examples',
    writable: false,
  }
  render(WorkflowsPage)
  await waitFor(() => expect(card('tti')).toBeTruthy())
  expect(card('flux-dev')).toBeNull()
  const link = screen.getByTitle(/read-only templates/)
  expect(link.getAttribute('href')).toBe('#/shared/examples')
  expect(link.textContent).toContain('1')
})

it('the workspace view links a card under the current workspace', async () => {
  location.hash = '#/ws/studio/workflows'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  await renderPage('tti')
  expect(card('tti')!.getAttribute('href')).toBe(
    '#/ws/studio/workflows/templates/tti',
  )
})

const settle = () => new Promise((r) => setTimeout(r, 10))

it('reloads the listing when workflows move, keeping the cards on screen', async () => {
  await renderPage('tti')
  await settle()
  const lists = api.listWorkflows.mock.calls.length
  const proofs = api.gallery.mock.calls.length
  listing.workflows = [...listing.workflows, 'templates/new']
  listing.details = {
    ...listing.details,
    'templates/new': { kinds: ['image'], steps: 1, variables: 0 },
  }
  feed.move('default', 'workflows')
  expect(card('tti')).toBeTruthy()
  await waitFor(() => expect(card('new')).toBeTruthy())
  expect(api.listWorkflows.mock.calls.length).toBe(lists + 1)
  expect(api.gallery.mock.calls.length).toBe(proofs)
})

it('reloads only the proofs when the gallery moves', async () => {
  await renderPage('tti')
  await settle()
  const lists = api.listWorkflows.mock.calls.length
  const proofs = api.gallery.mock.calls.length
  feed.move('default', 'gallery')
  await settle()
  expect(api.gallery.mock.calls.length).toBe(proofs + 1)
  expect(api.listWorkflows.mock.calls.length).toBe(lists)
})

it("another workspace's move, or an unwatched kind, reloads nothing", async () => {
  await renderPage('tti')
  await settle()
  const lists = api.listWorkflows.mock.calls.length
  const proofs = api.gallery.mock.calls.length
  feed.move('studio', 'workflows')
  feed.move('studio', 'gallery')
  feed.move('default', 'jobs')
  await settle()
  expect(api.listWorkflows.mock.calls.length).toBe(lists)
  expect(api.gallery.mock.calls.length).toBe(proofs)
})

it('a quiet reload that fails keeps the cards under a refresh line', async () => {
  await renderPage('tti')
  api.listWorkflows.mockImplementationOnce(() =>
    Promise.reject(new Error('boom')),
  )
  feed.move('default', 'workflows')
  await settle()
  expect(card('tti')).toBeTruthy()
  expect(screen.getByText(/Couldn't refresh workflows/)).toBeTruthy()
})

it('filtered to nothing keeps the shape and trait chips, and the library path', async () => {
  await renderPage('tti')
  const shapes = screen.getByRole('group', { name: 'filter by shape' })
  // image has no workflow that has audio: the grid empties
  await fireEvent.click(within(shapes).getByRole('button', { name: 'image' }))
  await fireEvent.click(screen.getByRole('button', { name: 'has-audio' }))
  await screen.findByText('Nothing matches those filters.')
  expect(screen.getByRole('group', { name: 'filter by shape' })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'has-audio' })).toBeTruthy()
  expect(screen.getByText('/workspace/workflows')).toBeTruthy()
})

it('shows a skeleton while loading, not the empty state', async () => {
  api.listWorkflows.mockReturnValueOnce(new Promise(() => {}))
  render(WorkflowsPage)
  expect(screen.getByRole('status', { name: 'loading workflows' })).toBeTruthy()
  expect(screen.queryByText(/No workflows yet/)).toBeNull()
})

it('a failed first load offers Retry and Retry loads', async () => {
  api.listWorkflows.mockRejectedValueOnce(new Error('boom'))
  render(WorkflowsPage)
  await screen.findByText(/Couldn't load workflows/)
  await fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await waitFor(() => expect(card('tti')).toBeTruthy())
})

it("a card's proof is its tile, from the listing's workspace, inside the one link", async () => {
  api.gallery.mockImplementationOnce(() =>
    Promise.resolve({
      workspace: 'studio',
      files: [
        {
          name: 'templates/shot/r1/a.mp4',
          folder: 'templates/shot',
          subfolder: '',
          run_id: 'r1',
          version: 1,
          url: '/outputs/a.mp4',
          kind: 'video',
          size: 1,
          mtime: 1,
          label: 'a.mp4',
        },
      ],
    } as never),
  )
  await renderPage('shot')
  await waitFor(() =>
    expect(card('shot')?.querySelector('.filetile')).toBeTruthy(),
  )
  const tile = card('shot')!.querySelector('.filetile')!
  expect(tile.classList).toContain('tile-card')
  expect(tile.querySelector('.videobadge')).toBeTruthy()
  expect(tile.querySelector('img')?.getAttribute('src')).toContain(
    'workspace=studio',
  )
  expect(card('shot')!.querySelectorAll('a')).toHaveLength(0)
  expect(card('tti')?.querySelector('.filetile')).toBeNull()
})

it('the count reads n of m while searching', async () => {
  await renderPage('tti')
  expect(screen.getByText('3 workflows', { selector: '.count' })).toBeTruthy()
  await fireEvent.input(screen.getByPlaceholderText('search workflows…'), {
    target: { value: 'shot with speech' },
  })
  await waitFor(() =>
    expect(
      screen.getByText('1 of 3 workflows', { selector: '.count' }),
    ).toBeTruthy(),
  )
  expect(card('shot')).toBeTruthy()
  expect(card('tti')).toBeNull()
})

it('a card says what it needs', async () => {
  listing.details['templates/tti'] = {
    ...listing.details['templates/tti'],
    inputs: { init_image: 'image', voice: 'audio' },
  }
  await renderPage('tti')
  expect(card('tti')?.textContent).toContain('needs init_image, voice')
  expect(card('tti')?.textContent).not.toContain('vars')
})

it('a card with no inputs says nothing about variables', async () => {
  await renderPage('tti')
  expect(card('tti')?.textContent).not.toContain('needs')
  expect(card('tti')?.textContent).not.toContain('vars')
})

it('a second press on a shape clears it to all', async () => {
  await renderPage('tti')
  const shapes = screen.getByRole('group', { name: 'filter by shape' })
  await fireEvent.click(within(shapes).getByRole('button', { name: 'shot' }))
  await waitFor(() => expect(card('tti')).toBeNull())
  const pressed = within(shapes).getByRole('button', { name: 'shot' })
  expect(pressed.getAttribute('title')).toBe('press again to show every shape')
  await fireEvent.click(pressed)
  await waitFor(() => expect(card('tti')).toBeTruthy())
  expect(
    within(shapes)
      .getByRole('button', { name: 'all' })
      .getAttribute('aria-pressed'),
  ).toBe('true')
})

it("a search shows each card's folder, and clearing restores the groups", async () => {
  listing.workflows = [
    'templates/text-to-image',
    'templates/flux3/text-to-image',
    'templates/minimax-h3-image/text-to-image',
  ]
  listing.details = Object.fromEntries(
    listing.workflows.map((w) => [
      w,
      { kinds: ['image'], steps: 1, variables: 0, shape: 'image', traits: [] },
    ]),
  )
  render(WorkflowsPage)
  await waitFor(() =>
    expect(document.querySelectorAll('.cardname')).toHaveLength(3),
  )
  expect(document.querySelectorAll('.cardfolder')).toHaveLength(0)

  const box = screen.getByPlaceholderText('search workflows…')
  await fireEvent.input(box, { target: { value: 'text to image' } })
  await waitFor(() =>
    expect(document.querySelectorAll('.cardfolder')).toHaveLength(3),
  )
  const folders = [...document.querySelectorAll('.cardfolder')].map(
    (e) => e.textContent,
  )
  expect(folders.sort()).toEqual([
    'templates',
    'templates/flux3',
    'templates/minimax-h3-image',
  ])

  await fireEvent.input(box, { target: { value: '' } })
  await waitFor(() =>
    expect(document.querySelectorAll('.cardfolder')).toHaveLength(0),
  )
  expect(document.querySelectorAll('.cardname')).toHaveLength(3)
  for (const e of document.querySelectorAll('.cardname'))
    expect(e.textContent).toBe('text-to-image')
})

it('WorkflowsPage says when each workflow changed', async () => {
  const now = Math.floor(Date.now() / 1000)
  listing.details['templates/tti'].mtime = now - 5 * 60
  await renderPage('tti')
  const stamp = card('tti')!.querySelector('time')!
  expect(stamp.textContent?.trim()).toBe('changed 5 min ago')
  expect(stamp.getAttribute('title')).toBeTruthy()
  // a listing entry with no time says nothing rather than a made-up one
  expect(card('shot')!.querySelector('time')).toBeNull()
})

it('the newest-first sort', async () => {
  const now = Math.floor(Date.now() / 1000)
  listing.details['templates/tti'].mtime = now - 86_400
  listing.details['templates/shot'].mtime = now - 3600
  listing.details['models/flux-dev'].mtime = now - 60
  await renderPage('tti')
  const names = () =>
    [...document.querySelectorAll('.cardname')].map((n) => n.textContent)
  // by name, in folders, until asked
  expect(names()).toEqual(['shot', 'tti', 'flux-dev'])
  await fireEvent.click(screen.getByRole('button', { name: 'Newest' }))
  await waitFor(() => expect(names()).toEqual(['flux-dev', 'shot', 'tti']))
  expect(location.hash).toContain('sort=newest')
})

/** Each card's leaf name, in the order shown */
const cardNames = () =>
  [...document.querySelectorAll('.cardname')].map((n) => n.textContent)

it('a page opened with sort=newest in its link is newest first', async () => {
  const now = Math.floor(Date.now() / 1000)
  listing.details['templates/tti'].mtime = now - 60
  listing.details['templates/shot'].mtime = now - 3600
  location.hash = '#/ws/default/workflows?sort=newest'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  await renderPage('tti')
  // flux-dev has no time, so it goes last
  expect(cardNames()).toEqual(['tti', 'shot', 'flux-dev'])
})

it("Newest orders a search's matches too", async () => {
  const now = Math.floor(Date.now() / 1000)
  listing.details['templates/tti'].mtime = now - 60
  listing.details['templates/shot'].mtime = now - 3600
  listing.details['models/flux-dev'].mtime = now - 86_400
  await renderPage('tti')
  const box = screen.getByPlaceholderText('search workflows…')
  await fireEvent.input(box, { target: { value: 'image' } })
  // the search's own order, best match first
  await waitFor(() => expect(cardNames()).toEqual(['flux-dev', 'shot', 'tti']))
  await fireEvent.click(screen.getByRole('button', { name: 'Newest' }))
  await waitFor(() => expect(cardNames()).toEqual(['tti', 'shot', 'flux-dev']))
  // and Name gives the search's own order back
  await fireEvent.click(screen.getByRole('button', { name: 'Name' }))
  await waitFor(() => expect(cardNames()).toEqual(['flux-dev', 'shot', 'tti']))
})

it('the Name order is natural: take2 before take10', async () => {
  listing.workflows = ['shots/take10', 'shots/take2']
  listing.details = Object.fromEntries(
    listing.workflows.map((w) => [
      w,
      { kinds: ['image'], steps: 1, variables: 0, shape: 'image', traits: [] },
    ]),
  )
  await renderPage('take2')
  expect(cardNames()).toEqual(['take2', 'take10'])
})

it('drops a shape or trait in the link that is not one', async () => {
  location.hash =
    '#/ws/default/workflows?shape=sculpture&traits=has-audio,wibble'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  await renderPage('shot')
  // the unknown shape filters nothing; the known trait still does
  expect(card('tti')).toBeNull()
  location.hash = '#/ws/default/workflows?shape=sculpture&traits=wibble'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  await waitFor(() => expect(card('tti')).toBeTruthy())
  expect(card('shot')).toBeTruthy()
})
