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
import { feed } from '../test/feedMock'
import '../router.svelte'

type Detail = Record<string, unknown>

const listing = vi.hoisted(() => ({
  workflows: [] as string[],
  details: {} as Record<string, Detail>,
  cost_basis: 'curated' as string,
}))

const listWorkflows = vi.hoisted(() =>
  vi.fn(
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
  ),
)

vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))
const gallery = vi.hoisted(() => vi.fn(() => Promise.resolve({ files: [] })))
vi.mock('../api', () => ({
  api: {
    listWorkflows: () => listWorkflows(),
    // The proof thumbnails load beside the listing. These tests are about
    // the catalog's text, so the gallery answers empty and every card
    // renders in its no-output form.
    gallery: () => gallery(),
    galleryThumbnailUrl: (name: string) => `/api/gallery/${name}/thumbnail`,
  },
}))

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
  listWorkflows.mockClear()
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
  expect(screen.getByText('~3 min · 22 GB VRAM')).toBeTruthy()
  // Under a minute reads as a bound rather than a rounded zero
  expect(screen.getByText('<1 min · 0.5 GB VRAM')).toBeTruthy()
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
  expect(screen.getByText('~3 min · 22 GB VRAM')).toBeTruthy()
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
  expect(screen.getByText('~1.5 min · 8 GB VRAM')).toBeTruthy()
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
  const lists = listWorkflows.mock.calls.length
  const proofs = gallery.mock.calls.length
  listing.workflows = [...listing.workflows, 'templates/new']
  listing.details = {
    ...listing.details,
    'templates/new': { kinds: ['image'], steps: 1, variables: 0 },
  }
  feed.move('default', 'workflows')
  expect(card('tti')).toBeTruthy()
  await waitFor(() => expect(card('new')).toBeTruthy())
  expect(listWorkflows.mock.calls.length).toBe(lists + 1)
  expect(gallery.mock.calls.length).toBe(proofs)
})

it('reloads only the proofs when the gallery moves', async () => {
  await renderPage('tti')
  await settle()
  const lists = listWorkflows.mock.calls.length
  const proofs = gallery.mock.calls.length
  feed.move('default', 'gallery')
  await settle()
  expect(gallery.mock.calls.length).toBe(proofs + 1)
  expect(listWorkflows.mock.calls.length).toBe(lists)
})

it("another workspace's move, or an unwatched kind, reloads nothing", async () => {
  await renderPage('tti')
  await settle()
  const lists = listWorkflows.mock.calls.length
  const proofs = gallery.mock.calls.length
  feed.move('studio', 'workflows')
  feed.move('studio', 'gallery')
  feed.move('default', 'jobs')
  await settle()
  expect(listWorkflows.mock.calls.length).toBe(lists)
  expect(gallery.mock.calls.length).toBe(proofs)
})

it('a quiet reload that fails keeps the cards under a refresh line', async () => {
  await renderPage('tti')
  listWorkflows.mockImplementationOnce(() => Promise.reject(new Error('boom')))
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
  listWorkflows.mockReturnValueOnce(new Promise(() => {}))
  render(WorkflowsPage)
  expect(screen.getByRole('status', { name: 'loading workflows' })).toBeTruthy()
  expect(screen.queryByText(/No workflows yet/)).toBeNull()
})

it('a failed first load offers Retry and Retry loads', async () => {
  listWorkflows.mockRejectedValueOnce(new Error('boom'))
  render(WorkflowsPage)
  await screen.findByText(/Couldn't load workflows/)
  await fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await waitFor(() => expect(card('tti')).toBeTruthy())
})
