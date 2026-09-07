import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
// Hoisted above the imports so the static import of the component below -
// itself hoisted - sees an initialized mock
import WorkflowsPage from './WorkflowsPage.svelte'

type Detail = Record<string, unknown>

const listing = vi.hoisted(() => ({
  workflows: [] as string[],
  details: {} as Record<string, Detail>,
}))

const listWorkflows = vi.hoisted(() =>
  vi.fn(
    () =>
      new Promise((resolve) =>
        setTimeout(
          () =>
            resolve({
              workflow_dir: '/workspace/workflows',
              workflows: listing.workflows,
              details: listing.details,
            }),
          0,
        ),
      ),
  ),
)

vi.mock('../api', () => ({
  api: {
    listWorkflows: () => listWorkflows(),
    // WorkspacePicker mounts with the page; a listing that never lands
    // leaves it hidden, which is what a single-workspace server shows
    listWorkspaces: vi.fn(() => new Promise(() => {})),
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
  listing.workflows = ['templates/tti', 'templates/shot', 'models/flux-dev']
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

  await fireEvent.change(screen.getByLabelText('shape'), {
    target: { value: 'shot' },
  })

  await waitFor(() => expect(card('tti')).toBeNull())
  expect(card('shot')).toBeTruthy()
  expect(card('flux-dev')).toBeNull()
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

  expect(screen.getByText('~3 min · 22 GB (RTX 4090)')).toBeTruthy()
  // Under a minute reads as a bound, and an entry naming no accelerator
  // drops the parenthetical rather than showing an empty one
  expect(screen.getByText('<1 min · 0.5 GB')).toBeTruthy()
  // cost: null - the card says nothing rather than guessing
  expect(card('tti')?.textContent).not.toContain('min')
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
  expect(screen.getByText('~1.5 min · 8 GB')).toBeTruthy()
})

it('shows the summary rather than the whole description', async () => {
  await renderPage('tti')
  expect(card('tti')?.textContent).toContain('A still.')
  expect(card('tti')?.textContent).not.toContain('Second sentence')
})
