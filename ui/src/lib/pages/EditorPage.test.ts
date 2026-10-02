// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import EditorPage from './EditorPage.svelte'
import { api } from '../api'

// EditorPage talks to the server on mount (pipeline/class/task catalogs,
// the workflow listing, the prompt library) purely to feed forms the flow
// view never touches - stubbed here so the view-switch test stays about
// the view switch, not the network.
// Monaco cannot boot in jsdom; the JSON view is read through this stub,
// which renders the value it was handed as plain text
vi.mock('../editor/JsonEditor.svelte', async () => ({
  default: (await import('./JsonEditorStub.svelte')).default,
}))

vi.mock('../api', () => ({
  api: {
    listPipelines: vi.fn().mockResolvedValue({ pipelines: [] }),
    listClasses: vi.fn().mockResolvedValue({ classes: [] }),
    listTasks: vi.fn().mockResolvedValue({
      commands: [],
      image_processors: [],
      video_processors: [],
    }),
    listWorkflows: vi.fn().mockResolvedValue({
      workflows: [],
      libraries: [{ origin: 'workspace', root: 'workflows', writable: true }],
      shadowed: [],
    }),
    getWorkflow: vi.fn().mockResolvedValue({
      definition: { id: 'ZImage', steps: [{ name: 'generate', pipeline: {} }] },
      origin: 'workspace',
      writable: true,
    }),
    listPrompts: vi.fn().mockResolvedValue({ prompts: [], details: {} }),
    validate: vi.fn().mockResolvedValue({
      valid: true,
      error: null,
      errors: [],
      warnings: [],
      plan: {
        fingerprint: 'sha256:abc',
        steps: 3,
        list_entries: { shots: 2 },
        cached_steps: 0,
        downloads_required: [{ repo: 'org/model', gb: 41.2 }],
        estimate: {
          minutes: 12,
          basis: 'catalog',
          device: 'cuda',
          measured_on: 'card',
          partial: false,
        },
      },
    }),
  },
}))

afterEach(() => {
  cleanup()
})

describe('EditorPage view switch', () => {
  it('defaults to the form view and shows/hides the flow diagram on toggle', async () => {
    render(EditorPage, { name: '' })

    // Form view is up by default (no saved view-mode preference in a
    // fresh jsdom localStorage) - the flow diagram's read-only hint text
    // is the flow view's fingerprint and should be absent.
    await waitFor(() =>
      expect(screen.getByLabelText('workflow id')).toBeTruthy(),
    )
    expect(screen.queryByText(/Read-only data-flow view/)).toBeNull()

    const flowButton = screen.getByTitle(
      'read-only data-flow diagram: steps as boxes, previous_result as edges',
    )
    await flowButton.click()

    await waitFor(() =>
      expect(screen.queryByText(/Read-only data-flow view/)).not.toBeNull(),
    )
    // Switching away hides it again - it isn't just always-rendered and
    // toggled with CSS.
    const formButton = screen.getByRole('button', { name: /^form$/i })
    await formButton.click()
    await waitFor(() =>
      expect(screen.queryByText(/Read-only data-flow view/)).toBeNull(),
    )
  })
})

describe('EditorPage validation plan', () => {
  it('validates the definition it opened, without the transport metadata', async () => {
    render(EditorPage, { name: 'models/z-image' })
    await waitFor(() =>
      expect(screen.getByLabelText('workflow id')).toBeTruthy(),
    )
    await screen.getByRole('button', { name: /validate/i }).click()
    await waitFor(() =>
      expect(vi.mocked(api.validate).mock.calls).toHaveLength(1),
    )
    const payload = vi.mocked(api.validate).mock.calls[0][0] as Record<
      string,
      unknown
    >
    // origin and writable are how the fetch says where a file came from -
    // they are not the file's, and the schema refuses unknown root keys
    expect(payload).not.toHaveProperty('origin')
    expect(payload).not.toHaveProperty('writable')
  })

  it('shows what a run will do under a valid verdict', async () => {
    render(EditorPage, { name: '' })
    await waitFor(() =>
      expect(screen.getByLabelText('workflow id')).toBeTruthy(),
    )

    await screen.getByRole('button', { name: /validate/i }).click()

    await waitFor(() =>
      expect(screen.getByLabelText('what a run will do')).toBeTruthy(),
    )
    const plan = screen.getByLabelText('what a run will do')
    expect(plan.textContent).toContain('3 steps (shots: 2)')
    expect(plan.textContent).toContain('~12 min on cuda')
    expect(plan.textContent).toContain('0 of 3 steps cached')
    expect(plan.textContent).toContain('needs download: org/model (41.2 GB)')
  })
})

describe('EditorPage step list', () => {
  /** The step names the form shows, top to bottom. */
  const formSteps = () =>
    [
      ...document.querySelectorAll<HTMLInputElement>('.panel.step input.name'),
    ].map((input) => input.value)
  /** The step's own panel, found by the name in its header. */
  const stepPanel = (name: string) =>
    [...document.querySelectorAll<HTMLElement>('.panel.step')].find(
      (panel) =>
        panel.querySelector<HTMLInputElement>('input.name')?.value === name,
    )!

  it('keeps the form and the JSON in step with an add, a move and a remove', async () => {
    try {
      localStorage.removeItem('dw-editor-view')
    } catch {
      /* no storage */
    }
    render(EditorPage, { name: '' })
    await waitFor(() => expect(formSteps()).toEqual(['generate']))

    await fireEvent.click(
      screen.getByTitle(
        'add a utility step - upscaling, segmentation, captioning, frame tools',
      ),
    )
    await fireEvent.click(
      screen.getByTitle('add a step that runs a diffusers pipeline'),
    )
    await waitFor(() =>
      expect(formSteps()).toEqual(['generate', 'process', 'generate-2']),
    )

    const moveUp = stepPanel('generate-2').querySelector<HTMLButtonElement>(
      'button[title="move up"]',
    )!
    await fireEvent.click(moveUp)
    await waitFor(() =>
      expect(formSteps()).toEqual(['generate', 'generate-2', 'process']),
    )

    const remove = stepPanel('generate').querySelector<HTMLButtonElement>(
      'button[title="remove step"]',
    )!
    await fireEvent.click(remove)
    await waitFor(() => expect(formSteps()).toEqual(['generate-2', 'process']))

    await fireEvent.click(screen.getByTitle('edit the raw JSON, schema-aware'))
    const json = JSON.parse(
      (await screen.findByTestId('json-editor')).textContent ?? '',
    )
    expect(json.steps.map((step: { name: string }) => step.name)).toEqual([
      'generate-2',
      'process',
    ])
  })
})
