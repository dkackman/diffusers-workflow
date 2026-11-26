// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/svelte'
import EditorPage from './EditorPage.svelte'
import { api } from '../test/apiMock'
import { route } from '../router.svelte'
import { ApiError } from '../apiErrors'
import { chooseDialog, confirmDialog } from '../confirm.svelte'
import { changeOnBlur } from '../test/changeOnBlur'
import { feed } from '../test/feedMock'
import { notify } from '../toast'
import { emptyStep } from '../editor'
import { formatMtime } from '../format'
import { quiet, useQuietClock } from '../test/quiet'

// EditorPage talks to the server on mount (pipeline/class/task catalogs,
// the workflow listing, the prompt library) purely to feed forms the flow
// view never touches - stubbed here so the view-switch test stays about
// the view switch, not the network.
// Monaco cannot boot in jsdom; the JSON view is read through this stub,
// which renders the value it was handed as plain text
useQuietClock()

vi.mock('../editor/JsonEditor.svelte', async () => ({
  default: (await import('./JsonEditorStub.svelte')).default,
}))

// The conflict question is answered by each test that meets one
vi.mock('../confirm.svelte', () => ({
  chooseDialog: vi.fn(),
  confirmDialog: vi.fn().mockResolvedValue(true),
}))

vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())
api.saveWorkflow.mockResolvedValue({ saved: {}, etag: '"e2"' })
api.submitJob.mockResolvedValue({ id: 'job-1' })
api.listPipelines.mockResolvedValue({ pipelines: [] })
api.listClasses.mockResolvedValue({ classes: [] })
api.listTasks.mockResolvedValue({
  commands: [],
  image_processors: [],
  video_processors: [],
})
api.listWorkflows.mockResolvedValue({
  workflows: [],
  libraries: [{ origin: 'workspace', root: 'workflows', writable: true }],
  shadowed: [],
})
api.getWorkflow.mockResolvedValue({
  definition: { id: 'ZImage', steps: [{ name: 'generate', pipeline: {} }] },
  origin: 'workspace',
  writable: true,
  etag: '"e1"',
})
api.listPrompts.mockResolvedValue({ prompts: [], details: {} })
api.listWorkspaces.mockResolvedValue({
  workspaces: [{ name: 'default' }, { name: 'studio' }],
  workspace_root: '/w',
})
api.validate.mockResolvedValue({
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
})

afterEach(() => {
  cleanup()
})

describe('EditorPage view switch', () => {
  it('defaults to the form view and shows/hides the flow diagram on toggle', async () => {
    render(EditorPage, { name: '' })

    // Form view is up by default (no saved view-mode preference in a
    // fresh jsdom localStorage) - the flow diagram region
    // is the flow view's fingerprint and should be absent.
    await waitFor(() =>
      expect(screen.getByLabelText('workflow id')).toBeTruthy(),
    )
    expect(
      screen.queryByRole('region', { name: 'workflow data-flow' }),
    ).toBeNull()

    const flowButton = screen.getByTitle(
      'read-only data-flow diagram: steps as boxes, previous_result as edges',
    )
    await flowButton.click()

    await waitFor(() =>
      expect(
        screen.queryByRole('region', { name: 'workflow data-flow' }),
      ).not.toBeNull(),
    )
    // Switching away hides it again - it isn't just always-rendered and
    // toggled with CSS.
    const formButton = screen.getByRole('button', { name: /^form$/i })
    await formButton.click()
    await waitFor(() =>
      expect(
        screen.queryByRole('region', { name: 'workflow data-flow' }),
      ).toBeNull(),
    )
  })
})

describe('EditorPage header', () => {
  it('sets the tab title to the open workflow, and its back link to its own page', async () => {
    render(EditorPage, { name: 'e2e-card' })
    await waitFor(() =>
      expect(document.title).toBe('e2e-card · diffusers-workflow'),
    )
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe(
      'e2e-card',
    )
    expect(screen.getByRole('link', { name: '← e2e-card' })).toHaveAttribute(
      'href',
      '#/ws/default/workflows/e2e-card',
    )
  })

  it('sets the tab title to New workflow, and its back link to Workflows', async () => {
    render(EditorPage, { name: '' })
    await waitFor(() =>
      expect(document.title).toBe('New workflow · diffusers-workflow'),
    )
    expect(screen.getByRole('link', { name: '← Workflows' })).toHaveAttribute(
      'href',
      '#/ws/default/workflows',
    )
  })
})

describe('EditorPage file bar', () => {
  it('EditorPage says when the workflow was last saved', async () => {
    const changed = 1_790_000_000
    vi.mocked(api.listWorkflows).mockResolvedValueOnce({
      workflows: ['e2e-card'],
      details: { 'e2e-card': { mtime: changed } },
      libraries: [{ origin: 'workspace', root: 'workflows', writable: true }],
      shadowed: [],
    } as never)
    render(EditorPage, { name: 'e2e-card' })
    await waitFor(() =>
      expect(document.querySelector('.stamp time')?.textContent?.trim()).toBe(
        `Changed ${formatMtime(changed)}`,
      ),
    )
  })

  it('EditorPage says its own save at once, from no change feed', async () => {
    const list = vi.mocked(api.listWorkflows)
    const original = list.getMockImplementation()!
    let changed = 1_790_000_000
    list.mockImplementation(
      async () =>
        ({
          workflows: ['e2e-card'],
          details: { 'e2e-card': { mtime: changed } },
          libraries: [
            { origin: 'workspace', root: 'workflows', writable: true },
          ],
          shadowed: [],
        }) as never,
    )
    try {
      render(EditorPage, { name: 'e2e-card' })
      const stamp = () => document.querySelector('.stamp time')?.textContent
      await waitFor(() =>
        expect(stamp()).toBe(`Changed ${formatMtime(changed)}`),
      )
      changed += 3600
      await fireEvent.click(screen.getByRole('button', { name: /^save/i }))
      // the feed never moves here: the save itself reads the time back
      await waitFor(() =>
        expect(stamp()).toBe(`Changed ${formatMtime(changed)}`),
      )
    } finally {
      list.mockImplementation(original)
    }
  })
})

describe('EditorPage primary action', () => {
  it('Run is the one primary button, and Save is secondary', async () => {
    render(EditorPage, { name: 'e2e-card' })
    await waitFor(() => expect(screen.getByRole('button', { name: 'Run' })))
    const primary = within(
      screen
        .getByRole('heading', { level: 1 })
        .closest('.pagehead') as HTMLElement,
    )
      .getAllByRole('button')
      .filter((b) => b.matches('.withicon:not(.quiet, .bare)'))
      .map((b) => b.textContent?.trim())
    expect(primary).toEqual(['Run'])
    expect(screen.getByRole('button', { name: /^Save/ })).toHaveClass('quiet')
  })
})

describe('EditorPage primary action in every state', () => {
  const primaryNow = () =>
    within(
      screen
        .getByRole('heading', { level: 1 })
        .closest('.pagehead') as HTMLElement,
    )
      .getAllByRole('button')
      .filter((b) => b.matches('.withicon:not(.quiet, .bare)'))
      .map((b) => b.textContent?.trim())

  it('the back link is the full path', async () => {
    render(EditorPage, { name: 'models/z-image' })
    await waitFor(() =>
      expect(
        screen.getByRole('link', { name: '← models/z-image' }),
      ).toBeTruthy(),
    )
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe(
      'models/z-image',
    )
  })

  it('Run is the primary button when dirty, on an example and while running', async () => {
    vi.mocked(api.getWorkflow).mockResolvedValueOnce({
      definition: { id: 'Ex', steps: [{ name: 'generate', pipeline: {} }] },
      origin: 'builtin',
      writable: false,
      etag: '"e1"',
    })
    render(EditorPage, { name: 'models/z-image' })
    await waitFor(() =>
      expect(screen.getByText(/This is an example/)).toBeTruthy(),
    )
    expect(primaryNow()).toEqual(['Run'])
    expect(screen.getByRole('button', { name: /^Save/ })).toHaveClass('quiet')
    cleanup()

    render(EditorPage, { name: 'e2e-card' })
    await waitFor(() => expect(screen.getByRole('button', { name: 'Run' })))
    await fireEvent.input(screen.getByLabelText('workflow id'), {
      target: { value: 'Changed' },
    })
    expect(screen.getByText('unsaved')).toBeTruthy()
    expect(primaryNow()).toEqual(['Run'])
    vi.mocked(api.submitJob).mockReturnValueOnce(new Promise(() => {}))
    await screen.getByRole('button', { name: 'Run' }).click()
    await waitFor(() =>
      expect(
        (screen.getByRole('button', { name: /^Save/ }) as HTMLButtonElement)
          .disabled,
      ).toBe(true),
    )
    expect(primaryNow()).toEqual(['Run'])
    expect(screen.getByRole('button', { name: /^Save/ })).toHaveClass('quiet')
  })
})

describe('EditorPage validation plan', () => {
  it('validates the definition it opened, without the transport metadata', async () => {
    render(EditorPage, { name: 'models/z-image' })
    // Validate waits for the document: until it arrives there is none
    await waitFor(() =>
      expect(
        (screen.getByLabelText('workflow id') as HTMLInputElement).value,
      ).toBe('ZImage'),
    )
    vi.mocked(api.validate).mockClear()
    await screen.getByRole('button', { name: /validate/i }).click()
    await waitFor(() => expect(api.validate).toHaveBeenCalled())
    // the press hands over the definition; the live check wraps it
    for (const [asked] of vi.mocked(api.validate).mock.calls) {
      const payload = ('workflow' in asked ? asked.workflow : asked) as object
      // origin and writable are how the fetch says where a file came from -
      // they are not the file's, and the schema refuses unknown root keys
      expect(payload).not.toHaveProperty('origin')
      expect(payload).not.toHaveProperty('writable')
    }
  })

  it('a change is checked without pressing Validate', async () => {
    vi.mocked(api.validate).mockClear()
    render(EditorPage, { name: 'models/z-image' })
    const id = () => screen.getByLabelText('workflow id') as HTMLInputElement
    await waitFor(() => expect(id().value).toBe('ZImage'))
    // the file just read is checked at once, without asking the hub
    await waitFor(() => expect(api.validate).toHaveBeenCalledTimes(1))
    expect(vi.mocked(api.validate).mock.calls[0][0]).toMatchObject({
      workflow: { id: 'ZImage' },
      sizes: false,
    })
    await waitFor(() => expect(screen.getByText('Ready to run.')).toBeTruthy())
    // a burst of typing is one check, of what was typed last
    await fireEvent.input(id(), { target: { value: 'Mi' } })
    await fireEvent.input(id(), { target: { value: 'Mine' } })
    expect(api.validate).toHaveBeenCalledTimes(1)
    // the debounce is real time: wait well past it, not up to waitFor's 1 s
    await waitFor(() => expect(api.validate).toHaveBeenCalledTimes(2), {
      timeout: 5000,
    })
    expect(vi.mocked(api.validate).mock.calls[1][0]).toMatchObject({
      workflow: { id: 'Mine' },
      sizes: false,
    })
  })

  it('a document still being read is not checked', async () => {
    vi.mocked(api.validate).mockClear()
    let arrive!: (read: unknown) => void
    vi.mocked(api.getWorkflow).mockReturnValueOnce(
      new Promise((resolve) => (arrive = resolve)) as never,
    )
    render(EditorPage, { name: 'models/z-image' })
    await quiet()
    expect(api.validate).not.toHaveBeenCalled()
    arrive({
      definition: { id: 'Late', steps: [{ name: 'generate', pipeline: {} }] },
      origin: 'workspace',
      writable: true,
      etag: '"e1"',
    })
    await waitFor(() => expect(api.validate).toHaveBeenCalledTimes(1), {
      timeout: 5000,
    })
  })

  it('a new draft is checked without pressing Validate', async () => {
    vi.mocked(api.validate).mockClear()
    render(EditorPage, { name: '' })
    const id = () => screen.getByLabelText('workflow id') as HTMLInputElement
    await waitFor(() => expect(id()).toBeTruthy())
    await fireEvent.input(id(), { target: { value: 'Drafted' } })
    await waitFor(
      () =>
        expect(vi.mocked(api.validate).mock.calls.at(-1)?.[0]).toMatchObject({
          workflow: { id: 'Drafted' },
        }),
      { timeout: 5000 },
    )
  }, 10_000)

  it('a blank new draft is not checked until its first edit', async () => {
    vi.mocked(api.validate).mockClear()
    render(EditorPage, { name: '' })
    const id = () => screen.getByLabelText('workflow id') as HTMLInputElement
    await waitFor(() => expect(id()).toBeTruthy())
    await quiet()
    expect(api.validate).not.toHaveBeenCalled()
    await fireEvent.click(screen.getByRole('button', { name: /Validate/ }))
    await waitFor(() => expect(api.validate).toHaveBeenCalledTimes(1))
  })

  it('a live check that fails says so', async () => {
    vi.mocked(api.validate).mockRejectedValueOnce(new Error('server down'))
    render(EditorPage, { name: 'models/z-image' })
    await waitFor(() =>
      expect(screen.getByText("Couldn't check this workflow.")).toBeTruthy(),
    )
    expect(screen.getByRole('alert')).toBeTruthy()
  })

  it('dismissing a failed live check closes it', async () => {
    vi.mocked(api.validate).mockRejectedValueOnce(new Error('server down'))
    render(EditorPage, { name: 'models/z-image' })
    await waitFor(() =>
      expect(screen.getByText("Couldn't check this workflow.")).toBeTruthy(),
    )
    await fireEvent.click(
      screen.getByRole('button', { name: 'dismiss validation results' }),
    )
    await waitFor(() =>
      expect(screen.queryByText("Couldn't check this workflow.")).toBeNull(),
    )
  })

  /** An opened workflow whose check names a step and a variable */
  async function openRefused() {
    vi.mocked(api.getWorkflow).mockResolvedValueOnce({
      definition: {
        id: 'ZImage',
        variables: { prompt: 'a cat' },
        // a whole step: Go to opens it full, which shows every field
        steps: [emptyStep()],
      },
      origin: 'workspace',
      writable: true,
      etag: '"e1"',
    } as never)
    vi.mocked(api.validate).mockResolvedValueOnce({
      valid: false,
      error: null,
      errors: [
        { path: 'steps[0].pipeline', message: 'no pipeline' },
        { path: 'variables.prompt', message: 'not a number' },
      ],
      warnings: [],
      plan: null,
    } as never)
    // its own name: the step mode a Go to stores is this file's alone
    render(EditorPage, { name: 'models/refused' })
  }

  it('Go to opens the step and scrolls to it', async () => {
    const scroll = vi.fn()
    Element.prototype.scrollIntoView = scroll
    try {
      await openRefused()
      await fireEvent.click(
        await screen.findByRole('button', { name: 'Go to step generate' }),
      )
      const row = document.getElementById('step-generate')!
      await waitFor(() => expect(scroll.mock.contexts).toContain(row))
      const full = within(row).getByRole('button', { name: 'full' })
      expect(full.getAttribute('aria-pressed')).toBe('true')
      // keyboard focus lands in the step, not left on the Go to button
      await waitFor(() =>
        expect(row.contains(document.activeElement)).toBe(true),
      )
    } finally {
      delete (Element.prototype as Partial<Element>).scrollIntoView
    }
  })

  it('Go to a variable focuses its field', async () => {
    await openRefused()
    await fireEvent.click(
      await screen.findByRole('button', { name: 'Go to variable prompt' }),
    )
    await waitFor(() => expect(document.activeElement?.id).toBe('wfvar-prompt'))
  })

  it('shows what a job will do under a valid verdict', async () => {
    render(EditorPage, { name: '' })
    await waitFor(() =>
      expect(screen.getByLabelText('workflow id')).toBeTruthy(),
    )

    await screen.getByRole('button', { name: /validate/i }).click()

    await waitFor(() =>
      expect(screen.getByText('What a job will do')).toBeTruthy(),
    )
    const plan = screen.getByText('What a job will do').closest('details')!
    expect(plan.textContent).toContain('3 steps (shots: 2)')
    expect(plan.textContent).toContain('about 12 min on cuda')
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

describe('EditorPage new workflow origin', () => {
  // one example beside one of the workspace's own
  const withExample = {
    workflows: ['templates/tti', 'mine'],
    details: {
      'templates/tti': { writable: false, origin: 'examples' },
      mine: { writable: true, origin: 'workspace' },
    },
    libraries: [{ origin: 'workspace', root: 'workflows', writable: true }],
    shadowed: [],
  }
  const startFrom = () =>
    screen.getByLabelText('start from') as HTMLSelectElement
  const fileName = () => screen.getByLabelText('file name') as HTMLInputElement

  afterEach(() => {
    sessionStorage.clear()
    location.hash = ''
  })

  it('offers blank or an example, and starting from one loads it and names it', async () => {
    vi.mocked(api.listWorkflows).mockResolvedValue(withExample as never)
    render(EditorPage, { name: '' })

    await waitFor(() =>
      expect([...startFrom().options].map((option) => option.value)).toEqual([
        '',
        'templates/tti',
      ]),
    )
    expect(startFrom().value).toBe('')
    await fireEvent.change(startFrom(), { target: { value: 'templates/tti' } })

    await waitFor(() => expect(startFrom().value).toBe('templates/tti'))
    expect(api.getWorkflow).toHaveBeenCalledWith('templates/tti')
    expect(fileName().value).toBe('tti')
    expect(
      (screen.getByLabelText('workflow id') as HTMLInputElement).value,
    ).toBe('ZImage')
  })

  it('the example banner goes after Save', async () => {
    vi.mocked(api.listWorkflows).mockResolvedValue(withExample as never)
    render(EditorPage, { name: '' })
    await waitFor(() => expect(startFrom().options.length).toBe(2))
    await fireEvent.change(startFrom(), { target: { value: 'templates/tti' } })
    await waitFor(() => expect(startFrom().value).toBe('templates/tti'))
    await fireEvent.click(screen.getByRole('button', { name: /^Save/ }))
    await waitFor(() => expect(api.saveWorkflow).toHaveBeenCalled())
    await waitFor(() =>
      expect(screen.queryByLabelText('start from')).toBeNull(),
    )
  })

  it('a draft started from an example is checked at once', async () => {
    vi.mocked(api.listWorkflows).mockResolvedValue(withExample as never)
    render(EditorPage, { name: '' })
    await waitFor(() => expect(startFrom().options.length).toBe(2))
    await quiet()
    vi.mocked(api.validate).mockClear()
    await fireEvent.change(startFrom(), { target: { value: 'templates/tti' } })
    await waitFor(() => expect(api.validate).toHaveBeenCalled(), {
      timeout: 2000,
    })
  })

  it('Start from asks before discarding a JSON draft that does not parse', async () => {
    vi.mocked(api.listWorkflows).mockResolvedValue(withExample as never)
    vi.mocked(confirmDialog).mockClear()
    vi.mocked(confirmDialog).mockResolvedValueOnce(false)
    render(EditorPage, { name: '' })
    await waitFor(() => expect(startFrom().options.length).toBe(2))
    await fireEvent.click(screen.getByTitle('edit the raw JSON, schema-aware'))
    await fireEvent.input(await screen.findByTestId('json-input'), {
      target: { value: '{ not json' },
    })
    vi.mocked(api.getWorkflow).mockClear()

    await fireEvent.change(startFrom(), { target: { value: 'templates/tti' } })

    await waitFor(() => expect(confirmDialog).toHaveBeenCalledTimes(1))
    expect(vi.mocked(confirmDialog).mock.calls[0][1]).toMatchObject({
      danger: true,
    })
    // Cancel keeps the draft: nothing was fetched or replaced
    expect(api.getWorkflow).not.toHaveBeenCalled()
    expect(startFrom().value).toBe('')
  })

  it('takes up what a New from handed over', async () => {
    vi.mocked(api.listWorkflows).mockResolvedValue(withExample as never)
    sessionStorage.setItem(
      'dw-editor-import',
      JSON.stringify({
        id: 'Tti',
        steps: [{ name: 'generate', pipeline: {} }],
      }),
    )
    sessionStorage.setItem('dw-editor-import-from', 'templates/tti')
    sessionStorage.setItem('dw-editor-import-name', 'tti')
    render(EditorPage, { name: '' })

    await waitFor(() => expect(startFrom().value).toBe('templates/tti'))
    expect(fileName().value).toBe('tti')
  })

  it('moves the draft to another workspace without losing it', async () => {
    vi.mocked(api.listWorkflows).mockResolvedValue(withExample as never)
    render(EditorPage, { name: '' })
    const picker = () => screen.getByLabelText('workspace') as HTMLSelectElement
    await waitFor(() =>
      expect([...picker().options].map((option) => option.value)).toEqual([
        'default',
        'studio',
      ]),
    )
    await fireEvent.input(fileName(), { target: { value: 'Keep' } })

    await fireEvent.change(picker(), { target: { value: 'studio' } })

    await waitFor(() => expect(location.hash).toBe('#/ws/studio/edit'))
    await waitFor(() => expect(picker().value).toBe('studio'))
    expect(fileName().value).toBe('Keep')
  })

  it('moves unsaved edits without asking, and they stay unsaved', async () => {
    vi.mocked(api.listWorkflows).mockResolvedValue(withExample as never)
    location.hash = '#/ws/default/edit'
    // the router has heard it before anything is edited
    await waitFor(() =>
      expect(`#/${route.parts.join('/')}`).toBe('#/ws/default/edit'),
    )
    render(EditorPage, { name: '' })
    const picker = () => screen.getByLabelText('workspace') as HTMLSelectElement
    const id = () => screen.getByLabelText('workflow id') as HTMLInputElement
    await waitFor(() => expect(picker().options.length).toBe(2))
    await fireEvent.input(id(), { target: { value: 'Edited' } })
    expect(screen.getByText('unsaved')).toBeTruthy()
    vi.mocked(confirmDialog).mockClear()

    await fireEvent.change(picker(), { target: { value: 'studio' } })

    // the picker shows the choice at once: wait for the editor to take the draft
    await waitFor(() =>
      expect(Object.keys(sessionStorage)).toEqual(['dw-scrolls']),
    )
    expect(location.hash).toBe('#/ws/studio/edit')
    expect(confirmDialog).not.toHaveBeenCalled()
    expect(id().value).toBe('Edited')
    expect(screen.getByText('unsaved')).toBeTruthy()
  })
})

describe('EditorPage saving', () => {
  const fileName = () => screen.getByLabelText('file name') as HTMLInputElement
  const saveButton = () =>
    screen.getByRole('button', { name: /^save/i }) as HTMLButtonElement
  const notFound = () =>
    new ApiError('Workflow not found: models/missing', 404, undefined)
  const ctrl = (key: string) =>
    fireEvent.keyDown(window, { key, ctrlKey: true })

  /** A new workflow that validates, named `leaf`, ready to save. */
  async function newNamed(leaf: string) {
    render(EditorPage, { name: '' })
    await waitFor(() => expect(fileName()).toBeTruthy())
    await fireEvent.input(fileName(), { target: { value: leaf } })
  }

  afterEach(() => {
    vi.mocked(api.saveWorkflow).mockClear()
    vi.mocked(chooseDialog).mockReset()
    location.hash = ''
  })

  it('a failed load shows Retry and disables Save', async () => {
    vi.mocked(api.getWorkflow).mockRejectedValueOnce(
      new ApiError('server exploded', 500),
    )
    render(EditorPage, { name: 'models/z-image' })
    await screen.findByRole('button', { name: 'Retry' })
    expect(
      screen.getByText("Couldn't open models/z-image: The server hit an error"),
    ).toBeTruthy()
    // a 500 is not a missing file: there is nothing to start in its place
    expect(
      screen.queryByRole('button', { name: /start a new one/i }),
    ).toBeNull()
    expect(saveButton().disabled).toBe(true)
    expect(
      (screen.getByRole('button', { name: /^run/i }) as HTMLButtonElement)
        .disabled,
    ).toBe(true)
    expect(
      (screen.getByRole('button', { name: /^validate/i }) as HTMLButtonElement)
        .disabled,
    ).toBe(true)

    await fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(saveButton().disabled).toBe(false))
    expect(
      (screen.getByLabelText('workflow id') as HTMLInputElement).value,
    ).toBe('ZImage')
  })

  it('a 404 offers to start a new workflow with that name', async () => {
    vi.mocked(api.getWorkflow).mockRejectedValueOnce(notFound())
    render(EditorPage, { name: 'models/missing' })
    expect(
      await screen.findByText(
        "Couldn't open models/missing: That wasn't found",
      ),
    ).toBeTruthy()
    await fireEvent.click(
      screen.getByRole('button', {
        name: 'Start a new one named models/missing',
      }),
    )
    await waitFor(() => expect(saveButton().disabled).toBe(false))
    await fireEvent.click(saveButton())
    await waitFor(() => expect(api.saveWorkflow).toHaveBeenCalledTimes(1))
    const [path, , pre] = vi.mocked(api.saveWorkflow).mock.calls[0]
    expect(path).toBe('models/missing')
    // nothing is there yet, and a save must not land on a file made meanwhile
    expect(pre).toEqual({ createOnly: true })
  })

  it('Save after a load failure never calls saveWorkflow', async () => {
    vi.mocked(api.getWorkflow).mockRejectedValueOnce(
      new ApiError('timed out', 504),
    )
    render(EditorPage, { name: 'models/z-image' })
    await screen.findByRole('button', { name: 'Retry' })
    await ctrl('s')
    await ctrl('Enter')
    await fireEvent.click(saveButton())
    await quiet()
    expect(api.saveWorkflow).not.toHaveBeenCalled()
    expect(api.validate).not.toHaveBeenCalled()
  })

  it('a double Save writes once', async () => {
    render(EditorPage, { name: 'models/z-image' })
    await waitFor(() => expect(saveButton().disabled).toBe(false))
    // Both presses land before the page renders the first one's busy state
    saveButton().click()
    saveButton().click()
    await waitFor(() => expect(api.saveWorkflow).toHaveBeenCalled())
    await quiet()
    expect(api.saveWorkflow).toHaveBeenCalledTimes(1)
  })

  it('re-saves an opened workflow at the version it was read', async () => {
    render(EditorPage, { name: 'models/z-image' })
    await waitFor(() => expect(saveButton().disabled).toBe(false))
    await ctrl('s')
    await waitFor(() => expect(api.saveWorkflow).toHaveBeenCalledTimes(1))
    expect(vi.mocked(api.saveWorkflow).mock.calls[0][2]).toEqual({
      ifMatch: '"e1"',
    })
  })

  it('a new workflow saved under an existing name asks before replacing', async () => {
    vi.mocked(api.saveWorkflow).mockRejectedValueOnce(
      new ApiError('mine already exists', 409, 'exists'),
    )
    vi.mocked(chooseDialog).mockResolvedValueOnce('cancel')
    await newNamed('mine')
    await fireEvent.click(saveButton())

    await waitFor(() => expect(chooseDialog).toHaveBeenCalledTimes(1))
    expect(vi.mocked(chooseDialog).mock.calls[0][0]).toBe(
      'mine already exists. Replace it?',
    )
    expect(vi.mocked(api.saveWorkflow).mock.calls[0][2]).toEqual({
      createOnly: true,
    })
    // Cancel writes nothing more and leaves the edits unsaved
    expect(api.saveWorkflow).toHaveBeenCalledTimes(1)
    expect(location.hash).not.toContain('/edit/mine')
  })

  it('the URL moves to the saved name after the first save', async () => {
    location.hash = '#/ws/default/edit'
    await new Promise((r) => setTimeout(r, 0))
    const entries = history.length
    await newNamed('mine')
    await fireEvent.click(saveButton())

    await waitFor(() => expect(location.hash).toBe('#/ws/default/edit/mine'))
    // replaced in place: Back still leaves the editor, not the unsaved draft
    expect(history.length).toBe(entries)
    await waitFor(() =>
      expect(screen.getByRole('heading', { level: 1 }).textContent).toBe(
        'mine',
      ),
    )
  })

  it('a handed-over draft keeps its document after its first save', async () => {
    location.hash = '#/ws/default/edit'
    await new Promise((r) => setTimeout(r, 0))
    sessionStorage.setItem(
      'dw-editor-import',
      JSON.stringify({
        id: 'Tti',
        steps: [{ name: 'generate', pipeline: {} }],
      }),
    )
    sessionStorage.setItem('dw-editor-import-from', 'templates/tti')
    sessionStorage.setItem('dw-editor-import-name', 'tti')
    render(EditorPage, { name: '' })
    const id = () => screen.getByLabelText('workflow id') as HTMLInputElement
    await waitFor(() => expect(id().value).toBe('Tti'))
    await fireEvent.click(saveButton())

    await waitFor(() => expect(location.hash).toBe('#/ws/default/edit/tti'))
    // the save named the document; it did not start the draft again
    await waitFor(() =>
      expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('tti'),
    )
    expect(id().value).toBe('Tti')
    sessionStorage.clear()
  })

  it('a failure after a save is reported, not thrown', async () => {
    const failure = vi.spyOn(notify, 'failure')
    const broken = new Error('no history')
    // only the save's own move to the saved name fails; the router's
    // other bookkeeping goes through
    const replaceState = history.replaceState.bind(history)
    const replace = vi
      .spyOn(history, 'replaceState')
      .mockImplementation((state, unused, url) => {
        if (String(url).endsWith('/edit/mine')) throw broken
        replaceState(state, unused, url)
      })
    try {
      await newNamed('mine')
      await fireEvent.click(saveButton())
      await waitFor(() =>
        expect(failure).toHaveBeenCalledWith(
          'update the editor after saving',
          broken,
        ),
      )
      expect(api.saveWorkflow).toHaveBeenCalledTimes(1)
    } finally {
      replace.mockRestore()
      failure.mockRestore()
    }
  })

  it('Cmd+S saves text typed into the JSON view without a blur', async () => {
    await newNamed('mine')
    await fireEvent.click(screen.getByTitle('edit the raw JSON, schema-aware'))
    const typed = {
      id: 'typed',
      steps: [
        {
          name: 'generate',
          pipeline: {
            configuration: { component_type: 'FluxPipeline' },
            from_pretrained_arguments: { model_name: 'org/flux' },
          },
        },
      ],
    }
    await fireEvent.input(await screen.findByTestId('json-input'), {
      target: { value: JSON.stringify(typed) },
    })
    await fireEvent.keyDown(window, { key: 's', metaKey: true })

    await waitFor(() => expect(api.saveWorkflow).toHaveBeenCalledTimes(1))
    expect(vi.mocked(api.saveWorkflow).mock.calls[0][1]).toEqual(typed)
  })

  it('Cmd+S saves text typed into a form field that has not blurred', async () => {
    const uninstall = changeOnBlur()
    try {
      vi.mocked(api.getWorkflow).mockResolvedValueOnce({
        definition: {
          id: 'ZImage',
          variables: { line: 'Title card' },
          steps: [{ name: 'generate', pipeline: {} }],
        },
        origin: 'workspace',
        writable: true,
        etag: '"e1"',
      } as never)
      localStorage.removeItem('dw-editor-view')
      render(EditorPage, { name: 'models/z-image' })
      const field = (await screen.findByLabelText('line')) as HTMLInputElement
      await waitFor(() => expect(field.value).toBe('Title card'))
      field.focus()
      await fireEvent.input(field, { target: { value: 'Typed, not blurred' } })
      await fireEvent.keyDown(window, { key: 's', metaKey: true })

      await waitFor(() => expect(api.saveWorkflow).toHaveBeenCalledTimes(1))
      expect(vi.mocked(api.saveWorkflow).mock.calls[0][1]).toMatchObject({
        variables: { line: 'Typed, not blurred' },
      })
    } finally {
      uninstall()
    }
  })

  it('going back to a file after another failed to open shows the file', async () => {
    const { rerender } = render(EditorPage, { name: 'a' })
    await waitFor(() => expect(saveButton().disabled).toBe(false))
    vi.mocked(api.getWorkflow).mockRejectedValueOnce(
      new ApiError('server exploded', 500),
    )
    await rerender({ name: 'b' })
    await screen.findByText("Couldn't open b: The server hit an error")
    await rerender({ name: 'a' })
    await waitFor(() => expect(saveButton().disabled).toBe(false))
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
    expect(screen.queryByText(/Couldn't open/)).toBeNull()
  })

  it('going back to a file while another is still being read shows the file', async () => {
    const { rerender } = render(EditorPage, { name: 'a' })
    await waitFor(() => expect(saveButton().disabled).toBe(false))
    let finishB: (v: unknown) => void = () => {}
    vi.mocked(api.getWorkflow).mockImplementationOnce(
      () => new Promise((r) => (finishB = r)) as never,
    )
    await rerender({ name: 'b' })
    await rerender({ name: 'a' })
    await waitFor(() => expect(saveButton().disabled).toBe(false))
    // b's read lands late and must change nothing
    finishB({
      definition: { id: 'Other', steps: [{ name: 'generate', pipeline: {} }] },
      origin: 'workspace',
      writable: true,
      etag: '"e9"',
    })
    await quiet()
    expect(
      (screen.getByLabelText('workflow id') as HTMLInputElement).value,
    ).toBe('ZImage')
    expect(saveButton().disabled).toBe(false)
  })
})

describe('EditorPage run', () => {
  const runButton = () =>
    screen.getByRole('button', { name: /^run/i }) as HTMLButtonElement

  afterEach(() => {
    vi.mocked(api.submitJob).mockClear()
    location.hash = ''
  })

  it('a double click sends one request', async () => {
    render(EditorPage, { name: 'models/z-image' })
    await waitFor(() => expect(runButton().disabled).toBe(false))
    // Both clicks land before the page renders the first one's busy state
    runButton().click()
    runButton().click()
    await waitFor(() => expect(location.hash).toContain('/jobs/job-1'))
    expect(api.submitJob).toHaveBeenCalledTimes(1)
  })
})

describe('EditorPage remote change', () => {
  const idField = () => screen.getByLabelText('workflow id') as HTMLInputElement

  const verdict = () =>
    screen.queryByRole('button', { name: 'dismiss validation results' })
  const theirs = () =>
    vi.mocked(api.getWorkflow).mockResolvedValueOnce({
      definition: { id: 'Theirs', steps: [{ name: 'generate', pipeline: {} }] },
      origin: 'workspace',
      writable: true,
      etag: '"e9"',
    } as never)

  it('a reload from the server drops the verdict on the replaced copy', async () => {
    render(EditorPage, { name: 'models/z-image' })
    await waitFor(() => expect(idField().value).toBe('ZImage'))
    await fireEvent.click(screen.getByRole('button', { name: /^validate/i }))
    await waitFor(() => expect(verdict()).toBeTruthy())
    theirs()
    feed.move('default', 'workflows')
    await waitFor(() => expect(idField().value).toBe('Theirs'))
    expect(verdict()).toBeNull()
  })

  it('Load theirs drops the verdict on the edits it replaced', async () => {
    render(EditorPage, { name: 'models/z-image' })
    await waitFor(() => expect(idField().value).toBe('ZImage'))
    await fireEvent.input(idField(), { target: { value: 'Mine' } })
    await fireEvent.change(idField())
    await fireEvent.click(screen.getByRole('button', { name: /^validate/i }))
    await waitFor(() => expect(verdict()).toBeTruthy())
    theirs()
    feed.move('default', 'workflows')
    await fireEvent.click(
      await screen.findByRole('button', { name: 'Load theirs' }),
    )
    await waitFor(() => expect(idField().value).toBe('Theirs'))
    expect(verdict()).toBeNull()
  })

  it('Load theirs replaces the edits', async () => {
    render(EditorPage, { name: 'models/z-image' })
    await waitFor(() => expect(idField().value).toBe('ZImage'))
    await fireEvent.input(idField(), { target: { value: 'Mine' } })
    await fireEvent.change(idField())
    vi.mocked(api.getWorkflow).mockResolvedValueOnce({
      definition: { id: 'Theirs', steps: [{ name: 'generate', pipeline: {} }] },
      origin: 'workspace',
      writable: true,
      etag: '"e9"',
    } as never)
    feed.move('default', 'workflows')
    const banner = await screen.findByText(
      'models/z-image changed since you opened it, in another tab or by an agent',
    )
    expect(banner.closest('[role="status"]')).toBeTruthy()
    expect(idField().value).toBe('Mine')
    await fireEvent.click(screen.getByRole('button', { name: 'Load theirs' }))
    await waitFor(() => expect(idField().value).toBe('Theirs'))
    expect(screen.queryByRole('button', { name: 'Load theirs' })).toBeNull()
  })
})

describe('EditorPage keeps up with the server', () => {
  const rows = () =>
    Array.from(document.querySelectorAll<HTMLElement>('.steprow'))
  const stepNames = () => rows().map((row) => row.id.replace(/^step-/, ''))
  const collapsed = (name: string) =>
    !!document
      .getElementById(`step-${name}`)
      ?.querySelector('button[aria-label="expand this step"]')
  const serve = (etag: string, ...names: string[]) =>
    vi.mocked(api.getWorkflow).mockResolvedValueOnce({
      definition: { id: 'M', steps: names.map((name) => ({ name, task: {} })) },
      origin: 'workspace',
      writable: true,
      etag,
    } as never)

  it('a failed workflow listing says so', async () => {
    vi.mocked(api.listWorkflows).mockRejectedValueOnce(
      new ApiError('listing exploded', 500),
    )
    render(EditorPage, { name: '' })
    const line = await screen.findByText(/^Couldn't list the workflows/)
    expect(line.closest('[role="alert"]')).toBeTruthy()
  })

  it("a remote reload keeps the steps' modes by name", async () => {
    serve('"m1"', 'generate', 'upscale')
    render(EditorPage, { name: 'modes/kept' })
    await waitFor(() => expect(stepNames()).toEqual(['generate', 'upscale']))
    await fireEvent.click(
      document
        .getElementById('step-upscale')!
        .querySelector<HTMLButtonElement>(
          'button[aria-label="collapse this step"]',
        )!,
    )
    await waitFor(() => expect(collapsed('upscale')).toBe(true))
    // The agent reorders the steps and adds one: each keeps its own mode
    serve('"m2"', 'upscale', 'generate', 'caption')
    feed.move('default', 'workflows')
    await waitFor(() =>
      expect(stepNames()).toEqual(['upscale', 'generate', 'caption']),
    )
    expect(collapsed('upscale')).toBe(true)
    expect(collapsed('generate')).toBe(false)
    expect(collapsed('caption')).toBe(false)
    // A step that left and came back opens as a new one
    serve('"m3"', 'generate')
    feed.move('default', 'workflows')
    await waitFor(() => expect(stepNames()).toEqual(['generate']))
    serve('"m4"', 'generate', 'upscale')
    feed.move('default', 'workflows')
    await waitFor(() => expect(stepNames()).toEqual(['generate', 'upscale']))
    expect(collapsed('upscale')).toBe(false)
  })
})

describe('EditorPage undo', () => {
  const stepNames = () =>
    Array.from(document.querySelectorAll('.steprow')).map((row) =>
      row.id.replace(/^step-/, ''),
    )

  it('an Undo pressed after switching workflow leaves the other one alone', async () => {
    const undo = vi.spyOn(notify, 'undo')
    const error = vi.spyOn(notify, 'error')
    try {
      vi.mocked(api.getWorkflow).mockResolvedValueOnce({
        definition: {
          id: 'A',
          steps: [
            { name: 'generate', pipeline: {} },
            { name: 'upscale', pipeline: {} },
          ],
        },
        origin: 'workspace',
        writable: true,
        etag: '"ea"',
      } as never)
      const { rerender } = render(EditorPage, { name: 'a' })
      await waitFor(() => expect(stepNames()).toEqual(['generate', 'upscale']))
      await fireEvent.click(
        screen.getAllByRole('button', { name: 'remove step' })[1],
      )
      await waitFor(() => expect(undo).toHaveBeenCalledTimes(1))
      await rerender({ name: 'b' })
      await waitFor(() =>
        expect(
          (screen.getByLabelText('workflow id') as HTMLInputElement).value,
        ).toBe('ZImage'),
      )
      undo.mock.calls[0][1]()
      await waitFor(() =>
        expect(error).toHaveBeenCalledWith(
          'Nothing undone: another workflow, or a newer copy of this one, has opened since.',
        ),
      )
      expect(stepNames()).toEqual(['generate'])
    } finally {
      undo.mockRestore()
      error.mockRestore()
    }
  })
})
