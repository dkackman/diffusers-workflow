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
import { ApiError } from '../apiErrors'
import { chooseDialog, confirmDialog } from '../confirm.svelte'
import { changeOnBlur } from '../test/changeOnBlur'
import { feed } from '../test/feedMock'

// EditorPage talks to the server on mount (pipeline/class/task catalogs,
// the workflow listing, the prompt library) purely to feed forms the flow
// view never touches - stubbed here so the view-switch test stays about
// the view switch, not the network.
// Monaco cannot boot in jsdom; the JSON view is read through this stub,
// which renders the value it was handed as plain text
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

vi.mock('../api', () => ({
  api: {
    saveWorkflow: vi.fn().mockResolvedValue({ saved: {}, etag: '"e2"' }),
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
      etag: '"e1"',
    }),
    listPrompts: vi.fn().mockResolvedValue({ prompts: [], details: {} }),
    listWorkspaces: vi.fn().mockResolvedValue({
      workspaces: [{ name: 'default' }, { name: 'studio' }],
      workspace_root: '/w',
    }),
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

describe('EditorPage validation plan', () => {
  it('validates the definition it opened, without the transport metadata', async () => {
    render(EditorPage, { name: 'models/z-image' })
    // Validate waits for the document: until it arrives there is none
    await waitFor(() =>
      expect(
        (screen.getByLabelText('workflow id') as HTMLInputElement).value,
      ).toBe('ZImage'),
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
    await new Promise((r) => setTimeout(r, 0))
    expect(api.saveWorkflow).not.toHaveBeenCalled()
    expect(api.validate).not.toHaveBeenCalled()
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
    const crumbs = screen.getByRole('navigation', { name: 'breadcrumb' })
    await waitFor(() => expect(crumbs.textContent).toContain('mine'))
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
    await new Promise((r) => setTimeout(r, 0))
    expect(
      (screen.getByLabelText('workflow id') as HTMLInputElement).value,
    ).toBe('ZImage')
    expect(saveButton().disabled).toBe(false)
  })
})

describe('EditorPage remote change', () => {
  const idField = () => screen.getByLabelText('workflow id') as HTMLInputElement

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
