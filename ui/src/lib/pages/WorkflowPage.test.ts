// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { tick } from 'svelte'
import WorkflowPage from './WorkflowPage.svelte'
import { api } from '../api'
import { notify } from '../toast'
import { connection } from '../connection.svelte'
import { ApiError, NetworkError } from '../apiErrors'
import type { ValidationResult } from '../types'
import { feed } from '../test/feedMock'
import { workspace } from '../workspace.svelte'
import { storageGet } from '../storage'

// Monaco cannot boot in jsdom, and this page only mounts it on "show JSON"
vi.mock('../editor/JsonEditor.svelte', async () => ({
  default: (await import('./JsonEditorStub.svelte')).default,
}))

vi.mock('../toast', () => ({
  notify: {
    error: vi.fn(),
    failure: vi.fn(),
    success: vi.fn(),
    dismiss: vi.fn(),
  },
}))

const valid: ValidationResult = {
  valid: true,
  error: null,
  errors: [],
  warnings: [],
  plan: {
    fingerprint: 'sha256:abc',
    steps: 1,
    list_entries: {},
    cached_steps: 0,
    downloads_required: [],
    estimate: {
      minutes: 2,
      basis: 'catalog',
      device: 'mps',
      measured_on: 'card',
      partial: false,
    },
  },
} as unknown as ValidationResult

vi.mock('../api', () => ({
  api: {
    getWorkflow: vi.fn(),
    listWorkflows: vi.fn().mockResolvedValue({
      workflows: [],
      details: {},
      libraries: [],
      shadowed: [],
    }),
    gallery: vi.fn().mockResolvedValue({ files: [] }),
    listPrompts: vi.fn().mockResolvedValue({ prompts: [], details: {} }),
    validate: vi.fn(),
    submitJob: vi.fn(),
    workflowDownloadUrl: vi.fn().mockReturnValue('/download'),
  },
}))

vi.mock('../changes.svelte', async () => ({
  changes: (await import('../test/feedMock')).feed.changes,
}))

const definition = {
  id: 'TextToImage',
  variables: { prompt: 'a cat', num_images_per_prompt: 1, seed: 0 },
  steps: [{ name: 'generate', pipeline: {} }],
}

/** Let pending promises and the renders they cause land, moving the fake
 * clock by `ms` first. */
async function advance(ms = 0) {
  await vi.advanceTimersByTimeAsync(ms)
  await tick()
}

const field = (name: string) =>
  document.getElementById(`var-${name}`) as HTMLTextAreaElement

async function type(name: string, value: string) {
  await fireEvent.input(field(name), { target: { value } })
}

const validateCalls = () => vi.mocked(api.validate).mock.calls

async function openPage() {
  render(WorkflowPage, { name: 'templates/text-to-image' })
  await advance()
  await advance()
}

beforeEach(() => {
  localStorage.clear()
  vi.useFakeTimers()
  vi.mocked(api.getWorkflow).mockResolvedValue({
    definition,
    origin: 'examples',
    writable: false,
  } as never)
  vi.mocked(api.validate).mockReset()
  vi.mocked(api.validate).mockResolvedValue(valid)
  vi.mocked(api.listWorkflows).mockResolvedValue({
    workflows: [],
    details: {},
    libraries: [],
    shadowed: [],
  } as never)
  vi.mocked(api.submitJob).mockReset()
  vi.mocked(notify.error).mockReset()
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

it('sets the tab title to the workflow', async () => {
  await openPage()
  expect(document.title).toBe('templates/text-to-image · diffusers-workflow')
})

it('the back link names Workflows here and Examples there', async () => {
  location.hash = '#/ws/default/workflows/templates%2Ftext-to-image'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  await openPage()
  expect(screen.getByRole('link', { name: '← Workflows' })).toBeTruthy()
  cleanup()
  location.hash = '#/shared/examples/templates%2Ftext-to-image'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  await openPage()
  expect(screen.getByRole('link', { name: '← Examples' })).toBeTruthy()
})

it('the inpaint form offers upload and browse for its image and mask', async () => {
  vi.mocked(api.getWorkflow).mockResolvedValue({
    definition: {
      id: 'Inpaint',
      variables: { prompt: 'a cat', image: '', mask: '' },
      steps: [{ name: 'generate', pipeline: {} }],
    },
    origin: 'examples',
    writable: false,
  } as never)
  vi.mocked(api.listWorkflows).mockResolvedValue({
    workflows: [],
    details: {
      'templates/text-to-image': { inputs: { image: 'image', mask: 'image' } },
    },
    libraries: [],
    shadowed: [],
  } as never)
  await openPage()
  expect(screen.getAllByRole('button', { name: /^Choose/ })).toHaveLength(2)
  expect(screen.getAllByRole('button', { name: /^Upload/ })).toHaveLength(2)
  expect(api.listWorkflows).toHaveBeenCalledWith({ inputs: true })
  expect(field('prompt').closest('.reference')).toBeNull()
})

it('a proof opens its file in the gallery', async () => {
  vi.mocked(api.gallery).mockResolvedValueOnce({
    workspace: 'studio',
    files: [
      {
        name: 'templates/text-to-image/r1/a.mp4',
        folder: 'templates/text-to-image',
        subfolder: '',
        run_id: 'r1',
        version: 1,
        url: '/outputs/a.mp4',
        kind: 'video',
        size: 1,
        mtime: 1,
        label: 'a.mp4',
      },
      {
        name: 'other/r2/b.png',
        folder: 'other',
        subfolder: '',
        run_id: 'r2',
        version: 1,
        url: '/outputs/b.png',
        kind: 'image',
        size: 1,
        mtime: 1,
        label: 'b.png',
      },
    ],
  } as never)
  await openPage()
  const tiles = [...document.querySelectorAll('.proofs a.filetile')]
  expect(tiles.map((a) => a.getAttribute('href'))).toEqual([
    '#/ws/studio/gallery?open=templates%2Ftext-to-image%2Fr1%2Fa.mp4',
  ])
  expect(tiles[0].querySelector('.videobadge')).toBeTruthy()
  expect(tiles[0].querySelector('img')?.getAttribute('src')).toContain(
    'workspace=studio',
  )
})

it('opens with an output: reference filled in from Use as input', async () => {
  sessionStorage.setItem(
    'dw-workflow-prefill',
    JSON.stringify({
      workflow: 'templates/text-to-image',
      arguments: { prompt: 'output:run/cat.png' },
    }),
  )
  await openPage()
  expect(field('prompt').value).toBe('output:run/cat.png')
  expect(sessionStorage.getItem('dw-workflow-prefill')).toBeNull()
})

describe('WorkflowPage run check', () => {
  it('validates on load by name, without sizes, and shows the plan', async () => {
    await openPage()
    expect(validateCalls()).toHaveLength(1)
    expect(validateCalls()[0][0]).toEqual({
      workflow_path: 'templates/text-to-image',
      arguments: {},
      sizes: false,
    })
    const plan = screen.getByText('What a job will do').closest('details')!
    expect(plan.textContent).toContain('about 2 min on mps')
    expect(screen.getByText('Ready to run.')).toBeTruthy()
  })

  it('asks about exactly the arguments Run sends, blanks dropped', async () => {
    vi.mocked(api.submitJob).mockResolvedValue({ id: 'job-1' } as never)
    await openPage()
    await type('prompt', 'a dog')
    await type('seed', '')
    await advance(600)
    const asked = validateCalls().at(-1)![0] as { arguments: unknown }
    expect(asked.arguments).toEqual({ prompt: 'a dog' })

    await fireEvent.click(screen.getByRole('button', { name: /Run/ }))
    await advance()
    expect(vi.mocked(api.submitJob).mock.calls[0][0]).toEqual({
      workflow_path: 'templates/text-to-image',
      arguments: asked.arguments,
    })
  })

  it('a double click sends one request', async () => {
    vi.mocked(api.submitJob).mockResolvedValue({ id: 'job-1' } as never)
    await openPage()
    const runButton = screen.getByRole('button', { name: /Run/ })
    // Both clicks land before the page renders the first one's busy state
    runButton.click()
    runButton.click()
    await advance()
    expect(api.submitJob).toHaveBeenCalledTimes(1)
  })

  it('checks once per pause, and asks for sizes only after 3 s idle', async () => {
    await openPage()
    vi.mocked(api.validate).mockClear()
    for (const value of ['a', 'a d', 'a do', 'a dog']) {
      await type('prompt', value)
      await advance(200)
    }
    expect(validateCalls()).toHaveLength(0)
    await advance(400)
    expect(validateCalls()).toHaveLength(1)
    expect(validateCalls()[0][0]).toMatchObject({ sizes: false })
    await advance(1500)
    expect(validateCalls()).toHaveLength(1)
    await advance(1100)
    expect(validateCalls()).toHaveLength(2)
    expect(validateCalls()[1][0]).toMatchObject({
      sizes: true,
      arguments: { prompt: 'a dog' },
    })
  })

  it('drops an answer from before the latest change, a late sizes one too', async () => {
    const refusal = {
      ...valid,
      valid: false,
      plan: undefined,
      errors: [{ path: 'arguments.seed', message: 'stale refusal' }],
    } as unknown as ValidationResult
    let answerSizes: (result: ValidationResult) => void = () => {}
    await openPage()
    // the sizes check for the load fires, and hangs
    vi.mocked(api.validate).mockImplementationOnce(
      () => new Promise((resolve) => (answerSizes = resolve)),
    )
    await advance(3000)
    expect(validateCalls().at(-1)![0]).toMatchObject({ sizes: true })
    // the user changes a value before it answers
    await type('seed', '7')
    await advance(600)
    // it answers now, after the change: not wanted
    answerSizes(refusal)
    await advance()
    expect(screen.queryByText(/stale refusal/)).toBeNull()
    expect(screen.getByText('Ready to run.')).toBeTruthy()
    // and the signal it was handed was aborted
    const sizesCall = validateCalls().find(
      (c) => (c[0] as { sizes?: boolean }).sizes === true,
    )!
    expect((sizesCall[1] as AbortSignal).aborted).toBe(true)
  })

  it('puts an arguments.x error under field x and the rest in the general list', async () => {
    vi.mocked(api.validate).mockResolvedValue({
      valid: false,
      error: 'refused',
      errors: [
        {
          path: 'arguments.num_images_per_prompt',
          message: 'must be at most 8',
        },
        { path: 'model_name', message: 'unknown model' },
      ],
      warnings: [
        'variables.seed[0]: a list where one value belongs',
        'cost estimate withheld: not priced from templates/x',
      ],
    } as unknown as ValidationResult)
    await openPage()

    const underCount = document.querySelector(
      '[data-findings-for="num_images_per_prompt"]',
    )!
    expect(underCount.textContent).toContain('must be at most 8')
    const underSeed = document.querySelector('[data-findings-for="seed"]')!
    expect(underSeed.textContent).toContain(
      '[0]: a list where one value belongs',
    )

    const general = screen.getByLabelText('findings')
    expect(general.textContent).toContain('model_name: unknown model')
    expect(general.textContent).not.toContain('must be at most 8')
    expect(general.querySelector('.bad')!.textContent).toContain('model_name')
    // a warning naming no field is the workflow author's, behind its fold
    expect(general.textContent).not.toContain('withheld')
    expect(
      screen.getByText("1 note for the workflow's author").closest('details')!
        .textContent,
    ).toContain('cost estimate withheld: not priced from templates/x')
    expect(
      screen.getByText('A job would be refused: 2 issues to fix.'),
    ).toBeTruthy()
  })

  it('clears a field error once the value is fixed', async () => {
    vi.mocked(api.validate).mockResolvedValueOnce(valid)
    await openPage()
    vi.mocked(api.validate).mockResolvedValueOnce({
      valid: false,
      error: 'refused',
      errors: [{ path: 'arguments.seed', message: 'not an integer' }],
      warnings: [],
    } as unknown as ValidationResult)
    await type('seed', 'x')
    await advance(600)
    expect(
      document.querySelector('[data-findings-for="seed"]')!.textContent,
    ).toContain('not an integer')
    await type('seed', '3')
    await advance(600)
    expect(document.querySelector('[data-findings-for="seed"]')).toBeNull()
  })

  const refused = (errors: { path: string; message: string }[]) =>
    ({
      valid: false,
      error: 'refused',
      errors,
      warnings: [],
    }) as unknown as ValidationResult

  it('a refused check turns Run into Fix 2 issues, which focuses the first field', async () => {
    vi.mocked(api.validate).mockResolvedValue(
      refused([
        { path: 'arguments.seed', message: 'not an integer' },
        { path: 'arguments.prompt', message: 'too long' },
      ]),
    )
    await openPage()
    const fix = screen.getByRole('button', { name: 'Fix 2 issues' })
    expect(fix.getAttribute('title')).toBe('Shows what would stop this job')
    await fireEvent.click(fix)
    expect(document.activeElement).toBe(field('prompt'))
    expect(api.submitJob).not.toHaveBeenCalled()
  })

  it('Fix scrolls to the first general issue when no field has one', async () => {
    vi.mocked(api.validate).mockResolvedValue(
      refused([{ path: 'model_name', message: 'unknown model' }]),
    )
    await openPage()
    const target = document.getElementById('issue-0')!
    target.scrollIntoView = vi.fn()
    await fireEvent.click(screen.getByRole('button', { name: 'Fix 1 issue' }))
    expect(target.scrollIntoView).toHaveBeenCalled()
  })

  it('Run comes back once the check stops refusing', async () => {
    vi.mocked(api.validate).mockResolvedValueOnce(
      refused([{ path: 'arguments.seed', message: 'not an integer' }]),
    )
    await openPage()
    expect(screen.getByRole('button', { name: 'Fix 1 issue' })).toBeTruthy()
    await type('seed', '3')
    await advance(600)
    const run = screen.getByRole('button', { name: 'Run' })
    vi.mocked(api.submitJob).mockRejectedValue(new Error('boom'))
    await fireEvent.click(run)
    await advance()
    expect(notify.failure).toHaveBeenCalledWith(
      expect.stringMatching(/^run /),
      expect.objectContaining({ message: 'boom' }),
    )
  })

  it('a gated verdict keeps Run', async () => {
    vi.mocked(api.validate).mockResolvedValue({
      ...valid,
      plan: {
        ...valid.plan,
        downloads_required: [
          { repo: 'a/b', access_blocked: true, not_found: false },
        ],
      },
    } as unknown as ValidationResult)
    await openPage()
    expect(
      screen.getByText('This model needs Hugging Face access.'),
    ).toBeTruthy()
    const run = screen.getByRole('button', { name: 'Run' })
    expect((run as HTMLButtonElement).disabled).toBe(false)
  })

  it('a check in flight does not block Run', async () => {
    vi.mocked(api.validate).mockImplementation(() => new Promise(() => {}))
    await openPage()
    const runButton = screen.getByRole('button', { name: /Run/ })
    expect((runButton as HTMLButtonElement).disabled).toBe(false)
  })

  it('renders hostile finding text as text', async () => {
    vi.mocked(api.validate).mockResolvedValue({
      ...valid,
      warnings: ['<img src=x onerror="alert(1)"> warned'],
    } as unknown as ValidationResult)
    await openPage()
    const notes = screen
      .getByText("1 note for the workflow's author")
      .closest('details')!
    expect(notes.textContent).toContain('<img src=x onerror="alert(1)">')
    expect(notes.querySelector('img')).toBeNull()
  })

  it('names a check the server refused outright in the general list', async () => {
    vi.mocked(api.validate).mockRejectedValue(
      new ApiError('unknown workflow', 400),
    )
    await openPage()
    // a refusal's own first line is its title, led by what was being done
    expect(screen.getByLabelText('findings').textContent).toContain(
      "Couldn't check these values: unknown workflow",
    )
  })
})

describe('WorkflowPage curated cost', () => {
  const card = (cost: unknown, observed: unknown = null) => ({
    workflows: ['templates/text-to-image'],
    cost_basis: 'curated',
    details: {
      'templates/text-to-image': {
        cost,
        // the card's preset-applied value, not the definition's 1
        cost_drivers: { num_images_per_prompt: 4 },
        observed,
      },
    },
    libraries: [],
    shadowed: [],
  })

  it("shows the maintainer's figure and its settings above the observed text", async () => {
    vi.mocked(api.listWorkflows).mockResolvedValueOnce(
      card([{ device: 'mps', name: 'M4 Max', vram_gb: 6.5, minutes: 2 }], {
        warm_minutes: 1.5,
        warm_runs: 2,
        runs: 2,
      }) as never,
    )
    await openPage()
    const line = screen.getByLabelText('curated cost')
    expect(line.textContent).toContain(
      "a maintainer's figure: about 2 min · 6.5 GB VRAM, measured on M4 Max at num_images_per_prompt=4",
    )
    expect(line.textContent).toContain('other values re-price it in the plan')
    const seen = screen.getByText('about 1.5 min here')
    expect(
      line.compareDocumentPosition(seen) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
  })

  it('prints no curated line for a null cost', async () => {
    vi.mocked(api.listWorkflows).mockResolvedValueOnce(card(null) as never)
    await openPage()
    expect(screen.queryByLabelText('curated cost')).toBeNull()
    expect(document.body.textContent).not.toContain("maintainer's figure")
  })
})

describe('WorkflowPage recovery', () => {
  it('a failed load offers Retry and recovers on reconnect', async () => {
    vi.mocked(api.getWorkflow).mockRejectedValueOnce(
      new NetworkError('fetch failed'),
    )
    await openPage()
    expect(
      screen.getByText(
        "Couldn't load templates/text-to-image: Can't reach the server",
      ),
    ).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
    // The raw message never reaches the page
    expect(screen.queryByText('fetch failed')).toBeNull()
    // The heading and the way back stay above the failure
    expect(
      screen.getByRole('heading', { name: 'templates/text-to-image' }),
    ).toBeTruthy()
    connection.epoch++
    await advance()
    await advance()
    expect(screen.queryByText(/Couldn't load/)).toBeNull()
    expect(screen.getByText('Ready to run.')).toBeTruthy()
  })

  it('a missing workflow is named in its 404', async () => {
    vi.mocked(api.getWorkflow).mockRejectedValueOnce(
      new ApiError('Workflow not found', 404),
    )
    await openPage()
    expect(
      screen.getByText(
        "Couldn't load templates/text-to-image: templates/text-to-image wasn't found",
      ),
    ).toBeTruthy()
  })

  it('a failed proofs read leaves the form alone', async () => {
    vi.mocked(api.gallery).mockRejectedValueOnce(
      new NetworkError('fetch failed'),
    )
    await openPage()
    expect(screen.queryByText(/Couldn't load/)).toBeNull()
    expect(screen.queryByRole('alert')).toBeNull()
    expect(document.getElementById('var-prompt')).not.toBeNull()
  })

  it('Retry reads the definition again', async () => {
    vi.mocked(api.getWorkflow).mockRejectedValueOnce(
      new NetworkError('fetch failed'),
    )
    await openPage()
    await fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await advance()
    await advance()
    expect(screen.queryByText(/Couldn't load/)).toBeNull()
    expect(document.getElementById('var-prompt')).not.toBeNull()
  })
})

describe('WorkflowPage remote change', () => {
  it('a definition changed elsewhere says so', async () => {
    await openPage()
    expect(screen.queryByText('Updated by an agent or another tab')).toBeNull()
    vi.mocked(api.getWorkflow).mockResolvedValue({
      definition: { ...definition, description: 'edited by an agent' },
      origin: 'examples',
      writable: false,
    } as never)
    feed.move('default', 'workflows')
    await advance()
    await advance()
    expect(screen.getByText('Updated by an agent or another tab')).toBeTruthy()
  })

  it('a move that leaves the definition alone says nothing', async () => {
    await openPage()
    feed.move('default', 'workflows')
    await advance()
    await advance()
    expect(screen.queryByText('Updated by an agent or another tab')).toBeNull()
  })
})

describe('WorkflowPage run form', () => {
  const folded = {
    id: 'TextToImage',
    variables: { prompt: 'a cat', dtype: 'bfloat16', seed: null },
    seed: 'variable:seed',
    steps: [{ name: 'generate', pipeline: {} }],
  }
  const advanced = () => screen.getByRole('button', { name: /^Advanced/ })

  beforeEach(() => {
    localStorage.clear()
    sessionStorage.clear()
    vi.mocked(api.getWorkflow).mockResolvedValue({
      definition: folded,
      origin: 'examples',
      writable: false,
    } as never)
    vi.mocked(api.listWorkflows).mockResolvedValue({
      workflows: [],
      details: { 'templates/text-to-image': { inputs: {} } },
      libraries: [],
      shadowed: [],
    } as never)
  })

  it('advanced variables are folded until one holds a value', async () => {
    await openPage()
    expect(advanced().textContent).toContain('(2)')
    expect(advanced().getAttribute('aria-expanded')).toBe('false')
    expect(document.getElementById('var-dtype')).toBeNull()
    await fireEvent.click(advanced())
    expect(field('dtype')).toBeTruthy()
    cleanup()
    sessionStorage.setItem(
      'dw-workflow-prefill',
      JSON.stringify({
        workflow: 'templates/text-to-image',
        arguments: { dtype: 'float16' },
      }),
    )
    await openPage()
    expect(advanced().getAttribute('aria-expanded')).toBe('true')
    expect(field('dtype').value).toBe('float16')
  })

  it('a field with a finding opens the fold, so Fix focuses it', async () => {
    vi.mocked(api.validate).mockResolvedValue({
      valid: false,
      error: null,
      errors: [{ path: 'arguments.dtype', message: 'unknown dtype' }],
      warnings: [],
      plan: null,
    } as never)
    await openPage()
    expect(advanced().getAttribute('aria-expanded')).toBe('true')
    await fireEvent.click(screen.getByRole('button', { name: 'Fix 1 issue' }))
    expect(document.activeElement).toBe(field('dtype'))
  })

  it('a seed with no default says random each job', async () => {
    await openPage()
    await fireEvent.click(advanced())
    expect(field('seed').getAttribute('placeholder')).toBe('random each job')
  })

  it('values typed here are back on the next visit', async () => {
    await openPage()
    await type('prompt', 'a dog')
    await advance(300)
    cleanup()
    await openPage()
    expect(field('prompt').value).toBe('a dog')
    expect(screen.getByText(/Restored what you last typed here/)).toBeTruthy()
  })

  it('a prefill wins over the draft', async () => {
    await openPage()
    await type('prompt', 'a dog')
    await advance(300)
    cleanup()
    sessionStorage.setItem(
      'dw-workflow-prefill',
      JSON.stringify({
        workflow: 'templates/text-to-image',
        arguments: { prompt: 'output:run/cat.png' },
      }),
    )
    await openPage()
    expect(field('prompt').value).toBe('output:run/cat.png')
  })

  it('Reset to defaults clears the form and the draft', async () => {
    await openPage()
    await type('prompt', 'a dog')
    await advance(300)
    cleanup()
    await openPage()
    await fireEvent.click(
      screen.getByRole('button', { name: 'Reset to defaults' }),
    )
    await advance(300)
    expect(field('prompt').value).toBe('')
    expect(screen.queryByText(/Restored what you last typed/)).toBeNull()
    cleanup()
    await openPage()
    expect(field('prompt').value).toBe('')
    expect(localStorage.length).toBe(0)
  })

  it('a workspace change reseeds from that workspace, leaving the first draft alone', async () => {
    const home = workspace.current
    try {
      await openPage()
      await type('prompt', 'a dog')
      await advance(300)
      const homeKey = `run-draft:${home}:templates/text-to-image`
      const before = storageGet(homeKey, null)
      expect(before).toEqual({ prompt: 'a dog' })
      workspace.current = 'other'
      await tick()
      await advance(300)
      expect(field('prompt').value).toBe('')
      expect(storageGet(homeKey, null)).toEqual(before)
      expect(
        storageGet('run-draft:other:templates/text-to-image', null),
      ).toBeNull()
    } finally {
      workspace.current = home
    }
  })

  it('a Run keeps the draft', async () => {
    vi.mocked(api.submitJob).mockResolvedValue({ id: 'job-1' } as never)
    await openPage()
    await type('prompt', 'a dog')
    await advance(300)
    await fireEvent.click(screen.getByRole('button', { name: 'Run' }))
    await advance(300)
    expect(api.submitJob).toHaveBeenCalledTimes(1)
    expect(localStorage.length).toBe(1)
    expect(localStorage.key(0)).toContain('run-draft:')
  })
})

it('only free-text fields offer Enhance', async () => {
  vi.mocked(api.getWorkflow).mockResolvedValue({
    definition: {
      ...definition,
      variables: { prompt: 'a cat', dtype: 'float16', seed: 0 },
    },
    origin: 'examples',
    writable: false,
  } as never)
  await openPage()
  expect(screen.getAllByRole('button', { name: 'Enhance…' })).toHaveLength(1)
})
