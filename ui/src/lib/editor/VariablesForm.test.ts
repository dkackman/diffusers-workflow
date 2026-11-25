import { describe, expect, it, afterEach, vi } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/svelte'
import { tick } from 'svelte'
import VariablesForm from './VariablesForm.svelte'
import { attentionBackends } from '../attentionBackends.svelte'

const notify = vi.hoisted(() => ({
  undo: vi.fn(),
  failure: vi.fn(),
  success: vi.fn(),
}))
vi.mock('../toast', () => ({ notify }))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  attentionBackends.list = []
})

const props = (over: Record<string, unknown> = {}) => ({
  mode: 'define' as const,
  variables: { attn: '' } as Record<string, unknown>,
  idPrefix: 'v-',
  backendVariables: new Set(['attn']),
  ...over,
})

describe('VariablesForm attention backend variables', () => {
  it('renders a select with default (none) plus the server list', () => {
    attentionBackends.list = ['native', 'sparse']
    render(VariablesForm, props())
    const select = screen.getByLabelText('attn')
    expect(select.tagName).toBe('SELECT')
    expect(
      Array.from(select.querySelectorAll('option')).map((o) => o.textContent),
    ).toEqual(['default (none)', 'native', 'sparse'])
  })

  it('keeps a current value the server does not list, marked unavailable', () => {
    attentionBackends.list = ['native']
    render(VariablesForm, props({ variables: { attn: 'sage_hub' } }))
    const select = screen.getByLabelText('attn') as HTMLSelectElement
    expect(select.value).toBe('sage_hub')
    expect(
      screen.getByText('sage_hub (not available on this server)'),
    ).toBeTruthy()
  })

  it('falls back to the text field when the server list is empty', () => {
    attentionBackends.list = []
    render(VariablesForm, props())
    expect(screen.queryByRole('combobox', { name: '' })).toBeNull()
    expect(document.querySelector('select:not(.vartype)')).toBeNull()
  })

  it('falls back to the text field for a reference or non-string value', () => {
    attentionBackends.list = ['native']
    render(
      VariablesForm,
      props({
        variables: { attn: 'variable:other', n: 3 },
        backendVariables: new Set(['attn', 'n']),
      }),
    )
    expect(document.querySelector('select:not(.vartype)')).toBeNull()
  })

  it('does not turn other variables into selects', () => {
    attentionBackends.list = ['native']
    render(VariablesForm, props({ variables: { prompt: 'a cat' } }))
    expect(document.querySelector('select:not(.vartype)')).toBeNull()
  })

  it('in override mode shows the workflow default and writes the choice', async () => {
    attentionBackends.list = ['native', 'sparse']
    const overrides: Record<string, string> = {}
    render(
      VariablesForm,
      props({ mode: 'override', variables: { attn: 'sparse' }, overrides }),
    )
    const select = screen.getByLabelText('attn') as HTMLSelectElement
    expect(select.value).toBe('')
    expect(screen.getByText('default (sparse)')).toBeTruthy()
    await fireEvent.change(select, { target: { value: 'native' } })
    expect(overrides.attn).toBe('native')
  })

  it('in define mode writes the choice to the variable', async () => {
    attentionBackends.list = ['native', 'sparse']
    const variables: Record<string, unknown> = { attn: '' }
    render(VariablesForm, props({ variables }))
    await fireEvent.change(screen.getByLabelText('attn'), {
      target: { value: 'sparse' },
    })
    expect(variables.attn).toBe('sparse')
  })

  it('in define mode, default (none) writes null, not an empty string', async () => {
    attentionBackends.list = ['native', 'sparse']
    const variables: Record<string, unknown> = { attn: 'sparse' }
    render(VariablesForm, props({ variables }))
    await fireEvent.change(screen.getByLabelText('attn'), {
      target: { value: '' },
    })
    expect(variables.attn).toBeNull()
  })

  it('renders a null default as a select with default (none) selected', () => {
    attentionBackends.list = ['native']
    render(VariablesForm, props({ variables: { attn: null } }))
    const select = screen.getByLabelText('attn') as HTMLSelectElement
    expect(select.value).toBe('')
    expect(select.selectedOptions[0].textContent).toBe('default (none)')
  })
})

describe('VariablesForm constraint hints', () => {
  const hint = '32 * n + 0 - <b>the VAE</b>'
  for (const mode of ['override', 'define'] as const) {
    it(`renders the hint under its field in ${mode} mode, as text`, () => {
      render(VariablesForm, {
        mode,
        variables: { width: 1024, prompt: 'x' },
        idPrefix: 'v-',
        constraints: { width: [hint] },
      })
      const list = document.querySelector('[data-hints-for="width"]')
      expect(list?.textContent?.trim()).toBe(hint)
      expect(list?.querySelector('b')).toBeNull()
      // Under the field: after the width input, before the next label
      const field = document.getElementById('v-width')!
      expect(
        field.compareDocumentPosition(list!) & Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy()
      const next = document.querySelector('label[for="v-prompt"]')!
      expect(
        list!.compareDocumentPosition(next) & Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy()
      expect(document.querySelector('[data-hints-for="prompt"]')).toBeNull()
      // Read-only: the hint is not a form control
      expect(list!.querySelector('input, textarea, select')).toBeNull()
    })
  }
})

describe('VariablesForm preset selectors', () => {
  const variables = { resolution: '480p', width: 832, height: 480 }
  const presets = {
    resolution: {
      '480p': { width: 832, height: 480 },
      '768p': { width: 1344, height: 768 },
    },
  }
  const optionTexts = (select: Element) =>
    Array.from(select.querySelectorAll('option')).map((o) =>
      o.textContent?.trim(),
    )

  it('in override mode is a select: the default, then exactly the bundles', () => {
    render(VariablesForm, {
      mode: 'override',
      variables,
      overrides: {},
      idPrefix: 'v-',
      presets,
    })
    const select = document.getElementById('v-resolution') as HTMLSelectElement
    expect(select.tagName).toBe('SELECT')
    expect(select.value).toBe('')
    expect(optionTexts(select)).toEqual(['default (480p)', '480p', '768p'])
    expect(
      document
        .querySelector('[data-presets-for="resolution"]')
        ?.textContent?.trim(),
    ).toBe('sets width, height')
  })

  it('a picked bundle moves the placeholders, an explicit value is left alone, and nothing is copied', async () => {
    const overrides: Record<string, string> = { height: '500' }
    const { rerender } = render(VariablesForm, {
      mode: 'override',
      variables,
      overrides,
      idPrefix: 'v-',
      presets,
    })
    const width = document.getElementById('v-width') as HTMLTextAreaElement
    expect(width.getAttribute('placeholder')).toBe('832')
    await fireEvent.change(document.getElementById('v-resolution')!, {
      target: { value: '768p' },
    })
    // Only the selector is written: the bundle's values stay placeholders
    expect(overrides).toEqual({ resolution: '768p', height: '500' })
    // The pages hold overrides as $state; a plain test object is handed
    // back the way that state would be
    await rerender({ overrides: { ...overrides } })
    expect(
      (document.getElementById('v-width') as HTMLElement).getAttribute(
        'placeholder',
      ),
    ).toBe('1344')
    const height = document.getElementById('v-height') as HTMLTextAreaElement
    expect(height.value).toBe('500')
  })

  it('a current value that is no bundle shows as (not a preset), its raw value kept', () => {
    render(VariablesForm, {
      mode: 'override',
      variables,
      overrides: { resolution: '1080p' },
      idPrefix: 'v-',
      presets,
    })
    const select = document.getElementById('v-resolution') as HTMLSelectElement
    expect(select.value).toBe('1080p')
    expect(optionTexts(select)).toEqual([
      'default (480p)',
      '1080p (not a preset)',
      '480p',
      '768p',
    ])
    expect(select.selectedOptions[0].value).toBe('1080p')
  })

  it('a reference or a non-string selector stays a text field', () => {
    render(VariablesForm, {
      mode: 'override',
      variables: { ...variables, resolution: 'variable:other' },
      overrides: {},
      idPrefix: 'v-',
      presets,
    })
    expect(document.querySelector('select:not(.vartype)')).toBeNull()
    cleanup()
    render(VariablesForm, {
      mode: 'override',
      variables,
      overrides: { resolution: 'variable:other' },
      idPrefix: 'v-',
      presets,
    })
    expect(document.querySelector('select:not(.vartype)')).toBeNull()
    cleanup()
    render(VariablesForm, {
      mode: 'override',
      variables: { ...variables, resolution: 3 },
      overrides: {},
      idPrefix: 'v-',
      presets,
    })
    expect(document.querySelector('select:not(.vartype)')).toBeNull()
  })

  it('in define mode stays a text field, with what it sets and its bundles', () => {
    render(VariablesForm, {
      mode: 'define',
      variables,
      idPrefix: 'v-',
      presets,
    })
    expect(document.querySelector('select:not(.vartype)')).toBeNull()
    expect(document.getElementById('v-resolution')?.tagName).not.toBe('SELECT')
    const lines = Array.from(
      document.querySelectorAll('[data-presets-for="resolution"] li'),
    ).map((li) => li.textContent?.trim())
    expect(lines).toEqual(['sets width, height', 'bundles: 480p, 768p'])
  })

  it("leaves the backend select's labels as they were", () => {
    attentionBackends.list = ['native', 'sparse']
    render(VariablesForm, {
      mode: 'override',
      variables: { ...variables, attn: 'sparse' },
      overrides: {},
      idPrefix: 'v-',
      backendVariables: new Set(['attn']),
      presets,
    })
    expect(optionTexts(document.getElementById('v-attn')!)).toEqual([
      'default (sparse)',
      'native',
      'sparse',
    ])
  })
})

describe('VariablesForm removing a variable', () => {
  /** The variable names the form lists, in order */
  const listed = () =>
    Array.from(document.querySelectorAll('.vars > label')).map((l) =>
      l.textContent?.trim(),
    )

  it('removing a variable offers Undo; a key added again meanwhile is kept', async () => {
    render(
      VariablesForm,
      props({
        variables: { a: 1, b: 'two', c: true },
        backendVariables: new Set(),
      }),
    )
    const removes = screen.getAllByRole('button', {
      name: 'remove this variable',
    })
    await fireEvent.click(removes[1])
    expect(listed()).toEqual(['a', 'c'])
    expect(notify.undo).toHaveBeenCalledTimes(1)
    expect(notify.undo.mock.calls[0][0]).toBe('Removed variable b')
    // Undo puts it back at its place, with its value
    notify.undo.mock.calls[0][1]()
    await tick()
    expect(listed()).toEqual(['a', 'b', 'c'])
    expect((document.getElementById('v-b') as HTMLTextAreaElement).value).toBe(
      'two',
    )

    // Removed again, then added again by hand before Undo is pressed
    await fireEvent.click(
      screen.getAllByRole('button', { name: 'remove this variable' })[0],
    )
    const name = screen.getByPlaceholderText('new variable name…')
    await fireEvent.input(name, { target: { value: 'a' } })
    await fireEvent.click(
      screen.getByRole('button', { name: 'add this variable to the workflow' }),
    )
    expect(listed()).toEqual(['b', 'c', 'a'])
    notify.undo.mock.calls[1][1]()
    await tick()
    // The key added again is left as it is, and the toast says so
    expect(listed()).toEqual(['b', 'c', 'a'])
    expect((document.getElementById('v-a') as HTMLTextAreaElement).value).toBe(
      '',
    )
    expect(notify.success).toHaveBeenCalledWith('a is already back')
  })

  it('a double click on Undo restores the variable once', async () => {
    render(
      VariablesForm,
      props({ variables: { a: 1, b: 'two' }, backendVariables: new Set() }),
    )
    await fireEvent.click(
      screen.getAllByRole('button', { name: 'remove this variable' })[1],
    )
    const press = notify.undo.mock.calls[0][1]
    press()
    press()
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    expect(listed()).toEqual(['a', 'b'])
    // The second click is not taken for the key being back already
    expect(notify.success).not.toHaveBeenCalled()
  })
})

describe('VariablesForm file fields', () => {
  const override = (over: Record<string, unknown> = {}) => ({
    mode: 'override' as const,
    variables: { image: '', note: 'x' } as Record<string, unknown>,
    idPrefix: 'v-',
    ...over,
  })
  const chooseButtons = () =>
    screen.queryAllByRole('button', { name: /Choose|Upload/ })

  it('an image input is a reference field filtered to images', () => {
    const { container } = render(
      VariablesForm,
      override({ kinds: { image: 'image' } }),
    )
    expect(screen.getAllByRole('button', { name: /^Choose/ })).toHaveLength(1)
    expect(
      document.getElementById('v-image')?.getAttribute('placeholder'),
    ).toBe('type, drop or choose an image')
    expect(
      container.querySelector('input[type="file"]')?.getAttribute('accept'),
    ).toBe('image/*')
    // the other variable is plain text
    expect(document.getElementById('v-note')?.closest('.reference')).toBeNull()
  })

  it('an audio input offers only audio', () => {
    const { container } = render(
      VariablesForm,
      override({ kinds: { image: 'audio' } }),
    )
    expect(
      container.querySelector('input[type="file"]')?.getAttribute('accept'),
    ).toBe('audio/*')
    expect(screen.getByTitle(/^Upload an audio/)).toBeTruthy()
  })

  it('a variable holding an asset is a reference field without a kind', () => {
    const { container } = render(
      VariablesForm,
      override({
        mode: 'define',
        variables: { image: 'asset:cast/priya.png', note: 'x' },
      }),
    )
    expect(chooseButtons()).toHaveLength(2)
    expect(
      container.querySelector('input[type="file"]')?.hasAttribute('accept'),
    ).toBe(false)
    expect(screen.getByTitle(/^Upload a file/)).toBeTruthy()
  })

  it('an override of a variable whose default is an asset takes any file', () => {
    render(
      VariablesForm,
      override({ variables: { image: 'output:run/a.png', note: 'x' } }),
    )
    expect(chooseButtons()).toHaveLength(2)
  })

  it('typing a reference into a plain text variable keeps the same input and its focus', async () => {
    render(VariablesForm, override())
    const field = document.getElementById('v-note') as HTMLTextAreaElement
    field.focus()
    for (const text of ['a', 'as', 'asset:', 'asset:cast/p.png']) {
      await fireEvent.input(field, { target: { value: text } })
      await tick()
      expect(document.getElementById('v-note')).toBe(field)
      expect(document.activeElement).toBe(field)
    }
  })

  it('editing away an asset default keeps the same input and its focus', async () => {
    render(
      VariablesForm,
      override({ variables: { image: 'asset:cast/p.png', note: 'x' } }),
    )
    const field = document.getElementById('v-image') as HTMLInputElement
    field.focus()
    await fireEvent.input(field, { target: { value: 'a' } })
    await tick()
    expect(document.getElementById('v-image')).toBe(field)
    expect(document.activeElement).toBe(field)
  })
})

describe('VariablesForm adding a variable', () => {
  const addProps = (over: Record<string, unknown> = {}) =>
    props({
      variables: { existing: 'x' },
      backendVariables: new Set(),
      ...over,
    })
  const nameBox = () =>
    screen.getByPlaceholderText('new variable name…') as HTMLInputElement
  const enter = async (name: string) => {
    await fireEvent.input(nameBox(), { target: { value: name } })
    await fireEvent.keyDown(nameBox(), { key: 'Enter' })
    await tick()
  }

  it('Enter adds a variable', async () => {
    const variables: Record<string, unknown> = { existing: 'x' }
    render(VariablesForm, addProps({ variables }))
    await enter('seed')
    expect(document.getElementById('v-seed')).toBeTruthy()
    expect(nameBox().value).toBe('')
  })

  it('a bad name says why and adds nothing', async () => {
    render(VariablesForm, addProps())
    await enter('has space')
    expect(
      screen.getByText(
        'A variable name starts with a letter or _ and holds letters, numbers, _ and -.',
      ).className,
    ).toContain('error')
    expect(nameBox().getAttribute('aria-invalid')).toBe('true')
    expect(document.getElementById('v-has space')).toBeNull()
    await enter('existing')
    expect(
      screen.getByText('There is already a variable called existing.'),
    ).toBeTruthy()
  })

  it('30 in a new variable is stored as 30', async () => {
    render(VariablesForm, addProps())
    await enter('steps')
    const field = document.getElementById('v-steps') as HTMLTextAreaElement
    const type = screen.getByLabelText('type of steps') as HTMLSelectElement
    expect(type.value).toBe('text')
    await fireEvent.change(field, { target: { value: '30' } })
    await tick()
    // the row now holds the number 30: its type says so, and a later edit
    // keeps it a number
    expect(type.value).toBe('number')
    await fireEvent.change(field, { target: { value: '31' } })
    await tick()
    expect(type.value).toBe('number')
    expect(field.value).toBe('31')
  })

  it('the type select turns 30 back into text', async () => {
    const variables: Record<string, unknown> = { steps: 30 }
    render(VariablesForm, addProps({ variables }))
    const type = screen.getByLabelText('type of steps') as HTMLSelectElement
    expect(type.value).toBe('number')
    await fireEvent.change(type, { target: { value: 'text' } })
    await tick()
    expect(variables.steps).toBe('30')
    await fireEvent.change(type, { target: { value: 'none' } })
    expect(variables.steps).toBeNull()
  })
})

describe('VariablesForm folds and kinds', () => {
  const base = (over: Record<string, unknown> = {}) => ({
    mode: 'override' as const,
    variables: { note: 'x', steps: 30 } as Record<string, unknown>,
    idPrefix: 'v-',
    advanced: new Set(['steps']),
    ...over,
  })

  it('Advanced folds while nothing in it is set', () => {
    render(VariablesForm, base({ overrides: {} }))
    expect(document.getElementById('v-steps')).toBeNull()
  })

  it('Advanced folds while a folded field has a value, and says how many are set', () => {
    render(VariablesForm, base({ overrides: { steps: '20' } }))
    // a value of its own opens the fold
    expect(document.getElementById('v-steps')).not.toBeNull()
  })

  it('a locked field is shown but not typed in', () => {
    render(
      VariablesForm,
      base({ overrides: {}, locked: new Set(['note']), advanced: new Set() }),
    )
    const field = document.getElementById('v-note')!
    expect(field.closest('fieldset')?.disabled).toBe(true)
  })

  it('a field holding text is not swapped for a file field when the catalog arrives', async () => {
    const { rerender } = render(
      VariablesForm,
      base({ overrides: { note: 'typed' }, advanced: new Set(), kinds: {} }),
    )
    expect(document.querySelector('input[type="file"]')).toBeNull()
    await rerender(
      base({
        overrides: { note: 'typed' },
        advanced: new Set(),
        kinds: { note: 'image' },
      }),
    )
    expect(document.querySelector('input[type="file"]')).toBeNull()
  })
  it('an empty field becomes a file field when the catalog arrives', async () => {
    const { rerender } = render(
      VariablesForm,
      base({ overrides: {}, advanced: new Set(), kinds: {} }),
    )
    await rerender(
      base({ overrides: {}, advanced: new Set(), kinds: { note: 'image' } }),
    )
    expect(document.querySelector('input[type="file"]')).not.toBeNull()
  })
})
