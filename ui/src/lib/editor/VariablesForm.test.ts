import { describe, expect, it, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/svelte'
import VariablesForm from './VariablesForm.svelte'
import { attentionBackends } from '../attentionBackends.svelte'

afterEach(() => {
  cleanup()
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
    const select = screen.getByRole('combobox')
    expect(select.tagName).toBe('SELECT')
    expect(
      Array.from(select.querySelectorAll('option')).map((o) => o.textContent),
    ).toEqual(['default (none)', 'native', 'sparse'])
  })

  it('keeps a current value the server does not list, marked unavailable', () => {
    attentionBackends.list = ['native']
    render(VariablesForm, props({ variables: { attn: 'sage_hub' } }))
    const select = screen.getByRole('combobox') as HTMLSelectElement
    expect(select.value).toBe('sage_hub')
    expect(
      screen.getByText('sage_hub (not available on this server)'),
    ).toBeTruthy()
  })

  it('falls back to the text field when the server list is empty', () => {
    attentionBackends.list = []
    render(VariablesForm, props())
    expect(screen.queryByRole('combobox', { name: '' })).toBeNull()
    expect(document.querySelector('select')).toBeNull()
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
    expect(document.querySelector('select')).toBeNull()
  })

  it('does not turn other variables into selects', () => {
    attentionBackends.list = ['native']
    render(VariablesForm, props({ variables: { prompt: 'a cat' } }))
    expect(document.querySelector('select')).toBeNull()
  })

  it('in override mode shows the workflow default and writes the choice', async () => {
    attentionBackends.list = ['native', 'sparse']
    const overrides: Record<string, string> = {}
    render(
      VariablesForm,
      props({ mode: 'override', variables: { attn: 'sparse' }, overrides }),
    )
    const select = screen.getByRole('combobox') as HTMLSelectElement
    expect(select.value).toBe('')
    expect(screen.getByText('default (sparse)')).toBeTruthy()
    await fireEvent.change(select, { target: { value: 'native' } })
    expect(overrides.attn).toBe('native')
  })

  it('in define mode writes the choice to the variable', async () => {
    attentionBackends.list = ['native', 'sparse']
    const variables: Record<string, unknown> = { attn: '' }
    render(VariablesForm, props({ variables }))
    await fireEvent.change(screen.getByRole('combobox'), {
      target: { value: 'sparse' },
    })
    expect(variables.attn).toBe('sparse')
  })

  it('in define mode, default (none) writes null, not an empty string', async () => {
    attentionBackends.list = ['native', 'sparse']
    const variables: Record<string, unknown> = { attn: 'sparse' }
    render(VariablesForm, props({ variables }))
    await fireEvent.change(screen.getByRole('combobox'), {
      target: { value: '' },
    })
    expect(variables.attn).toBeNull()
  })

  it('renders a null default as a select with default (none) selected', () => {
    attentionBackends.list = ['native']
    render(VariablesForm, props({ variables: { attn: null } }))
    const select = screen.getByRole('combobox') as HTMLSelectElement
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
