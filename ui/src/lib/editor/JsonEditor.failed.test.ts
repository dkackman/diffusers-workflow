// A tab opened before the server was upgraded asks for a Monaco chunk that
// no longer exists: the JSON view says so, and the form view stays usable.
// Its own file because the module mock that makes the import reject holds
// for the whole file (JsonEditor.test.ts mocks a Monaco that loads)
import { render, screen } from '@testing-library/svelte'
import { createRawSnippet } from 'svelte'
import { describe, expect, it, vi } from 'vitest'

vi.mock('../monaco', () => {
  throw new TypeError(
    'Failed to fetch dynamically imported module: /assets/monaco-0ld.js',
  )
})

import EditorBody from './EditorBody.svelte'

const form = createRawSnippet(() => ({
  render: () => '<p data-testid="form">the form</p>',
}))

const COPY =
  "The JSON editor couldn't load. Reload the page; if that doesn't help, check that the server is running."

describe('JsonEditor without Monaco', () => {
  it('a JSON editor that cannot load Monaco says so', async () => {
    const props = {
      view: 'json' as const,
      jsonDraft: '{}',
      onjson: () => {},
      hint: '',
      stickyTop: '0',
      form,
    }
    const view = render(EditorBody, props)
    const failed = await screen.findByRole('alert')
    expect(failed).toHaveTextContent(COPY)
    expect(screen.queryByText('loading editor…')).toBeNull()
    // the failure sits behind its disclosure (vitest words the rejection
    // a mocked module makes, so only its presence is asserted)
    const details = screen.getByText('What failed').closest('details')
    expect(details?.querySelector('pre')?.textContent?.trim()).not.toBe('')
    expect(details).not.toHaveAttribute('open')

    await view.rerender({ ...props, view: 'form' })
    expect(screen.getByTestId('form')).toHaveTextContent('the form')
    expect(screen.queryByRole('alert')).toBeNull()
  })
})
