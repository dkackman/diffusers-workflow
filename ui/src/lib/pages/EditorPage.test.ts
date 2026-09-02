// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/svelte'
import EditorPage from './EditorPage.svelte'

// EditorPage talks to the server on mount (pipeline/class/task catalogs,
// the workflow listing, the prompt library) purely to feed forms the flow
// view never touches - stubbed here so the view-switch test stays about
// the view switch, not the network.
vi.mock('../api', () => ({
  api: {
    listPipelines: vi.fn().mockResolvedValue({ pipelines: [] }),
    listClasses: vi.fn().mockResolvedValue({ classes: [] }),
    listTasks: vi.fn().mockResolvedValue({
      commands: [],
      image_processors: [],
      video_processors: [],
    }),
    listWorkflows: vi
      .fn()
      .mockResolvedValue({ workflows: [], workflow_dir: 'workflows' }),
    listPrompts: vi.fn().mockResolvedValue({ prompts: [], details: {} }),
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
