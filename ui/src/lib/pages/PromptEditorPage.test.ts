// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import PromptEditorPage from './PromptEditorPage.svelte'
import { api } from '../api'

// Monaco cannot boot in jsdom; the JSON view is read through this stub,
// which renders the value it was handed as plain text
vi.mock('../editor/JsonEditor.svelte', async () => ({
  default: (await import('./JsonEditorStub.svelte')).default,
}))

vi.mock('../api', () => ({
  api: {
    listPrompts: vi.fn().mockResolvedValue({
      prompts: ['portraits/hero'],
      libraries: [{ origin: 'workspace', root: 'prompts', writable: true }],
      shadowed: [],
      details: {},
    }),
    getPrompt: vi.fn().mockResolvedValue({
      prompt: { text: 'a hero at dawn' },
      writable: true,
    }),
    savePrompt: vi.fn().mockResolvedValue({}),
    listModels: vi.fn().mockResolvedValue({ repos: [] }),
    listEnhancers: vi.fn().mockResolvedValue({
      presets: [
        {
          key: 'h3',
          label: 'H3',
          default_model: 'org/enhancer',
          models: ['org/enhancer'],
          intended_models: [],
          placeholder: 'an idea',
        },
      ],
    }),
    startDownload: vi.fn().mockResolvedValue({}),
    listDownloads: vi.fn().mockResolvedValue({
      downloads: [{ repo_id: 'org/enhancer', status: 'downloading' }],
    }),
    listWorkflows: vi.fn().mockResolvedValue({ workflows: [], details: {} }),
    promptDownloadUrl: (name: string) => `/api/prompts/${name}/download`,
  },
  fetchOutputText: vi.fn(),
  streamJobEvents: vi.fn(() => () => {}),
}))
vi.mock('../toast', () => ({
  notify: { error: vi.fn(), success: vi.fn(), dismiss: vi.fn() },
}))

beforeEach(() => {
  try {
    localStorage.removeItem('dw-prompt-editor-view')
  } catch {
    /* no storage */
  }
})

afterEach(() => {
  cleanup()
  vi.mocked(api.savePrompt).mockClear()
  vi.mocked(api.startDownload).mockClear()
})

const promptText = () =>
  document.getElementById('prompt-text') as HTMLTextAreaElement

async function openHero() {
  render(PromptEditorPage, { name: 'portraits/hero' })
  await waitFor(() => expect(promptText().value).toBe('a hero at dawn'))
}

function unloadPrevented() {
  const event = new Event('beforeunload', { cancelable: true })
  window.dispatchEvent(event)
  return event.defaultPrevented
}

it('mounts a stored prompt under the folder and name it was saved as', async () => {
  await openHero()
  expect((document.querySelector('.savename') as HTMLInputElement).value).toBe(
    'hero',
  )
  await waitFor(() =>
    expect(
      (document.querySelector('.folderpick') as HTMLSelectElement).value,
    ).toBe('portraits'),
  )
})

it('keeps an edit across form -> JSON -> form', async () => {
  await openHero()
  await fireEvent.change(promptText(), { target: { value: 'a hero at dusk' } })
  await fireEvent.click(screen.getByTitle('edit the raw JSON, schema-aware'))
  await waitFor(() =>
    expect(screen.getByTestId('json-editor').textContent).toContain(
      'a hero at dusk',
    ),
  )
  await fireEvent.click(screen.getByTitle('edit with a form'))
  await waitFor(() => expect(promptText().value).toBe('a hero at dusk'))
})

it('saves to the folder and name shown on Ctrl+S', async () => {
  await openHero()
  await waitFor(() =>
    expect(
      (document.querySelector('.folderpick') as HTMLSelectElement).value,
    ).toBe('portraits'),
  )
  await fireEvent.keyDown(window, { key: 's', ctrlKey: true })
  await waitFor(() => expect(api.savePrompt).toHaveBeenCalledTimes(1))
  expect(vi.mocked(api.savePrompt).mock.calls[0][0]).toBe('portraits/hero')
  expect(vi.mocked(api.savePrompt).mock.calls[0][1]).toEqual({
    text: 'a hero at dawn',
  })
})

it('guards a tab close only while there are unsaved edits', async () => {
  await openHero()
  expect(unloadPrevented()).toBe(false)
  await fireEvent.change(promptText(), { target: { value: 'changed' } })
  expect(unloadPrevented()).toBe(true)
})

it('keeps a model download in flight across a view switch', async () => {
  await openHero()
  const get = await screen.findByTitle(
    'download this model to the local cache now - otherwise the first enhancement downloads it',
  )
  await fireEvent.click(get)
  expect(api.startDownload).toHaveBeenCalledWith('org/enhancer')
  await waitFor(() => expect(screen.getByText('downloading…')).toBeTruthy())
  await fireEvent.click(screen.getByTitle('edit the raw JSON, schema-aware'))
  await fireEvent.click(screen.getByTitle('edit with a form'))
  await waitFor(() => expect(screen.getByText('downloading…')).toBeTruthy())
  expect(
    screen.queryByTitle(
      'download this model to the local cache now - otherwise the first enhancement downloads it',
    ),
  ).toBeNull()
})
