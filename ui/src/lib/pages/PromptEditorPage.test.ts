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
import { ApiError } from '../apiErrors'
import { chooseDialog } from '../confirm.svelte'
import { feed } from '../test/feedMock'

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
    listPrompts: vi.fn().mockResolvedValue({
      prompts: ['portraits/hero'],
      libraries: [{ origin: 'workspace', root: 'prompts', writable: true }],
      shadowed: [],
      details: {},
    }),
    getPrompt: vi.fn().mockResolvedValue({
      prompt: { text: 'a hero at dawn' },
      writable: true,
      etag: '"p1"',
    }),
    savePrompt: vi.fn().mockResolvedValue({ saved: {}, etag: '"p2"' }),
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
  vi.mocked(chooseDialog).mockReset()
  location.hash = ''
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
  await fireEvent.input(promptText(), { target: { value: 'a hero at dusk' } })
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
  // a re-save insists on the version it read
  expect(vi.mocked(api.savePrompt).mock.calls[0][2]).toEqual({
    ifMatch: '"p1"',
  })
})

it('guards a tab close only while there are unsaved edits', async () => {
  await openHero()
  expect(unloadPrevented()).toBe(false)
  await fireEvent.input(promptText(), { target: { value: 'changed' } })
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

const saveButton = () =>
  screen.getByRole('button', { name: /^save/i }) as HTMLButtonElement

/** A new prompt named `leaf` with some text, ready to save. */
async function newNamed(leaf: string) {
  render(PromptEditorPage, { name: '' })
  await waitFor(() => expect(promptText()).toBeTruthy())
  await fireEvent.input(document.querySelector('.savename')!, {
    target: { value: leaf },
  })
  await fireEvent.input(promptText(), { target: { value: 'a new idea' } })
}

it('Cmd+S saves prompt text typed with no blur', async () => {
  await openHero()
  await fireEvent.input(promptText(), { target: { value: 'a hero at noon' } })
  await fireEvent.keyDown(window, { key: 's', metaKey: true })
  await waitFor(() => expect(api.savePrompt).toHaveBeenCalledTimes(1))
  expect(vi.mocked(api.savePrompt).mock.calls[0][1]).toEqual({
    text: 'a hero at noon',
  })
})

it('a new prompt saved under an existing name asks before replacing', async () => {
  vi.mocked(api.savePrompt).mockRejectedValueOnce(
    new ApiError('hero already exists', 409, 'exists'),
  )
  vi.mocked(chooseDialog).mockResolvedValueOnce('replace')
  await newNamed('hero')
  await fireEvent.click(saveButton())

  await waitFor(() => expect(api.savePrompt).toHaveBeenCalledTimes(2))
  expect(vi.mocked(chooseDialog).mock.calls[0][0]).toBe(
    'hero already exists. Replace it?',
  )
  expect(vi.mocked(api.savePrompt).mock.calls[0][2]).toEqual({
    createOnly: true,
  })
  // Replace writes again with no precondition
  expect(vi.mocked(api.savePrompt).mock.calls[1][2]).toEqual({})
})

it('a failed load shows Retry, disables Save and never saves', async () => {
  vi.mocked(api.getPrompt).mockRejectedValueOnce(
    new ApiError('server exploded', 500),
  )
  render(PromptEditorPage, { name: 'portraits/hero' })
  expect(
    await screen.findByText(
      "Couldn't open portraits/hero: The server hit an error",
    ),
  ).toBeTruthy()
  expect(saveButton().disabled).toBe(true)
  await fireEvent.keyDown(window, { key: 's', ctrlKey: true })
  await new Promise((r) => setTimeout(r, 0))
  expect(api.savePrompt).not.toHaveBeenCalled()

  await fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await waitFor(() => expect(promptText().value).toBe('a hero at dawn'))
  expect(saveButton().disabled).toBe(false)
})

it('a 404 offers to start a new prompt with that name', async () => {
  vi.mocked(api.getPrompt).mockRejectedValueOnce(
    new ApiError('Prompt not found', 404),
  )
  render(PromptEditorPage, { name: 'portraits/missing' })
  expect(
    await screen.findByText(
      "Couldn't open portraits/missing: That wasn't found",
    ),
  ).toBeTruthy()
  await fireEvent.click(
    screen.getByRole('button', {
      name: 'Start a new one named portraits/missing',
    }),
  )
  await waitFor(() => expect(promptText()).toBeTruthy())
  await fireEvent.input(promptText(), { target: { value: 'found it' } })
  await fireEvent.click(saveButton())
  await waitFor(() => expect(api.savePrompt).toHaveBeenCalledTimes(1))
  expect(vi.mocked(api.savePrompt).mock.calls[0][0]).toBe('portraits/missing')
  expect(vi.mocked(api.savePrompt).mock.calls[0][2]).toEqual({
    createOnly: true,
  })
})

it('the URL and heading move to the saved name after the first save', async () => {
  location.hash = '#/shared/prompt-edit'
  await new Promise((r) => setTimeout(r, 0))
  const entries = history.length
  await newNamed('fresh')
  expect(screen.getByRole('heading', { level: 1 }).textContent).toBe(
    'New prompt',
  )
  await fireEvent.click(saveButton())

  await waitFor(() => expect(location.hash).toBe('#/shared/prompt-edit/fresh'))
  expect(history.length).toBe(entries)
  await waitFor(() =>
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('fresh'),
  )
})

it('the tags field keeps what is being typed while it applies', async () => {
  await openHero()
  const tags = document.getElementById('prompt-tags') as HTMLInputElement
  await fireEvent.input(tags, { target: { value: 'hero,' } })
  // mid-word: the trailing comma stays in the field
  expect(tags.value).toBe('hero,')
  await fireEvent.input(tags, { target: { value: 'hero, dawn' } })
  await fireEvent.keyDown(window, { key: 's', ctrlKey: true })
  await waitFor(() => expect(api.savePrompt).toHaveBeenCalledTimes(1))
  expect(vi.mocked(api.savePrompt).mock.calls[0][1]).toMatchObject({
    tags: ['hero', 'dawn'],
  })
})

it('going back to a prompt after another failed to open shows the prompt', async () => {
  const { rerender } = render(PromptEditorPage, { name: 'portraits/hero' })
  await waitFor(() => expect(promptText().value).toBe('a hero at dawn'))
  vi.mocked(api.getPrompt).mockRejectedValueOnce(
    new ApiError('server exploded', 500),
  )
  await rerender({ name: 'portraits/other' })
  await screen.findByText(
    "Couldn't open portraits/other: The server hit an error",
  )
  await rerender({ name: 'portraits/hero' })
  await waitFor(() => expect(promptText().value).toBe('a hero at dawn'))
  expect(screen.queryByText(/Couldn't open/)).toBeNull()
  expect(saveButton().disabled).toBe(false)
})

it('going back to a prompt while another is still being read shows the prompt', async () => {
  const { rerender } = render(PromptEditorPage, { name: 'portraits/hero' })
  await waitFor(() => expect(promptText().value).toBe('a hero at dawn'))
  let finishOther: (v: unknown) => void = () => {}
  vi.mocked(api.getPrompt).mockImplementationOnce(
    () => new Promise((r) => (finishOther = r)) as never,
  )
  await rerender({ name: 'portraits/other' })
  await rerender({ name: 'portraits/hero' })
  await waitFor(() => expect(saveButton().disabled).toBe(false))
  finishOther({ prompt: { text: 'late' }, writable: true, etag: '"p9"' })
  await new Promise((r) => setTimeout(r, 0))
  expect(promptText().value).toBe('a hero at dawn')
  expect(saveButton().disabled).toBe(false)
})

it('Load theirs replaces the edits', async () => {
  await openHero()
  await fireEvent.input(promptText(), { target: { value: 'my edit' } })
  vi.mocked(api.getPrompt).mockResolvedValueOnce({
    prompt: { text: 'their version' },
    writable: true,
    etag: '"p9"',
  } as never)
  feed.move('default', 'prompts')
  const banner = await screen.findByText(
    'portraits/hero changed since you opened it, in another tab or by an agent',
  )
  expect(banner.closest('[role="status"]')).toBeTruthy()
  expect(promptText().value).toBe('my edit')
  await fireEvent.click(screen.getByRole('button', { name: 'Load theirs' }))
  await waitFor(() => expect(promptText().value).toBe('their version'))
  expect(screen.queryByRole('button', { name: 'Load theirs' })).toBeNull()
})
