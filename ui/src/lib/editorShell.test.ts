// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render } from '@testing-library/svelte'
import { flushSync } from 'svelte'
import EditorShellHarness from './EditorShellHarness.svelte'
import type { DocumentEditor, EditorView } from './editorShell.svelte'
import { notify } from './toast'
import { addLeaveGuard } from './router.svelte'
import { confirmDialog } from './confirm.svelte'
import { changeOnBlur } from './test/changeOnBlur'

vi.mock('./toast', () => ({
  notify: { error: vi.fn(), success: vi.fn(), dismiss: vi.fn() },
}))
vi.mock('./router.svelte', () => ({ addLeaveGuard: vi.fn(() => vi.fn()) }))
vi.mock('./confirm.svelte', () => ({ confirmDialog: vi.fn() }))

const KEY = 'test-editor-view'

function mount(
  views: readonly EditorView[] = ['form', 'split', 'json'],
  legacyView?: () => EditorView | null,
): DocumentEditor {
  let shell: DocumentEditor | undefined
  render(EditorShellHarness, {
    viewKey: KEY,
    views,
    legacyView,
    label: () => 'portraits/hero',
    onready: (s: DocumentEditor) => (shell = s),
  })
  flushSync()
  return shell!
}

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  sessionStorage.clear()
})
afterEach(() => cleanup())

it('opens on the stored view and stores the one picked', () => {
  localStorage.setItem(KEY, 'json')
  const shell = mount()
  expect(shell.view).toBe('json')
  shell.setView('split')
  expect(shell.view).toBe('split')
  expect(localStorage.getItem(KEY)).toBe('split')
})

it('falls back to the legacy preference, then the first view', () => {
  localStorage.setItem(KEY, 'flow')
  expect(mount(['form', 'split', 'json'], () => 'split').view).toBe('split')
  cleanup()
  expect(mount(['form', 'split', 'json'], () => null).view).toBe('form')
})

it('mirrors the document into the JSON draft', () => {
  const shell = mount()
  shell.load({ text: 'a' })
  flushSync()
  expect(JSON.parse(shell.jsonDraft)).toEqual({ text: 'a' })
  shell.doc.text = 'b'
  flushSync()
  expect(JSON.parse(shell.jsonDraft)).toEqual({ text: 'b' })
})

it('pins a JSON draft that does not parse, and leaves the document alone', () => {
  const shell = mount()
  shell.load({ text: 'a' })
  flushSync()
  shell.applyJson('{ broken')
  flushSync()
  expect(shell.jsonParseFailed).toBe(true)
  expect(shell.jsonDraft).toBe('{ broken')
  expect(shell.doc).toEqual({ text: 'a' })
  shell.applyJson('{"text": "c"}')
  flushSync()
  expect(shell.jsonParseFailed).toBe(false)
  expect(shell.doc).toEqual({ text: 'c' })
})

it('is dirty between an edit and the save that records it', () => {
  const shell = mount()
  shell.load({ text: 'a' })
  expect(shell.dirty).toBe(false)
  shell.doc.text = 'b'
  expect(shell.dirty).toBe(true)
  shell.markSaved()
  expect(shell.dirty).toBe(false)
})

it('builds the save path from the folder, a new folder and the name', () => {
  const shell = mount()
  expect(shell.savePath()).toBeNull()
  shell.saveName = 'hero'
  expect(shell.savePath()).toBe('hero')
  shell.folder = 'portraits'
  expect(shell.savePath()).toBe('portraits/hero')
  shell.folder = '__new__'
  shell.newFolder = ' fresh '
  expect(shell.directory()).toBe('fresh')
  expect(shell.savePath()).toBe('fresh/hero')
  shell.newFolder = 'not/one'
  expect(shell.savePath()).toBeNull()
})

it('turns a new folder into the picked one once it exists', () => {
  const shell = mount()
  shell.folder = '__new__'
  shell.newFolder = ' fresh '
  shell.commitNewFolder()
  expect(shell.folder).toBe('fresh')
  expect(shell.newFolder).toBe('')
  shell.commitNewFolder()
  expect(shell.folder).toBe('fresh')
})

it('takes a one-shot hand-off from session storage', () => {
  sessionStorage.setItem('imp', '{"text": "copied"}')
  sessionStorage.setItem('fold', 'portraits')
  const shell = mount()
  expect(shell.takeImport('imp', 'fold')).toEqual({ text: 'copied' })
  expect(shell.folder).toBe('portraits')
  expect(sessionStorage.getItem('imp')).toBeNull()
  expect(sessionStorage.getItem('fold')).toBeNull()
  expect(shell.takeImport('imp', 'fold')).toBeNull()
  expect(shell.folder).toBe('')
})

it('drops an unreadable hand-off', () => {
  sessionStorage.setItem('imp', '{ broken')
  const shell = mount()
  expect(shell.takeImport('imp', 'fold')).toBeNull()
  expect(sessionStorage.getItem('imp')).toBeNull()
})

it('guards a tab close only while dirty', () => {
  const shell = mount()
  shell.load({ text: 'a' })
  flushSync()
  const unload = () => {
    const event = new Event('beforeunload', { cancelable: true })
    window.dispatchEvent(event)
    return event.defaultPrevented
  }
  expect(unload()).toBe(false)
  shell.doc.text = 'b'
  expect(unload()).toBe(true)
})

it('flush runs every registered flusher and a removed one is not run', () => {
  const shell = mount()
  const a = vi.fn()
  const b = vi.fn()
  shell.registerFlush(a)
  const removeB = shell.registerFlush(b)
  shell.flush()
  expect(a).toHaveBeenCalledTimes(1)
  expect(b).toHaveBeenCalledTimes(1)
  removeB()
  shell.flush()
  expect(a).toHaveBeenCalledTimes(2)
  expect(b).toHaveBeenCalledTimes(1)
})

it('a failed load cannot write', () => {
  const shell = mount()
  expect(shell.loadState).toBe('new')
  expect(shell.canWrite).toBe(true)
  shell.beginLoad()
  expect(shell.loadState).toBe('loading')
  expect(shell.canWrite).toBe(false)
  const error = new Error('gone')
  shell.failLoad(error)
  expect(shell.loadState).toBe('failed')
  expect(shell.loadError).toBe(error)
  expect(shell.canWrite).toBe(false)
  shell.startNew({ text: '' })
  expect(shell.loadState).toBe('new')
  expect(shell.canWrite).toBe(true)
})

it('preconditions: if-match for the loaded writable path, create-only for a new path', () => {
  const shell = mount()
  expect(shell.preconditions('hero')).toEqual({ createOnly: true })
  shell.load({ text: 'a' }, { etag: '"abc"', path: 'hero', writable: true })
  expect(shell.loadState).toBe('loaded')
  expect(shell.etag).toBe('"abc"')
  expect(shell.loadedPath).toBe('hero')
  expect(shell.preconditions('hero')).toEqual({ ifMatch: '"abc"' })
  expect(shell.preconditions('other')).toEqual({ createOnly: true })
})

it('a document from a read-only origin saves create-only', () => {
  const shell = mount()
  shell.load({ text: 'a' }, { etag: '"abc"', path: 'hero', writable: false })
  expect(shell.preconditions('hero')).toEqual({ createOnly: true })
})

it('setView flushes, and refuses to leave JSON while it does not parse', () => {
  const shell = mount()
  shell.load({ text: 'a' })
  const flusher = vi.fn()
  shell.registerFlush(flusher)
  shell.setView('json')
  expect(flusher).toHaveBeenCalledTimes(1)
  shell.applyJson('{ broken')
  shell.setView('form')
  expect(flusher).toHaveBeenCalledTimes(2)
  expect(shell.view).toBe('json')
  expect(notify.error).toHaveBeenLastCalledWith(
    'Fix the JSON before switching view',
    'json-parse',
  )
  shell.applyJson('{"text": "b"}')
  shell.setView('form')
  expect(shell.view).toBe('form')
})

it('a quiet parse failure does not toast', () => {
  const shell = mount()
  shell.applyJson('{ broken', { quiet: true })
  expect(shell.jsonParseFailed).toBe(true)
  expect(notify.error).not.toHaveBeenCalled()
})

it('asks before leaving while dirty', async () => {
  const shell = mount()
  shell.load({ text: 'a' })
  flushSync()
  const guard = vi.mocked(addLeaveGuard).mock.calls[0][0]
  expect(await guard()).toBe(true)
  expect(confirmDialog).not.toHaveBeenCalled()
  shell.doc.text = 'b'
  vi.mocked(confirmDialog).mockResolvedValueOnce(false)
  expect(await guard()).toBe(false)
  expect(confirmDialog).toHaveBeenCalledWith(
    'Leave without saving your changes to portraits/hero?',
    { confirmLabel: 'Leave', cancelLabel: 'Keep editing', danger: true },
  )
  vi.mocked(confirmDialog).mockResolvedValueOnce(true)
  expect(await guard()).toBe(true)
})

it('removes its leave guard when the component is destroyed', () => {
  mount()
  const remove = vi.mocked(addLeaveGuard).mock.results[0].value
  cleanup()
  expect(remove).toHaveBeenCalled()
})

/** A field inside the editor's root that commits on change, as the form
 * view's fields do, focused with `typed` in it and no blur yet. */
function typedField(shell: DocumentEditor, typed: string) {
  const root = document.createElement('div')
  const field = document.createElement('input')
  field.addEventListener('change', () => (shell.doc.text = field.value))
  root.append(field)
  document.body.append(root)
  shell.root = root
  field.focus()
  field.value = typed
  return { root, field }
}

describe('flush and the leave guard see every edit', () => {
  let uninstall = () => {}
  beforeEach(() => (uninstall = changeOnBlur()))
  afterEach(() => {
    uninstall()
    document.body.replaceChildren()
  })

  it('flush commits the field still focused inside the editor', async () => {
    const shell = mount()
    shell.load({ text: 'a' })
    const { field } = typedField(shell, 'typed')
    shell.flush()
    expect(shell.doc.text).toBe('typed')
    // the caret stays where the user was typing
    await vi.waitFor(() => expect(document.activeElement).toBe(field))
  })

  it('flush does not blur a field mid-composition', () => {
    const shell = mount()
    shell.load({ text: 'a' })
    const { field } = typedField(shell, 'typed')
    const flusher = vi.fn()
    shell.registerFlush(flusher)
    const blurred = vi.fn()
    field.addEventListener('blur', blurred)
    field.dispatchEvent(new Event('compositionstart', { bubbles: true }))
    shell.flush()
    expect(blurred).not.toHaveBeenCalled()
    expect(flusher).toHaveBeenCalledTimes(1)
    field.dispatchEvent(new Event('compositionend', { bubbles: true }))
    shell.flush()
    expect(blurred).toHaveBeenCalledTimes(1)
  })

  it('flush restores focus to a field the commit re-rendered', async () => {
    const shell = mount()
    shell.load({ text: 'a' })
    const root = document.createElement('div')
    const field = document.createElement('input')
    field.id = 'title'
    // the blur commit re-renders the field as a new node with the same id
    field.addEventListener('blur', () => {
      const next = document.createElement('input')
      next.id = 'title'
      next.value = field.value
      field.replaceWith(next)
    })
    root.append(field)
    document.body.append(root)
    shell.root = root
    field.focus()
    field.value = 'typed'
    field.setSelectionRange(2, 4)

    shell.flush()

    const now = root.querySelector('input') as HTMLInputElement
    expect(now).not.toBe(field)
    await vi.waitFor(() => expect(document.activeElement).toBe(now))
    expect([now.selectionStart, now.selectionEnd]).toEqual([2, 4])
  })

  it('flush leaves a field outside the editor focused and alone', () => {
    const shell = mount()
    shell.load({ text: 'a' })
    shell.root = document.createElement('div')
    const outside = document.createElement('input')
    const blurred = vi.fn()
    outside.addEventListener('blur', blurred)
    document.body.append(outside)
    outside.focus()
    shell.flush()
    expect(blurred).not.toHaveBeenCalled()
    expect(document.activeElement).toBe(outside)
  })

  it('the leave guard asks for an edit still in a focused field', async () => {
    const shell = mount()
    shell.load({ text: 'a' })
    flushSync()
    typedField(shell, 'typed')
    const guard = vi.mocked(addLeaveGuard).mock.calls[0][0]
    vi.mocked(confirmDialog).mockResolvedValueOnce(false)
    expect(await guard()).toBe(false)
    expect(confirmDialog).toHaveBeenCalledTimes(1)
  })

  it('the leave guard asks for a pending debounced edit', async () => {
    const shell = mount()
    shell.load({ text: 'a' })
    flushSync()
    shell.registerFlush(() => (shell.doc.text = 'pending'))
    const guard = vi.mocked(addLeaveGuard).mock.calls[0][0]
    vi.mocked(confirmDialog).mockResolvedValueOnce(false)
    expect(await guard()).toBe(false)
    expect(confirmDialog).toHaveBeenCalledTimes(1)
  })
})

it('a JSON draft that does not parse is guarded on leave and on tab close', async () => {
  const shell = mount()
  shell.load({ text: 'a' })
  flushSync()
  shell.applyJson('{ broken', { quiet: true })
  expect(shell.dirty).toBe(false)
  const guard = vi.mocked(addLeaveGuard).mock.calls[0][0]
  vi.mocked(confirmDialog).mockResolvedValueOnce(false)
  expect(await guard()).toBe(false)
  expect(confirmDialog).toHaveBeenCalledTimes(1)
  const event = new Event('beforeunload', { cancelable: true })
  window.dispatchEvent(event)
  expect(event.defaultPrevented).toBe(true)
})

it.each([
  ['load', (shell: DocumentEditor) => shell.load({ text: 'b' })],
  ['startNew', (shell: DocumentEditor) => shell.startNew({ text: 'b' })],
])("%s drops the last document's broken JSON draft", (_, open) => {
  const shell = mount()
  shell.load({ text: 'a' })
  flushSync()
  shell.applyJson('{ broken')
  open(shell)
  flushSync()
  expect(shell.jsonParseFailed).toBe(false)
  expect(JSON.parse(shell.jsonDraft)).toEqual({ text: 'b' })
  expect(notify.dismiss).toHaveBeenLastCalledWith('json-parse')
})
