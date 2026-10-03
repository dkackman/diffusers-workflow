// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { cleanup, render } from '@testing-library/svelte'
import { flushSync } from 'svelte'
import EditorShellHarness from './EditorShellHarness.svelte'
import type { DocumentEditor, EditorView } from './editorShell.svelte'

vi.mock('./toast', () => ({
  notify: { error: vi.fn(), success: vi.fn(), dismiss: vi.fn() },
}))

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
    onready: (s: DocumentEditor) => (shell = s),
  })
  flushSync()
  return shell!
}

beforeEach(() => {
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
