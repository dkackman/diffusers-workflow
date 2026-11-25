// @vitest-environment jsdom
import { afterEach, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/svelte'
import { flushSync } from 'svelte'
import EditorShellHarness from '../EditorShellHarness.svelte'
import DocumentFileBar from './DocumentFileBar.svelte'
import type { DocumentEditor } from '../editorShell.svelte'

afterEach(cleanup)

function bar(noun: 'workflow' | 'prompt', props: Record<string, unknown> = {}) {
  let ed: DocumentEditor | undefined
  render(EditorShellHarness, {
    viewKey: 'bar-test',
    views: ['form'],
    label: () => 'doc',
    onready: (s: DocumentEditor) => (ed = s),
  })
  ed!.startNew({})
  ed!.folder = 'portraits'
  ed!.saveName = 'hero'
  render(DocumentFileBar, {
    ed: ed!,
    dir: '/lib',
    folders: ['portraits'],
    noun,
    ...props,
  })
  flushSync()
}

it('shows the folder, the name and the path for either document', () => {
  for (const [noun, hint] of [
    ['workflow', 'MyWorkflow'],
    ['prompt', 'MyPrompt'],
  ] as const) {
    bar(noun)
    expect((screen.getByLabelText('folder') as HTMLSelectElement).value).toBe(
      'portraits',
    )
    const name = screen.getByLabelText('file name') as HTMLInputElement
    expect(name.value).toBe('hero')
    expect(name.placeholder).toBe(hint)
    expect(screen.getByText('/lib/portraits/hero.json')).toBeTruthy()
    cleanup()
  }
})

it('offers the reference to copy when it has one', () => {
  bar('prompt', { reference: 'prompt:portraits/hero', described: false })
  expect(screen.getByText('prompt:portraits/hero')).toBeTruthy()
  expect(screen.queryByLabelText('description')).toBeNull()
})

const BANNER =
  "This is an example, so it can't be changed in place. Save makes your own copy in default."

it('an example says Save makes your own copy', () => {
  for (const noun of ['workflow', 'prompt'] as const) {
    let ed: DocumentEditor | undefined
    render(EditorShellHarness, {
      viewKey: 'bar-test',
      views: ['form'],
      label: () => 'doc',
      onready: (s: DocumentEditor) => (ed = s),
    })
    ed!.load({}, { writable: false, path: 'models/z-image' })
    render(DocumentFileBar, { ed: ed!, dir: '/lib', folders: [], noun })
    flushSync()
    expect(screen.getByText(BANNER)).toHaveClass('hint')
    cleanup()
  }
})

it('a workspace file says nothing', () => {
  bar('workflow')
  expect(screen.queryByText(/This is an example/)).toBeNull()
})
