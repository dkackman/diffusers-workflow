import { flushSync } from 'svelte'
import { afterEach, expect, it } from 'vitest'
import { missingFile } from './missingFile.svelte'
import { DEFAULT_WORKSPACE, workspace } from './workspaceState.svelte'

// A page's view as the helper reads it: what open= names, whether the
// listing is in, and the names it holds
const view = $state({
  open: null as string | null,
  known: false,
  names: [] as string[],
})
let teardown = () => {}

function track() {
  let stray!: ReturnType<typeof missingFile>
  teardown = $effect.root(() => {
    stray = missingFile({
      open: () => view.open,
      known: () => view.known,
      names: () => view.names,
    })
  })
  flushSync()
  return stray
}

afterEach(() => {
  teardown()
  view.open = null
  view.known = false
  view.names = []
  workspace.current = DEFAULT_WORKSPACE
})

it('a stray open= is said once the listing is in, by its leaf', () => {
  view.open = 'run1/gone.png'
  view.names = ['run1/a.png']
  const stray = track()
  // not known yet: the file may still arrive
  expect(stray.missing).toBeNull()
  view.known = true
  flushSync()
  expect(stray.missing).toBe('gone.png')
})

it('an open= the view lists is never said', () => {
  view.open = 'run1/a.png'
  view.names = ['run1/a.png']
  view.known = true
  const stray = track()
  expect(stray.missing).toBeNull()
  // one that was listed and left is the viewer's to step on from
  view.names = []
  flushSync()
  expect(stray.missing).toBeNull()
})

it('stays said once the page clears open=', () => {
  view.open = 'gone.png'
  view.known = true
  const stray = track()
  view.open = null
  flushSync()
  expect(stray.missing).toBe('gone.png')
})

it('dismiss clears it', () => {
  view.open = 'gone.png'
  view.known = true
  const stray = track()
  stray.dismiss()
  flushSync()
  expect(stray.missing).toBeNull()
})

it('a workspace change resets it', () => {
  view.open = 'gone.png'
  view.known = true
  const stray = track()
  view.open = null
  flushSync()
  workspace.current = 'studio'
  flushSync()
  expect(stray.missing).toBeNull()
})

it('a file that arrives later clears it', () => {
  view.open = 'run1/late.png'
  view.known = true
  const stray = track()
  view.open = null
  flushSync()
  expect(stray.missing).toBe('late.png')
  view.names = ['run1/late.png']
  flushSync()
  expect(stray.missing).toBeNull()
})

it('another stray open= takes its place', () => {
  view.open = 'a/one.png'
  view.known = true
  const stray = track()
  view.open = 'b/two.png'
  flushSync()
  expect(stray.missing).toBe('two.png')
})
