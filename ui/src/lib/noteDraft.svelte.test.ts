import { flushSync } from 'svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { NoteDraft, noteKey, type NoteDraftOptions } from './noteDraft.svelte'

const confirmDialog = vi.hoisted(() => vi.fn())
vi.mock('./confirm.svelte', () => ({ confirmDialog }))

let teardowns: (() => void)[] = []
beforeEach(() => {
  sessionStorage.clear()
  confirmDialog.mockReset()
})
afterEach(() => {
  teardowns.forEach((t) => t())
  teardowns = []
})

/** A draft made inside an effect root, as a component's setup makes one */
function make(over: Partial<NoteDraftOptions> = {}) {
  let draft!: NoteDraft
  teardowns.push(
    $effect.root(() => {
      draft = new NoteDraft({
        key: noteKey('studio', 'wf/run-1'),
        saved: () => 'saved',
        save: vi.fn().mockResolvedValue(undefined),
        ...over,
      })
    }),
  )
  flushSync()
  return draft
}
const key = (e: Partial<KeyboardEventInit>) => new KeyboardEvent('keydown', e)

it('a draft survives a reload', () => {
  const first = make()
  first.open()
  first.text = 'half a thought'
  flushSync()
  expect(sessionStorage.getItem('note:studio:wf/run-1')).toBe('half a thought')
  teardowns.pop()?.()
  // the page comes back: the draft is set aside, and opening resumes it
  const again = make()
  expect(again.kept).toBe(true)
  again.open()
  expect(again.text).toBe('half a thought')
  // saved: nothing is left behind
  expect(again.done('half a thought')).toBe(true)
  flushSync()
  expect(sessionStorage.getItem('note:studio:wf/run-1')).toBeNull()
})

it('Ctrl+Enter saves', () => {
  const save = vi.fn().mockResolvedValue(undefined)
  const draft = make({ save })
  draft.open()
  draft.text = 'warmer'
  draft.keydown(key({ key: 'Enter', ctrlKey: true }))
  draft.keydown(key({ key: 'Enter', metaKey: true }))
  expect(save).toHaveBeenNthCalledWith(1, 'warmer')
  expect(save).toHaveBeenCalledTimes(2)
  draft.keydown(key({ key: 'Enter' }))
  expect(save).toHaveBeenCalledTimes(2)
})

it('Escape keeps the draft', () => {
  const onclose = vi.fn()
  const draft = make({ onclose })
  draft.open()
  draft.text = 'half'
  draft.keydown(key({ key: 'Escape' }))
  flushSync()
  expect(draft.editing).toBe(false)
  expect(draft.kept).toBe(true)
  expect(onclose).toHaveBeenCalled()
  expect(confirmDialog).not.toHaveBeenCalled()
  expect(sessionStorage.getItem('note:studio:wf/run-1')).toBe('half')
  draft.open()
  expect(draft.text).toBe('half')
})

it('cancel asks only when the draft differs', async () => {
  const draft = make()
  draft.open()
  expect(draft.text).toBe('saved')
  expect(await draft.cancel()).toBe(true)
  expect(confirmDialog).not.toHaveBeenCalled()

  draft.open()
  draft.text = 'changed'
  confirmDialog.mockResolvedValue(false)
  expect(await draft.cancel()).toBe(false)
  expect(confirmDialog).toHaveBeenCalledWith(
    'Discard this draft of your note?',
    { confirmLabel: 'Discard', cancelLabel: 'Keep editing', danger: true },
  )
  expect(draft.editing).toBe(true)
  expect(draft.text).toBe('changed')
  confirmDialog.mockResolvedValue(true)
  expect(await draft.cancel()).toBe(true)
  expect(draft.editing).toBe(false)
  draft.open()
  expect(draft.text).toBe('saved')
})

it('counts the characters only within 10% of the limit', () => {
  const draft = make()
  draft.text = 'x'.repeat(3599)
  expect(draft.counter).toBeNull()
  draft.text = 'x'.repeat(3600)
  expect(draft.counter).toBe('3600 / 4000')
})

it('a new key is another note: its own draft is read, the old one kept', () => {
  sessionStorage.setItem('note:studio:b', 'on b')
  let target = $state('a')
  const draft = make({ key: () => noteKey('studio', target) })
  draft.open()
  draft.text = 'on a'
  flushSync()
  target = 'b'
  flushSync()
  expect(draft.editing).toBe(false)
  expect(draft.kept).toBe(true)
  expect(sessionStorage.getItem('note:studio:a')).toBe('on a')
  draft.open()
  expect(draft.text).toBe('on b')
})

it('words typed while a save is out stay, in the box and in storage', () => {
  const draft = make()
  draft.open()
  draft.text = 'warmer'
  const sent = draft.text
  // the PUT is out; the box still takes typing
  draft.text = 'warmer light'
  flushSync()
  // the save lands with what was sent: the rest is not lost
  expect(draft.done(sent)).toBe(false)
  flushSync()
  expect(draft.editing).toBe(true)
  expect(draft.text).toBe('warmer light')
  expect(sessionStorage.getItem('note:studio:wf/run-1')).toBe('warmer light')
})

it('Escape while a save is out keeps words typed after the press', () => {
  const draft = make()
  draft.open()
  draft.text = 'warmer'
  const sent = draft.text
  draft.text = 'warmer light'
  draft.keydown(key({ key: 'Escape' }))
  expect(draft.done(sent)).toBe(false)
  flushSync()
  expect(draft.editing).toBe(false)
  expect(draft.kept).toBe(true)
  expect(sessionStorage.getItem('note:studio:wf/run-1')).toBe('warmer light')
  draft.open()
  expect(draft.text).toBe('warmer light')
})

it('Escape while a save is out, with nothing typed since, leaves nothing', () => {
  const draft = make()
  draft.open()
  draft.text = 'warmer'
  draft.keydown(key({ key: 'Escape' }))
  expect(draft.done('warmer')).toBe(true)
  flushSync()
  expect(draft.kept).toBe(false)
  expect(sessionStorage.getItem('note:studio:wf/run-1')).toBeNull()
})

it('a draft equal to the saved note is not kept', () => {
  let saved = $state('saved')
  const draft = make({ saved: () => saved })
  draft.open()
  draft.text = 'warmer light'
  draft.keydown(key({ key: 'Escape' }))
  expect(draft.kept).toBe(true)
  // the save went through while the box was closed
  saved = 'warmer light'
  flushSync()
  expect(draft.kept).toBe(false)
  draft.open()
  expect(draft.text).toBe('warmer light')
})
