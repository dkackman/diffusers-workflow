import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { tick } from 'svelte'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../api'
import { ApiError } from '../apiErrors'
import { confirmDialog } from '../confirm.svelte'
import { notify } from '../toast'
import NoteEditor from './NoteEditor.svelte'
import { quiet, useQuietClock } from '../test/quiet'

useQuietClock()

vi.mock('../confirm.svelte', () => ({ confirmDialog: vi.fn() }))

describe('NoteEditor', () => {
  /** What GET /api/notes answers: the server holds `note`, etag `e:<note>` */
  const holds = (note: string | null) =>
    vi.spyOn(api.api, 'noteForSave').mockImplementation(async (target) => ({
      target,
      note,
      etag: `e:${note ?? ''}`,
    }))
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.mocked(confirmDialog).mockReset()
    sessionStorage.clear()
    holds('old')
  })
  afterEach(cleanup)

  it('shows the note as text, never markup', () => {
    render(NoteEditor, {
      props: { note: '<b>bold</b>', target: 'workspace:default' },
    })
    expect(screen.getByText('<b>bold</b>')).toBeTruthy()
    expect(document.querySelector('b')).toBeNull()
  })

  it('shows the placeholder when empty', () => {
    render(NoteEditor, { props: { note: null, target: 'workspace:default' } })
    expect(screen.getByText('Add a note')).toBeTruthy()
  })

  it('saves through setNote and reports the stored note', async () => {
    const spy = vi
      .spyOn(api, 'setNote')
      .mockResolvedValue({ target: 't', note: 'new', note_source: 'run' })
    const onsaved = vi.fn()
    render(NoteEditor, { props: { note: 'old', target: 't', onsaved } })
    await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
    await fireEvent.input(screen.getByRole('textbox'), {
      target: { value: 'new' },
    })
    await fireEvent.click(screen.getByRole('button', { name: /save/i }))
    await waitFor(() =>
      expect(spy).toHaveBeenCalledWith('t', 'new', undefined, 'e:old'),
    )
    await waitFor(() => expect(onsaved).toHaveBeenCalledWith('new'))
  })

  it('words typed while the save is out stay as a draft over the saved note', async () => {
    let land!: (v: {
      target: string
      note: string
      note_source: string
    }) => void
    const spy = vi
      .spyOn(api, 'setNote')
      .mockReturnValue(new Promise((resolve) => (land = resolve)) as never)
    const onsaved = vi.fn()
    render(NoteEditor, { props: { note: 'old', target: 't', onsaved } })
    await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
    const box = screen.getByRole('textbox')
    await fireEvent.input(box, { target: { value: 'new' } })
    await fireEvent.click(screen.getByRole('button', { name: /save/i }))
    await waitFor(() =>
      expect(spy).toHaveBeenCalledWith('t', 'new', undefined, 'e:old'),
    )
    // the PUT is out; the user keeps typing
    await fireEvent.input(box, { target: { value: 'new and more' } })
    land({ target: 't', note: 'new', note_source: 'run' })
    await waitFor(() => expect(onsaved).toHaveBeenCalledWith('new'))
    expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(
      'new and more',
    )
    // a draft over the note now saved, not a newer note to take in
    expect(screen.queryByText(/changed while you were editing/)).toBeNull()
    expect(Object.values({ ...sessionStorage })).toContain('new and more')
  })

  it('Escape while the save is out keeps the words typed since', async () => {
    let land!: (v: {
      target: string
      note: string
      note_source: string
    }) => void
    vi.spyOn(api, 'setNote').mockReturnValue(
      new Promise((resolve) => (land = resolve)) as never,
    )
    const onsaved = vi.fn()
    render(NoteEditor, { props: { note: 'old', target: 't', onsaved } })
    await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
    const box = screen.getByRole('textbox')
    await fireEvent.input(box, { target: { value: 'new' } })
    await fireEvent.click(screen.getByRole('button', { name: /save/i }))
    await fireEvent.input(box, { target: { value: 'new and more' } })
    await fireEvent.keyDown(box, { key: 'Escape' })
    land({ target: 't', note: 'new', note_source: 'run' })
    await waitFor(() => expect(onsaved).toHaveBeenCalledWith('new'))
    expect(Object.values({ ...sessionStorage })).toContain('new and more')
    await fireEvent.click(
      screen.getByRole('button', { name: /continue|edit note/i }),
    )
    expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(
      'new and more',
    )
  })

  it('cancel restores the old note without calling the api', async () => {
    const spy = vi.spyOn(api, 'setNote')
    vi.mocked(confirmDialog).mockResolvedValue(true)
    render(NoteEditor, { props: { note: 'old', target: 't' } })
    await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
    await fireEvent.input(screen.getByRole('textbox'), {
      target: { value: 'changed' },
    })
    await fireEvent.click(screen.getByRole('button', { name: /cancel/i }))
    expect(spy).not.toHaveBeenCalled()
    expect(screen.getByText('old')).toBeTruthy()
  })

  it('saving an empty box clears the note', async () => {
    const spy = vi
      .spyOn(api, 'setNote')
      .mockResolvedValue({ target: 't', note: null, note_source: null })
    render(NoteEditor, { props: { note: 'old', target: 't' } })
    await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
    await fireEvent.input(screen.getByRole('textbox'), {
      target: { value: '' },
    })
    await fireEvent.click(screen.getByRole('button', { name: /save/i }))
    await waitFor(() =>
      expect(spy).toHaveBeenCalledWith('t', null, undefined, 'e:old'),
    )
  })

  it('shows the server error and stays in edit mode', async () => {
    vi.spyOn(api, 'setNote').mockRejectedValue(
      new Error('a note is at most 4000 characters'),
    )
    render(NoteEditor, { props: { note: 'old', target: 't' } })
    await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
    await fireEvent.click(screen.getByRole('button', { name: /save/i }))
    expect(await screen.findByText(/at most 4000/)).toBeTruthy()
    expect(screen.getByRole('textbox')).toBeTruthy()
  })

  it('saves in the workspace it was given', async () => {
    const spy = vi
      .spyOn(api, 'setNote')
      .mockResolvedValue({ target: 't', note: 'new', note_source: 'run' })
    render(NoteEditor, {
      props: { note: 'old', target: 't', workspace: 'ws2' },
    })
    await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
    await fireEvent.input(screen.getByRole('textbox'), {
      target: { value: 'new' },
    })
    await fireEvent.click(screen.getByRole('button', { name: /save/i }))
    await waitFor(() =>
      expect(spy).toHaveBeenCalledWith('t', 'new', 'ws2', 'e:old'),
    )
  })

  it('keeps the draft when the note prop changes mid-edit', async () => {
    const { rerender } = render(NoteEditor, {
      props: { note: 'old', target: 't' },
    })
    await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
    await fireEvent.input(screen.getByRole('textbox'), {
      target: { value: 'half typed' },
    })
    await rerender({ note: 'refetched', target: 't' })
    expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(
      'half typed',
    )
  })

  describe('a newer note arriving mid-edit', () => {
    const NOTICE =
      'Run note changed while you were editing, in another tab or by an agent'
    const stored = { target: 't', note: 'mine', note_source: 'run' as const }

    /** Editing 'old' with 'mine' typed, when an agent's note lands */
    async function overtaken(newer: string | null = 'agent wrote this') {
      const view = render(NoteEditor, {
        props: { note: 'old', target: 't', label: 'Run note' },
      })
      await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
      await fireEvent.input(screen.getByRole('textbox'), {
        target: { value: 'mine' },
      })
      holds(newer)
      await view.rerender({ note: newer, target: 't', label: 'Run note' })
      return view
    }
    const draft = () =>
      (screen.getByRole('textbox') as HTMLTextAreaElement).value

    it('stays up with the newer text until the user acts', async () => {
      vi.useFakeTimers()
      try {
        await overtaken()
        vi.advanceTimersByTime(60_000)
      } finally {
        vi.useRealTimers()
      }
      expect(screen.getByText(NOTICE)).toBeTruthy()
      expect(screen.getByText('agent wrote this')).toBeTruthy()
      expect(draft()).toBe('mine')
      expect(screen.getByRole('button', { name: 'Use theirs' })).toBeTruthy()
      expect(screen.getByRole('button', { name: 'Keep mine' })).toBeTruthy()
    })

    it('Use theirs asks before it throws the draft away, then puts the newer note in the box', async () => {
      await overtaken()
      const theirs = screen.getByRole('button', { name: 'Use theirs' })
      expect(theirs.classList).toContain('danger')
      vi.mocked(confirmDialog).mockResolvedValue(false)
      await fireEvent.click(theirs)
      expect(confirmDialog).toHaveBeenCalledWith(
        'Replace your draft with the newer note?',
        {
          confirmLabel: 'Use theirs',
          cancelLabel: 'Keep editing',
          danger: true,
        },
      )
      expect(draft()).toBe('mine')
      vi.mocked(confirmDialog).mockResolvedValue(true)
      await fireEvent.click(theirs)
      await waitFor(() => expect(draft()).toBe('agent wrote this'))
      expect(screen.queryByText(NOTICE)).toBeNull()
    })

    it('Use theirs over an untouched draft asks nothing', async () => {
      // The page's copy is behind the server: opening the box reads it
      holds('theirs')
      render(NoteEditor, { props: { note: 'old', target: 't' } })
      await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
      await fireEvent.click(
        await screen.findByRole('button', { name: 'Use theirs' }),
      )
      expect(confirmDialog).not.toHaveBeenCalled()
      expect(draft()).toBe('theirs')
    })

    it('Replace with the box switched to another note writes neither', async () => {
      const spy = vi.spyOn(api, 'setNote').mockResolvedValue(stored)
      let answer!: (yes: boolean) => void
      vi.mocked(confirmDialog).mockReturnValue(
        new Promise((resolve) => (answer = resolve)),
      )
      const view = await overtaken()
      await fireEvent.click(screen.getByRole('button', { name: /^save/i }))
      // the user picks another file while the question is up
      await view.rerender({ note: 'other', target: 'u', label: 'Run note' })
      answer(true)
      await quiet()
      expect(spy).not.toHaveBeenCalled()
    })

    it('a second Ctrl+Enter while the first asks sends one write', async () => {
      const spy = vi.spyOn(api, 'setNote').mockResolvedValue(stored)
      let answer!: (yes: boolean) => void
      vi.mocked(confirmDialog).mockReturnValue(
        new Promise((resolve) => (answer = resolve)),
      )
      await overtaken()
      const save = screen.getByRole('button', { name: /^save/i })
      await fireEvent.click(save)
      await fireEvent.keyDown(screen.getByRole('textbox'), {
        key: 'Enter',
        ctrlKey: true,
      })
      expect(save).toHaveProperty('disabled', true)
      answer(true)
      await waitFor(() => expect(spy).toHaveBeenCalledTimes(1))
      expect(confirmDialog).toHaveBeenCalledTimes(1)
    })

    it('a second click while the first asks sends one write', async () => {
      const spy = vi.spyOn(api, 'setNote').mockResolvedValue(stored)
      let answer!: (yes: boolean) => void
      vi.mocked(confirmDialog).mockReturnValue(
        new Promise((resolve) => (answer = resolve)),
      )
      await overtaken()
      const save = screen.getByRole('button', { name: /^save/i })
      // both before the button is redrawn disabled: the guard is what holds
      save.click()
      save.click()
      await tick()
      expect(save).toHaveProperty('disabled', true)
      answer(true)
      await waitFor(() => expect(spy).toHaveBeenCalledTimes(1))
      expect(confirmDialog).toHaveBeenCalledTimes(1)
    })

    it('a note that changed on the server is refused, shown, and never overwritten', async () => {
      const failure = vi.spyOn(notify, 'failure')
      const spy = vi
        .spyOn(api, 'setNote')
        .mockRejectedValue(
          new ApiError('changed since it was read', 409, 'stale_write'),
        )
      render(NoteEditor, {
        props: { note: 'old', target: 't', label: 'Run note' },
      })
      await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
      await fireEvent.input(screen.getByRole('textbox'), {
        target: { value: 'mine' },
      })
      // An agent writes after the box read its etag
      holds('agent wrote this')
      await fireEvent.click(screen.getByRole('button', { name: /^save/i }))
      await waitFor(() =>
        expect(spy).toHaveBeenCalledWith('t', 'mine', undefined, 'e:old'),
      )
      expect(await screen.findByText(NOTICE)).toBeTruthy()
      expect(screen.getByText('agent wrote this')).toBeTruthy()
      expect(draft()).toBe('mine')
      expect(spy).toHaveBeenCalledTimes(1)
      // Shown inline only: the save's action raises no toast of its own
      expect(failure).not.toHaveBeenCalled()
    })

    it('a cleared note says so', async () => {
      await overtaken(null)
      expect(screen.getByText(NOTICE)).toBeTruthy()
      expect(screen.getByText('It is empty now.')).toBeTruthy()
    })

    it('Save over the unseen note asks first, and Keep editing writes nothing', async () => {
      const spy = vi.spyOn(api, 'setNote').mockResolvedValue(stored)
      vi.mocked(confirmDialog).mockResolvedValue(false)
      await overtaken()
      await fireEvent.click(screen.getByRole('button', { name: /^save/i }))
      expect(confirmDialog).toHaveBeenCalledWith(
        'Run note changed while you were editing. Replace it with yours?',
        { confirmLabel: 'Replace', cancelLabel: 'Keep editing', danger: true },
      )
      expect(spy).not.toHaveBeenCalled()
      expect(draft()).toBe('mine')
      expect(screen.getByText(NOTICE)).toBeTruthy()
    })

    it('Replace writes the draft over the newer note', async () => {
      const spy = vi.spyOn(api, 'setNote').mockResolvedValue(stored)
      vi.mocked(confirmDialog).mockResolvedValue(true)
      await overtaken()
      await fireEvent.click(screen.getByRole('button', { name: /^save/i }))
      await waitFor(() =>
        expect(spy).toHaveBeenCalledWith(
          't',
          'mine',
          undefined,
          'e:agent wrote this',
        ),
      )
    })

    it('Keep mine takes the newer note in, so the save asks nothing', async () => {
      const spy = vi.spyOn(api, 'setNote').mockResolvedValue(stored)
      await overtaken()
      await fireEvent.click(screen.getByRole('button', { name: 'Keep mine' }))
      expect(screen.queryByText(NOTICE)).toBeNull()
      expect(draft()).toBe('mine')
      await fireEvent.click(screen.getByRole('button', { name: /^save/i }))
      expect(confirmDialog).not.toHaveBeenCalled()
      await waitFor(() =>
        expect(spy).toHaveBeenCalledWith(
          't',
          'mine',
          undefined,
          'e:agent wrote this',
        ),
      )
    })

    it('Cancel waits for the Replace question', async () => {
      let answer!: (yes: boolean) => void
      vi.mocked(confirmDialog).mockReturnValue(
        new Promise((resolve) => (answer = resolve)),
      )
      vi.spyOn(api, 'setNote').mockResolvedValue(stored)
      await overtaken()
      await fireEvent.click(screen.getByRole('button', { name: /^save/i }))
      const cancel = screen.getByRole('button', { name: 'Cancel' })
      // No discard question stacks on the Replace one
      expect(cancel).toHaveProperty('disabled', true)
      answer(false)
      await waitFor(() => expect(cancel).toHaveProperty('disabled', false))
    })

    it('a note that has not changed saves without asking', async () => {
      vi.spyOn(api, 'setNote').mockResolvedValue(stored)
      render(NoteEditor, { props: { note: 'old', target: 't' } })
      await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
      await fireEvent.click(screen.getByRole('button', { name: /^save/i }))
      expect(confirmDialog).not.toHaveBeenCalled()
    })
  })

  describe('the draft', () => {
    async function typed(text: string, props = { note: 'old', target: 't' }) {
      const view = render(NoteEditor, { props })
      await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
      await fireEvent.input(screen.getByRole('textbox'), {
        target: { value: text },
      })
      return view
    }

    it('Ctrl+Enter saves a file note', async () => {
      const spy = vi.spyOn(api, 'setNote').mockResolvedValue({
        target: 'wf/r/a.png',
        note: 'k',
        note_source: 'file',
      })
      await typed('keeper', { note: 'old', target: 'wf/r/a.png' })
      await fireEvent.keyDown(screen.getByRole('textbox'), {
        key: 'Enter',
        ctrlKey: true,
      })
      await waitFor(() =>
        expect(spy).toHaveBeenCalledWith(
          'wf/r/a.png',
          'keeper',
          undefined,
          'e:old',
        ),
      )
    })

    it('a half-written file note is there after a remount', async () => {
      const props = { note: 'old', target: 'wf/r/a.png', workspace: 'studio' }
      const first = await typed('half a thought', props)
      first.unmount()
      render(NoteEditor, { props })
      expect(screen.getByText('You have an unsaved draft.')).toBeTruthy()
      await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
      expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(
        'half a thought',
      )
      expect(sessionStorage.getItem('note:studio:wf/r/a.png')).toBe(
        'half a thought',
      )
    })

    it('a resumed draft over a newer note asks', async () => {
      const spy = vi.spyOn(api, 'setNote')
      const props = { note: 'old', target: 't', workspace: 'studio' }
      const first = await typed('mine', props)
      first.unmount()
      // An agent writes a newer note before the page comes back
      holds('agent wrote this')
      render(NoteEditor, { props: { ...props, note: 'agent wrote this' } })
      await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
      expect(
        screen.getByText(
          'Note changed while you were editing, in another tab or by an agent',
        ),
      ).toBeTruthy()
      expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(
        'mine',
      )
      vi.mocked(confirmDialog).mockResolvedValue(false)
      await fireEvent.click(screen.getByRole('button', { name: /^save/i }))
      expect(confirmDialog).toHaveBeenCalledWith(
        'Note changed while you were editing. Replace it with yours?',
        { confirmLabel: 'Replace', cancelLabel: 'Keep editing', danger: true },
      )
      expect(spy).not.toHaveBeenCalled()
    })

    it('Escape closes the box and keeps the draft', async () => {
      await typed('half')
      await fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Escape' })
      expect(screen.queryByRole('textbox')).toBeNull()
      expect(confirmDialog).not.toHaveBeenCalled()
      await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
      expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(
        'half',
      )
    })

    it('Cancel asks before discarding words, and keeps them when declined', async () => {
      await typed('half')
      vi.mocked(confirmDialog).mockResolvedValue(false)
      await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
      expect(confirmDialog).toHaveBeenCalledWith(
        'Discard this draft of your note?',
        { confirmLabel: 'Discard', cancelLabel: 'Keep editing', danger: true },
      )
      expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(
        'half',
      )
    })
  })
  describe('a save that waits', () => {
    const at = { note: null, workspace: 'studio', label: 'Run note' }
    /** The box open on `t` with `mine` typed, the other note `u` holding a
     * kept draft of its own */
    async function typedOnT() {
      sessionStorage.setItem('note:studio:u', 'u draft')
      const view = render(NoteEditor, { props: { ...at, target: 't' } })
      await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
      await fireEvent.input(screen.getByRole('textbox'), {
        target: { value: 'mine' },
      })
      return view
    }

    it('a note not written yet saves with the etag of an empty one', async () => {
      holds(null)
      const spy = vi
        .spyOn(api, 'setNote')
        .mockResolvedValue({ target: 't', note: 'mine', note_source: 'run' })
      await typedOnT()
      await fireEvent.click(screen.getByRole('button', { name: /^save/i }))
      await waitFor(() =>
        expect(spy).toHaveBeenCalledWith('t', 'mine', 'studio', 'e:'),
      )
    })

    it('switched to another note during the etag read, writes neither', async () => {
      const reads: (() => void)[] = []
      vi.spyOn(api.api, 'noteForSave').mockImplementation(
        (target) =>
          new Promise((resolve) =>
            reads.push(() => resolve({ target, note: null, etag: 'e:' })),
          ),
      )
      const spy = vi.spyOn(api, 'setNote')
      const view = await typedOnT()
      await fireEvent.click(screen.getByRole('button', { name: /^save/i }))
      // the user picks another file while the read is out; both show no note
      await view.rerender({ ...at, target: 'u' })
      for (const read of reads) read()
      await quiet()
      expect(spy).not.toHaveBeenCalled()
      expect(sessionStorage.getItem('note:studio:u')).toBe('u draft')
      expect(sessionStorage.getItem('note:studio:t')).toBe('mine')
    })

    it('switched to another note during the write, leaves that one alone', async () => {
      holds(null)
      let land!: () => void
      const spy = vi
        .spyOn(api, 'setNote')
        .mockImplementation(
          (target, note) =>
            new Promise(
              (resolve) =>
                (land = () => resolve({ target, note, note_source: 'run' })),
            ),
        )
      const onsaved = vi.fn()
      const view = await typedOnT()
      await view.rerender({ ...at, target: 't', onsaved })
      await quiet()
      await fireEvent.click(screen.getByRole('button', { name: /^save/i }))
      await waitFor(() =>
        expect(spy).toHaveBeenCalledWith('t', 'mine', 'studio', 'e:'),
      )
      await view.rerender({ ...at, target: 'u', onsaved })
      land()
      await quiet()
      expect(spy).toHaveBeenCalledTimes(1)
      expect(onsaved).not.toHaveBeenCalled()
      expect(sessionStorage.getItem('note:studio:u')).toBe('u draft')
      expect(screen.getByText('You have an unsaved draft.')).toBeTruthy()
      expect(screen.queryByText('mine')).toBeNull()
    })

    it('an etag read that fails writes nothing and says why', async () => {
      vi.spyOn(api.api, 'noteForSave').mockRejectedValue(
        new Error('the server did not answer'),
      )
      const spy = vi.spyOn(api, 'setNote')
      await typedOnT()
      await fireEvent.click(screen.getByRole('button', { name: /^save/i }))
      expect(await screen.findByRole('alert')).toBeTruthy()
      expect(screen.getByRole('alert').textContent).toContain(
        "Couldn't save the note",
      )
      expect(spy).not.toHaveBeenCalled()
      expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(
        'mine',
      )
      expect(sessionStorage.getItem('note:studio:t')).toBe('mine')
    })
  })
})
