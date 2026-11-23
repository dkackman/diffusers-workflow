import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/svelte'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../api'
import { confirmDialog } from '../confirm.svelte'
import NoteEditor from './NoteEditor.svelte'

vi.mock('../confirm.svelte', () => ({ confirmDialog: vi.fn() }))

describe('NoteEditor', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.mocked(confirmDialog).mockReset()
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
    expect(spy).toHaveBeenCalledWith('t', 'new', undefined)
    expect(onsaved).toHaveBeenCalledWith('new')
  })

  it('cancel restores the old note without calling the api', async () => {
    const spy = vi.spyOn(api, 'setNote')
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
    expect(spy).toHaveBeenCalledWith('t', null, undefined)
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
    expect(spy).toHaveBeenCalledWith('t', 'new', 'ws2')
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

    it('Use theirs puts the newer note in the box', async () => {
      await overtaken()
      await fireEvent.click(screen.getByRole('button', { name: 'Use theirs' }))
      expect(draft()).toBe('agent wrote this')
      expect(screen.queryByText(NOTICE)).toBeNull()
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
        expect(spy).toHaveBeenCalledWith('t', 'mine', undefined),
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
        expect(spy).toHaveBeenCalledWith('t', 'mine', undefined),
      )
    })

    it('a note that has not changed saves without asking', async () => {
      vi.spyOn(api, 'setNote').mockResolvedValue(stored)
      render(NoteEditor, { props: { note: 'old', target: 't' } })
      await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
      await fireEvent.click(screen.getByRole('button', { name: /^save/i }))
      expect(confirmDialog).not.toHaveBeenCalled()
    })
  })
})
