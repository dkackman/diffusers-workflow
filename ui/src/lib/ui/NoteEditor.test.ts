import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../api'
import NoteEditor from './NoteEditor.svelte'

describe('NoteEditor', () => {
  beforeEach(() => vi.restoreAllMocks())
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
    expect(spy).toHaveBeenCalledWith('t', 'new')
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
    expect(spy).toHaveBeenCalledWith('t', null)
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
})
