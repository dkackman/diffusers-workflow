import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/svelte'
import ArgumentsEditor from './ArgumentsEditor.svelte'
import { api } from '../api'

// componentType='' keeps the effect that fetches a pipeline description
// from ever firing (see ArgumentsEditor.svelte's `if (!componentType) return`),
// so every key resolves through mediaKindFor's name-based fallback - the
// same fallback dw's task arguments (no annotation at all) rely on. That
// keeps these tests independent of the introspection API.

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('ArgumentsEditor media arguments', () => {
  it('renders a file-picker control for an image-typed argument, not a text input', () => {
    render(ArgumentsEditor, {
      args: { image: 'photo.png' },
      componentType: '',
    })

    expect(
      screen.getByRole('button', { name: /browse for a local image file/i }),
    ).toBeInTheDocument()
    // the location itself is still a plain, directly-editable text input
    expect(screen.getByPlaceholderText('image path or URL')).toHaveValue(
      'photo.png',
    )
  })

  it('leaves a non-image argument as an ordinary text input', () => {
    render(ArgumentsEditor, {
      args: { prompt: 'a scenic landscape' },
      componentType: '',
    })

    expect(
      screen.queryByRole('button', { name: /browse for a local/i }),
    ).not.toBeInTheDocument()
    const input = screen.getByDisplayValue('a scenic landscape')
    expect(input.tagName).toBe('INPUT')
    expect(input).not.toHaveAttribute('placeholder', 'image path or URL')
  })

  it('picking a file uploads it and writes the server path into the argument', async () => {
    vi.spyOn(api, 'uploadMedia').mockResolvedValue({
      path: '/abs/outputs/uploads/abc123.png',
      url: '/outputs/uploads/abc123.png',
    })

    const args: Record<string, unknown> = { image: '' }
    render(ArgumentsEditor, { args, componentType: '' })

    const browseButton = screen.getByRole('button', {
      name: /browse for a local image file/i,
    })
    const container = browseButton.closest('.media') as HTMLElement

    const hiddenFileInput = container.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement
    const file = new File(['pixels'], 'local.png', { type: 'image/png' })
    await fireEvent.change(hiddenFileInput, {
      target: { files: [file] },
    })

    // resolves asynchronously - wait for the mocked upload to settle
    await vi.waitFor(() => expect(api.uploadMedia).toHaveBeenCalledWith(file))
    await vi.waitFor(() =>
      expect(args.image).toBe('/abs/outputs/uploads/abc123.png'),
    )
  })
})
