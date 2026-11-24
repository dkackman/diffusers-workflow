import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/svelte'
import ArgumentsEditor from './ArgumentsEditor.svelte'
import { api } from '../api'
import type { PipelineDescription } from '../types'

// componentType='' keeps the resource that fetches a pipeline description
// from ever asking (see ArgumentsEditor.svelte's `if (!wanted) return null`),
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

    expect(screen.getByRole('button', { name: 'Upload…' })).toHaveAttribute(
      'title',
      'Upload an image from this computer, or drop one here',
    )
    // the location itself is still a plain, directly-editable text input
    expect(
      screen.getByPlaceholderText('type, drop or choose an image'),
    ).toHaveValue('photo.png')
  })

  it('leaves a non-image argument as an ordinary text input', () => {
    render(ArgumentsEditor, {
      args: { prompt: 'a scenic landscape' },
      componentType: '',
    })

    expect(
      screen.queryByRole('button', { name: 'Upload…' }),
    ).not.toBeInTheDocument()
    const input = screen.getByDisplayValue('a scenic landscape')
    expect(input.tagName).toBe('INPUT')
    expect(input).not.toHaveAttribute(
      'placeholder',
      'type, drop or choose an image',
    )
  })

  it('picking a file uploads it and writes the asset reference into the argument', async () => {
    vi.spyOn(api, 'uploadMedia').mockResolvedValue({
      reference: 'asset:abc123.png',
      url: '/outputs/uploads/abc123.png',
      workspace: 'default',
      shared: false,
    })

    const args: Record<string, unknown> = { image: '' }
    render(ArgumentsEditor, { args, componentType: '' })

    const uploadButton = screen.getByRole('button', { name: 'Upload…' })
    const container = uploadButton.closest('.reference') as HTMLElement

    const hiddenFileInput = container.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement
    const file = new File(['pixels'], 'local.png', { type: 'image/png' })
    await fireEvent.change(hiddenFileInput, {
      target: { files: [file] },
    })

    // resolves asynchronously - wait for the mocked upload to settle
    await vi.waitFor(() =>
      expect(api.uploadMedia).toHaveBeenCalledWith(file, 'local.png', false, {
        createOnly: true,
      }),
    )
    await vi.waitFor(() => expect(args.image).toBe('asset:abc123.png'))
  })
})

describe('ArgumentsEditor class description', () => {
  it("lists the described class's unused arguments once the name settles", async () => {
    const describe = vi.spyOn(api, 'describePipeline').mockResolvedValue({
      parameters: [
        {
          name: 'guidance_scale',
          annotation: 'float',
          default: 3.5,
          required: false,
        },
      ],
      accepts_kwargs: false,
    } as unknown as PipelineDescription)
    render(ArgumentsEditor, {
      args: { prompt: 'a cat' },
      componentType: 'ProbeArgumentsPipeline',
    })
    expect(
      await screen.findByText(/show available arguments \(1\)/),
    ).toBeInTheDocument()
    expect(describe).toHaveBeenCalledWith('ProbeArgumentsPipeline')
  })
})
