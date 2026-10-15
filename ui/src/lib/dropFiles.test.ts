import { describe, expect, it, vi } from 'vitest'
import { dropFiles } from './dropFiles'

function dragEvent(type: string, files: File[]) {
  const event = new Event(type, { bubbles: true, cancelable: true })
  Object.defineProperty(event, 'dataTransfer', {
    value: { types: ['Files'], files, dropEffect: 'none' },
  })
  return event
}

describe('dropFiles', () => {
  it('hands over the accepted files and marks the target while dragging', () => {
    const node = document.createElement('div')
    const ondrop = vi.fn()
    const png = new File(['x'], 'a.png', { type: 'image/png' })
    const txt = new File(['x'], 'b.txt', { type: 'text/plain' })
    const action = dropFiles(node, {
      ondrop,
      accept: (f) => f.type.startsWith('image/'),
    })
    node.dispatchEvent(dragEvent('dragenter', [png, txt]))
    expect(node.classList.contains('dropping')).toBe(true)
    const drop = dragEvent('drop', [png, txt])
    node.dispatchEvent(drop)
    expect(drop.defaultPrevented).toBe(true)
    expect(ondrop).toHaveBeenCalledWith([png])
    expect(node.classList.contains('dropping')).toBe(false)
    action?.destroy?.()
  })

  it('claims but ignores drops while disabled', () => {
    const node = document.createElement('div')
    const ondrop = vi.fn()
    dropFiles(node, { ondrop, disabled: true })
    const drop = dragEvent('drop', [new File(['x'], 'a.png')])
    node.dispatchEvent(drop)
    expect(drop.defaultPrevented).toBe(true)
    expect(ondrop).not.toHaveBeenCalled()
  })

  it('keeps the mark until the last leave', () => {
    const node = document.createElement('div')
    const child = document.createElement('span')
    node.append(child)
    dropFiles(node, { ondrop: vi.fn() })
    const f = [new File(['x'], 'a.png')]
    node.dispatchEvent(dragEvent('dragenter', f))
    child.dispatchEvent(dragEvent('dragenter', f))
    node.dispatchEvent(dragEvent('dragleave', f))
    expect(node.classList.contains('dropping')).toBe(true)
    child.dispatchEvent(dragEvent('dragleave', f))
    expect(node.classList.contains('dropping')).toBe(false)
  })

  it('clears the mark when disabled flips on mid-drag', () => {
    const node = document.createElement('div')
    const ondrop = vi.fn()
    const action = dropFiles(node, { ondrop })
    const f = [new File(['x'], 'a.png')]
    node.dispatchEvent(dragEvent('dragenter', f))
    action?.update?.({ ondrop, disabled: true })
    node.dispatchEvent(dragEvent('drop', f))
    expect(node.classList.contains('dropping')).toBe(false)
    expect(ondrop).not.toHaveBeenCalled()
  })
})
