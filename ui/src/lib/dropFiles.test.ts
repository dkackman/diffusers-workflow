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

  it('ignores drops while disabled', () => {
    const node = document.createElement('div')
    const ondrop = vi.fn()
    dropFiles(node, { ondrop, disabled: true })
    node.dispatchEvent(dragEvent('drop', [new File(['x'], 'a.png')]))
    expect(ondrop).not.toHaveBeenCalled()
  })
})
