/** Drop files from the desktop onto an element. A drag that carries no
 * files (text, a link, a tile being dragged inside the page) is left alone
 * so it keeps its native behaviour. `dropping` marks the element while
 * files hover; enter/leave are counted because they fire for every child
 * the pointer crosses. */
import type { Action } from 'svelte/action'

export interface DropOptions {
  ondrop: (files: File[]) => void
  accept?: (file: File) => boolean
  disabled?: boolean
}

export const dropFiles: Action<HTMLElement, DropOptions> = (node, initial) => {
  let options = initial
  let depth = 0
  // A file drag is always claimed, even while disabled: left alone, the
  // browser would navigate to the dropped file and lose the page (and any
  // upload in flight). Only the highlight and the hand-over are gated
  const carriesFiles = (e: DragEvent) =>
    Array.from(e.dataTransfer?.types ?? []).includes('Files')
  const clear = () => {
    depth = 0
    node.classList.remove('dropping')
  }
  const enter = (e: DragEvent) => {
    if (!carriesFiles(e)) return
    e.preventDefault()
    depth++
    if (!options.disabled) node.classList.add('dropping')
  }
  const over = (e: DragEvent) => {
    if (!carriesFiles(e)) return
    e.preventDefault()
    if (e.dataTransfer)
      e.dataTransfer.dropEffect = options.disabled ? 'none' : 'copy'
  }
  const leave = () => {
    if (depth > 0 && --depth === 0) node.classList.remove('dropping')
  }
  const drop = (e: DragEvent) => {
    if (!carriesFiles(e)) return
    e.preventDefault()
    clear()
    if (options.disabled) return
    const files = Array.from(e.dataTransfer?.files ?? []).filter(
      (f) => options.accept?.(f) ?? true,
    )
    if (files.length) options.ondrop(files)
  }
  node.addEventListener('dragenter', enter)
  node.addEventListener('dragover', over)
  node.addEventListener('dragleave', leave)
  node.addEventListener('drop', drop)
  return {
    update(next) {
      options = next
    },
    destroy() {
      node.removeEventListener('dragenter', enter)
      node.removeEventListener('dragover', over)
      node.removeEventListener('dragleave', leave)
      node.removeEventListener('drop', drop)
    },
  }
}
