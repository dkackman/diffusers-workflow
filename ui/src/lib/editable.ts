/** Whether a keypress landed somewhere that owns its own keys: text entry,
 * a select, contenteditable, or a media element (a focused <video> seeks on
 * the arrow keys). A page-wide shortcut stands aside for all of them. */
export function isEditable(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    target instanceof HTMLMediaElement ||
    target.isContentEditable === true
  )
}
