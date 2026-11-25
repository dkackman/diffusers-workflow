/** Text onto the clipboard, wherever the UI is opened. `navigator.clipboard`
 * exists only in a secure context, and this UI is routinely opened over
 * plain http from another machine (http://<gpu-box>:8765), so without it -
 * or when it refuses - the text goes through a hidden textarea and
 * `execCommand('copy')`. True when the text got there. */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    /* refused: try the textarea */
  }
  return fallbackCopy(text)
}

function fallbackCopy(value: string): boolean {
  const textarea = document.createElement('textarea')
  textarea.value = value
  textarea.style.position = 'fixed'
  textarea.style.opacity = '0'
  document.body.appendChild(textarea)
  textarea.focus()
  textarea.select()
  let ok: boolean
  try {
    ok = document.execCommand('copy')
  } catch {
    ok = false
  }
  document.body.removeChild(textarea)
  return ok
}
