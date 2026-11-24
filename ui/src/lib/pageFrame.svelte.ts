import { isEditable } from './editable'
import { overlayOpen } from './ui/layers.svelte'

/** The page the app is showing, as the tab names it. `PageHeader` sets the
 * title; `finishAlert`'s `refreshTitle` is the one writer of
 * `document.title`, so a running job's prefix and the page's name never
 * fight over the tab. */
export const BASE_TITLE = 'diffusers-workflow'

export const pageFrame = $state({ title: '' })

/** "Gallery · diffusers-workflow", or the app's name when no page named
 * itself. */
export function pageTitle(): string {
  return pageFrame.title ? `${pageFrame.title} · ${BASE_TITLE}` : BASE_TITLE
}

/** Focuses the page's own search field - the first input in a page header's
 * `.filter` - and says whether there was one. */
export function focusPageFilter(): boolean {
  const field = document.querySelector<HTMLInputElement>(
    'main .pagehead .filter input, main .pagehead input.filter',
  )
  field?.focus()
  return field !== null
}

/** `/` jumps to the page's search - unless the key is a field's to type,
 * an overlay is open, or the page has no search (then the key is left
 * alone). */
export function searchOnSlash(event: KeyboardEvent): void {
  if (isEditable(event.target) || overlayOpen()) return
  if (event.ctrlKey || event.metaKey || event.altKey) return
  if (focusPageFilter()) event.preventDefault()
}
