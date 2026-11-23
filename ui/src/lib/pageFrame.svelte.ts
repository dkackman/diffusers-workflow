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
