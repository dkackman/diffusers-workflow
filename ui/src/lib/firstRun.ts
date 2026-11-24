import { sharedHref, wsHref } from './routes'
import type { WorkflowList } from './types'

/** The workflow the first-run card sends a newcomer to: small, ungated. */
export const FIRST_RUN_WORKFLOW = 'templates/text-to-image'
/** The storage key (see storage.ts) that remembers the card was dismissed. */
export const FIRST_RUN_KEY = 'first-run-dismissed'

export type FirstRunLink = {
  href: string
  label: 'Open text-to-image' | 'Browse examples' | 'Open Workflows'
}

/** Where the card's link goes: the template's own page where the listing has
 * it (the workspace's, or the Examples page when it is read-only, as
 * WorkflowsPage decides); otherwise the Examples page when the server has
 * any; otherwise the workspace's Workflows page. A listing that failed
 * (null) still orients: it gives the Workflows page. */
export function firstRunLink(
  listing: WorkflowList | null,
  ws: string,
): FirstRunLink {
  const parts = FIRST_RUN_WORKFLOW.split('/')
  const details = listing?.details ?? {}
  if (listing?.workflows.includes(FIRST_RUN_WORKFLOW)) {
    const writable = details[FIRST_RUN_WORKFLOW]?.writable !== false
    return {
      href: writable
        ? wsHref(ws, 'workflows', ...parts)
        : sharedHref('examples', ...parts),
      label: 'Open text-to-image',
    }
  }
  if (Object.values(details).some((d) => d?.writable === false)) {
    return { href: sharedHref('examples'), label: 'Browse examples' }
  }
  return { href: wsHref(ws, 'workflows'), label: 'Open Workflows' }
}
