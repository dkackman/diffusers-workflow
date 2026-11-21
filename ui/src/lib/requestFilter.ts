// The `req` hash filter shared by the Jobs and Gallery pages: `<id>` is a
// whole requests-ledger request, `<id>/<item>` one item of it. Ids and item
// keys are slugs, so the first `/` is the separator.
export type RequestFilter = { request: string; item: string }

export function parseRequestFilter(value: string | undefined): RequestFilter {
  const text = (value ?? '').trim()
  const slash = text.indexOf('/')
  if (slash < 0) return { request: text, item: '' }
  return { request: text.slice(0, slash), item: text.slice(slash + 1) }
}

export function formatRequestFilter(filter: RequestFilter): string {
  if (!filter.request) return ''
  return filter.item ? `${filter.request}/${filter.item}` : filter.request
}

/** The hash query that narrows Jobs or the Gallery to a request, or to one
 * item of it: what a request page's "its jobs" and "its outputs" append. */
export const requestQuery = (request: string, item = ''): string =>
  `?req=${encodeURIComponent(formatRequestFilter({ request, item }))}`
