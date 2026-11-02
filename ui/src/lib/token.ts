import { storageGet, storageSet } from './storage'

/** The optional API bearer token, entered once in the UI and persisted in
 * localStorage. An empty string means "no token configured" - matching the
 * server's default, unauthenticated mode. Kept as simple module state (not
 * a Svelte store) since it only needs to be read at request time, not
 * reactively rendered anywhere but the settings field itself. */
let current = storageGet<string>('api-token', '')

export function getApiToken(): string {
  return current
}

export function setApiToken(token: string): void {
  current = token.trim()
  storageSet('api-token', current)
}

/** A login link - `/?token=...`, the URL the Docker entrypoint logs
 * (docker/entrypoint.py) - saves its token the way the prompt does, then
 * drops it from the address bar so it stays out of history and screenshots.
 * The router lives in the hash, so the rest of the URL is kept as it was. */
export function adoptTokenFromUrl(
  loc: Location = window.location,
  hist: History = window.history,
): void {
  const url = new URL(loc.href)
  const token = url.searchParams.get('token')
  if (token === null) return
  if (token.trim()) setApiToken(token)
  url.searchParams.delete('token')
  hist.replaceState(hist.state, '', url.pathname + url.search + url.hash)
}
