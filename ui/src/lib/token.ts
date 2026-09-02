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
