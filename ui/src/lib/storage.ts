/** One home for the app's persisted UI state. Keys are namespaced,
 * values are JSON, and storage that is missing, full, or forbidden
 * degrades to "state just doesn't persist". */
const PREFIX = 'dw-'

export function storageGet<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key)
    return raw === null ? fallback : (JSON.parse(raw) as T)
  } catch {
    return fallback
  }
}

export function storageSet(key: string, value: unknown): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value))
  } catch {
    /* private mode or quota - session only */
  }
}
