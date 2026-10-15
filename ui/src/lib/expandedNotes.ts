/** Which long notes this browser left expanded, so a note someone opened
 * to read stays open when they come back to it. Newest last, capped, so the
 * list cannot grow without bound across thousands of run notes. */
import { storageGet, storageSet } from './storage'

const KEY = 'expanded-notes'
const CAP = 200

export function isRemembered(key: string): boolean {
  return storageGet<string[]>(KEY, []).includes(key)
}

export function remember(key: string, open: boolean): void {
  const list = storageGet<string[]>(KEY, []).filter((k) => k !== key)
  if (open) list.push(key)
  storageSet(KEY, list.slice(-CAP))
}
