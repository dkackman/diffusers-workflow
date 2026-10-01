import type { LibraryRoot } from './types'

/** The directory a save lands in: the search path's writable workspace
 * root. Every library listing carries the path as `libraries`, in order;
 * there is no separate "writable dir" field. Empty when the library has no
 * such root (a workspace with no assets, say). */
export function writableRoot(libraries: LibraryRoot[] | undefined): string {
  return (
    libraries?.find((l) => l.writable && l.origin === 'workspace')?.root ?? ''
  )
}
