/** Which named sections of a list are shut, remembered in storage under
 * `key`. FolderGroups' folders and the Assets page's libraries both read
 * it; the name is whatever the list calls a section. */
import { storageGet, storageSet } from './storage'

export function collapsedState(key: string): {
  has(name: string): boolean
  toggle(name: string): void
} {
  const shut = $state<Record<string, boolean>>(
    storageGet(key, {} as Record<string, boolean>),
  )
  return {
    has: (name) => !!shut[name],
    toggle(name) {
      shut[name] = !shut[name]
      storageSet(key, $state.snapshot(shut))
    },
  }
}
