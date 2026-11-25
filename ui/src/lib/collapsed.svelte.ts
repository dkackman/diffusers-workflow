/** Which named sections of a list are shut, remembered in storage under
 * `key`. FolderGroups' folders and the Assets page's libraries both read
 * it; the name is whatever the list calls a section. A list whose key
 * changes while it is up passes a getter rather than building one per key. */
import { storageGet, storageSet } from './storage'

export interface Collapsed {
  has(name: string): boolean
  toggle(name: string): void
}

export function collapsedState(key: string | (() => string)): Collapsed {
  const keyOf = typeof key === 'function' ? key : () => key
  // What was toggled here, by key; a key not toggled yet reads storage
  const held = $state<Record<string, Record<string, boolean>>>({})
  const shut = (at: string): Record<string, boolean> =>
    held[at] ?? storageGet(at, {} as Record<string, boolean>)
  return {
    has: (name) => !!shut(keyOf())[name],
    toggle(name) {
      const at = keyOf()
      const next = { ...shut(at), [name]: !shut(at)[name] }
      held[at] = next
      storageSet(at, next)
    },
  }
}
