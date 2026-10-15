/** Workspaces this browser keeps out of the sidebar - a viewing choice, so
 * it is local, not a change to the workspace on the server. */
import { storageGet, storageSet } from './storage'

export const hiddenWorkspaces = $state({
  names: storageGet<string[]>('hidden-workspaces', []),
})

export function setHidden(name: string, hidden: boolean): void {
  const rest = hiddenWorkspaces.names.filter((n) => n !== name)
  hiddenWorkspaces.names = hidden ? [...rest, name] : rest
  storageSet('hidden-workspaces', hiddenWorkspaces.names)
}
