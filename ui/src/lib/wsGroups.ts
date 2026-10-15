/** The sidebar's workspace list, folded: names that share a prefix (the
 * part before the first '-') become one group row once there are `min` of
 * them, at the place the first one stood; names this browser hid are left
 * out. The open workspace is never hidden - hiding the page you are on
 * would strand you. */
export type WsEntry =
  | { kind: 'ws'; name: string }
  | { kind: 'group'; prefix: string; names: string[] }

const prefixOf = (name: string) => {
  const at = name.indexOf('-')
  return at > 0 ? name.slice(0, at) : null
}

export function groupWorkspaces(
  names: readonly string[],
  {
    hidden,
    open,
    min = 3,
  }: { hidden: readonly string[]; open: string; min?: number },
): { entries: WsEntry[]; hidden: string[] } {
  const shown = names.filter((n) => n === open || !hidden.includes(n))
  const counts = new Map<string, number>()
  for (const n of shown) {
    const p = prefixOf(n)
    if (p) counts.set(p, (counts.get(p) ?? 0) + 1)
  }
  const entries: WsEntry[] = []
  const groups = new Map<
    string,
    { kind: 'group'; prefix: string; names: string[] }
  >()
  for (const n of shown) {
    const p = prefixOf(n)
    if (p && (counts.get(p) ?? 0) >= min) {
      let group = groups.get(p)
      if (!group) {
        group = { kind: 'group', prefix: p, names: [] }
        groups.set(p, group)
        entries.push(group)
      }
      group.names.push(n)
    } else entries.push({ kind: 'ws', name: n })
  }
  return {
    entries,
    hidden: names.filter((n) => n !== open && hidden.includes(n)),
  }
}
