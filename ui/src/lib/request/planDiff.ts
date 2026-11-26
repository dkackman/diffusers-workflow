// What the agent changed between two reads of a request, said plainly: the
// lines Plan changes shows under the write that made them while the page
// is open, and since the last visit from a per-browser snapshot. The
// snapshot is a convenience that may be empty: storage that throws or holds
// something else, or a revision above the current one, gives no diff.
import { storageGet, storageRemove, storageSet } from '../storage'
import type { LedgerRequest } from '../types'

type ChangeKind = 'added' | 'dropped' | 'restored' | 'renamed' | 'changed'

export type PlanChange = {
  kind: ChangeKind
  key: string
  title: string
  /** The old title of a renamed item */
  was?: string
  what?: 'workflow' | 'arguments' | 'depends_on'
  /** The arguments that changed */
  names?: string[]
}

type SeenItem = {
  title: string
  workflow: string | null
  dropped: boolean
  depends_on: string[]
  args: Record<string, string>
}

/** A request as it was last looked at: enough to diff; a long value is
 * kept as a hash of it */
export type Snapshot = { revision: number; items: Record<string, SeenItem> }

/** An argument value longer than this is kept as a hash of it */
const FINGERPRINT_MAX = 120

/** FNV-1a, 32 bits, as hex */
function fnv1a(text: string): string {
  let hash = 0x811c9dc5
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(16).padStart(8, '0')
}

/** A value's JSON when short, else `#` and a hash of it */
export function fingerprint(value: unknown): string {
  const json = JSON.stringify(value) ?? 'null'
  return json.length <= FINGERPRINT_MAX ? json : `#${fnv1a(json)}`
}

export function snapshotOf(request: LedgerRequest): Snapshot {
  const items: Record<string, SeenItem> = {}
  for (const item of request.items) {
    const args: Record<string, string> = {}
    for (const [name, value] of Object.entries(item.arguments ?? {}))
      args[name] = fingerprint(value)
    items[item.key] = {
      title: item.title,
      workflow: item.workflow ?? null,
      dropped: item.dropped,
      depends_on: [...item.depends_on],
      args,
    }
  }
  return { revision: request.revision, items }
}

const isRequest = (value: Snapshot | LedgerRequest): value is LedgerRequest =>
  Array.isArray(value.items)

/** The arguments whose values differ, those now held first */
function changedArgs(was: SeenItem, now: SeenItem): string[] {
  const names = [
    ...new Set([...Object.keys(now.args), ...Object.keys(was.args)]),
  ]
  return names.filter((name) => was.args[name] !== now.args[name])
}

const sameList = (a: string[], b: string[]) =>
  a.length === b.length && a.every((value, i) => value === b[i])

/** What became of one item still held: renamed first, so what follows
 * reads under the new name */
function itemDiff(key: string, was: SeenItem, now: SeenItem): PlanChange[] {
  const title = now.title
  const out: PlanChange[] = []
  if (was.title !== title)
    out.push({ kind: 'renamed', key, title, was: was.title })
  if (was.dropped !== now.dropped)
    out.push({ kind: now.dropped ? 'dropped' : 'restored', key, title })
  // "" and null are both no workflow, as the engine reads them
  if ((was.workflow || null) !== (now.workflow || null))
    out.push({ kind: 'changed', key, title, what: 'workflow' })
  const names = changedArgs(was, now)
  if (names.length)
    out.push({ kind: 'changed', key, title, what: 'arguments', names })
  if (!sameList(was.depends_on, now.depends_on))
    out.push({ kind: 'changed', key, title, what: 'depends_on' })
  return out
}

/** Every change from `before` to `after`, in the plan's order; then the
 * items the ledger deleted (left out of a write while they had no jobs),
 * as dropped under the title they last had */
export function planDiff(
  before: Snapshot | LedgerRequest,
  after: LedgerRequest,
): PlanChange[] {
  const was = isRequest(before) ? snapshotOf(before).items : before.items
  const now = snapshotOf(after).items
  const held = Object.entries(now).flatMap(([key, item]) => {
    const old = was[key]
    if (old) return itemDiff(key, old, item)
    return item.dropped
      ? []
      : [{ kind: 'added' as const, key, title: item.title }]
  })
  const gone = Object.entries(was)
    .filter(([key, old]) => !(key in now) && !old.dropped)
    .map(([key, old]) => ({ kind: 'dropped' as const, key, title: old.title }))
  return [...held, ...gone]
}

const WHAT: Record<
  NonNullable<PlanChange['what']>,
  (c: PlanChange) => string
> = {
  workflow: () => 'its workflow',
  arguments: (c) => `its arguments (${(c.names ?? []).join(', ')})`,
  depends_on: () => 'what it waits on',
}

/** One change as Plan changes lists it */
export function lineOf(change: PlanChange): string {
  switch (change.kind) {
    case 'added':
      return `Added ${change.title}`
    case 'dropped':
      return `Dropped ${change.title}`
    case 'restored':
      return `Restored ${change.title}`
    case 'renamed':
      return `Renamed ${change.was} to ${change.title}`
    case 'changed':
      return `Changed ${change.title}: ${WHAT[change.what ?? 'workflow'](change)}`
  }
}

const seenKey = (workspace: string, id: string) =>
  `request-seen:${workspace}:${id}`

const isSnapshot = (value: unknown): value is Snapshot =>
  typeof value === 'object' &&
  value !== null &&
  typeof (value as Snapshot).revision === 'number' &&
  typeof (value as Snapshot).items === 'object' &&
  (value as Snapshot).items !== null

export function readSnapshot(workspace: string, id: string): Snapshot | null {
  try {
    const found = storageGet<unknown>(seenKey(workspace, id), null)
    return isSnapshot(found) ? found : null
  } catch {
    return null
  }
}

/** How many requests' snapshots a browser keeps: the newest written */
export const SEEN_KEPT = 50
const SEEN_ORDER = 'request-seen-order'

/** Note `key` as the newest snapshot and forget the ones past SEEN_KEPT */
function keepNewest(key: string) {
  const found = storageGet<unknown>(SEEN_ORDER, [])
  const order = (Array.isArray(found) ? found : []).filter(
    (k): k is string => typeof k === 'string' && k !== key,
  )
  order.push(key)
  for (const old of order.splice(0, Math.max(0, order.length - SEEN_KEPT)))
    storageRemove(old)
  storageSet(SEEN_ORDER, order)
}

export function writeSnapshot(workspace: string, id: string, snap: Snapshot) {
  try {
    const key = seenKey(workspace, id)
    storageSet(key, snap)
    keepNewest(key)
  } catch {
    /* the snapshot is a convenience */
  }
}

/** This browser has now seen `read`: the next visit diffs against it */
export const markSeen = (workspace: string, read: LedgerRequest) =>
  writeSnapshot(workspace, read.id, snapshotOf(read))

export type Since = { from: number; to: number; changes: PlanChange[] }

/** What changed since this browser last looked at the request, from its
 * snapshot, which is then replaced by this read unless it already is it.
 * A snapshot at or above the read's revision (a request deleted and made
 * again) says nothing */
export function seenSince(
  workspace: string,
  read: LedgerRequest,
): Since | null {
  const seen = readSnapshot(workspace, read.id)
  if (seen?.revision !== read.revision) markSeen(workspace, read)
  if (!seen || seen.revision >= read.revision) return null
  try {
    const changes = planDiff(seen, read)
    return changes.length
      ? { from: seen.revision, to: read.revision, changes }
      : null
  } catch {
    return null
  }
}

/** What `next` brought over `previous`, listed under the write that made
 * each change (`history[].changed`), by revision; null for a first read, only
 * the user's own writes, or no item changed. A change no entry names goes
 * under the agent's newest write. */
export function agentChanges(
  previous: LedgerRequest | null,
  next: LedgerRequest,
): Record<number, PlanChange[]> | null {
  if (!previous || previous.id !== next.id) return null
  const brought = next.history
    .filter((entry) => entry.revision > previous.revision)
    .sort((a, b) => b.revision - a.revision)
  const agent = brought.find((entry) => entry.by === 'agent')
  if (!agent) return null
  const diffs: Record<number, PlanChange[]> = {}
  for (const change of planDiff(previous, next)) {
    const what = change.what ?? (change.kind as Exclude<ChangeKind, 'changed'>)
    const writer =
      brought.find((entry) => entry.changed?.[change.key]?.includes(what)) ??
      agent
    ;(diffs[writer.revision] ??= []).push(change)
  }
  return Object.keys(diffs).length ? diffs : null
}

/** What the page shows of what changed: `diffs` under each write,
 * `since` above them all */
export type Seen = { diffs: Record<number, PlanChange[]>; since: Since | null }
export const UNSEEN: Seen = { diffs: {}, since: null }

/** `held` after a read replaced `previous` with `next`: each write's
 * changes kept under it, and on a first read what changed since the last
 * visit. Each diff is against the read the page held, never the snapshot.
 * Every read that moves the revision is what was last seen, so the next
 * visit lists only what this one did not show, the user's writes included */
export function afterRead(
  workspace: string,
  previous: LedgerRequest | null,
  next: LedgerRequest,
  held: Seen,
): Seen {
  const brought = agentChanges(previous, next)
  const diffs = brought ? { ...held.diffs, ...brought } : held.diffs
  if (!previous) return { diffs, since: seenSince(workspace, next) }
  if (next.revision !== previous.revision) markSeen(workspace, next)
  return { diffs, since: held.since }
}

/** Dismiss: the panel goes, and the request as held now is what was seen */
export function dismissed(
  workspace: string,
  read: LedgerRequest | null,
  held: Seen,
): Seen {
  if (read) markSeen(workspace, read)
  return { ...held, since: null }
}
