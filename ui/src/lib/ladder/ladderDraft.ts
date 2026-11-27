// One preference key's ladder as the Server → Preferences editor holds it
// (#1762 stage C): each tier an ordered list of targets, first that fits
// the host wins. The server checks the ladder on PUT (dw/preferences.py:
// check_ladder) and resolves it; this only shapes the edit and what is sent.
import { ApiError } from '../apiErrors'
import { VARIABLE_NAME_CHARACTERS, VARIABLE_NAME_FIRST } from '../references'

/** A ladder entry: a catalog name, or one with the devices it may run on
 * and any arguments it adds (kept as read; this editor does not edit them) */
export type Target =
  | string
  | {
      workflow: string
      devices?: string[]
      arguments?: Record<string, unknown>
    }

export type Ladder = Record<string, Target[]>

/** The tiers in ladder order, cheapest first (dw/preferences.py: TIERS) */
export const TIERS = ['draft', 'standard', 'best'] as const

/** The devices a target may be limited to */
export const DEVICES = ['cuda', 'mps', 'cpu'] as const

/** The engine's key limit (dw/security.py: MAX_VARIABLE_NAME_LENGTH) */
const KEY_MAX = 100

export const targetName = (t: Target): string =>
  typeof t === 'string' ? t : t.workflow

export const targetDevices = (t: Target): string[] =>
  typeof t === 'string' ? [] : (t.devices ?? [])

export const targetArguments = (t: Target): string[] =>
  typeof t === 'string' ? [] : Object.keys(t.arguments ?? {})

/** The draft of a stored ladder: every tier present, in order, empty where
 * the ladder has none; a tier the server added later is kept after them */
export function draftOf(ladder: Record<string, unknown> | undefined): Ladder {
  const draft: Ladder = {}
  for (const tier of TIERS) draft[tier] = []
  for (const [tier, entries] of Object.entries(ladder ?? {}))
    draft[tier] = Array.isArray(entries) ? [...(entries as Target[])] : []
  return draft
}

/** `list` with entry `i` moved `by` places, clamped to the list */
export function moved(list: Target[], i: number, by: number): Target[] {
  const to = Math.max(0, Math.min(list.length - 1, i + by))
  if (to === i) return list
  const next = [...list]
  const [entry] = next.splice(i, 1)
  next.splice(to, 0, entry)
  return next
}

export const removed = (list: Target[], i: number): Target[] =>
  list.filter((_, at) => at !== i)

/** `list` with `name` last: a bare name, or limited to `devices` */
export function added(
  list: Target[],
  name: string,
  devices: readonly string[] = [],
): Target[] {
  const workflow = name.trim()
  if (!workflow) return list
  return [
    ...list,
    devices.length ? { workflow, devices: [...devices] } : workflow,
  ]
}

/** What a PUT sends: the tiers that hold a target (the server refuses an
 * empty rung) */
export function sendable(draft: Ladder): Ladder {
  return Object.fromEntries(
    Object.entries(draft).filter(([, entries]) => entries.length),
  )
}

export const sameLadder = (a: Ladder, b: Ladder): boolean =>
  JSON.stringify(sendable(a)) === JSON.stringify(sendable(b))

/** Why a new key is refused before it is sent, or null. The engine's rule
 * is a variable name's (dw/preferences.py: check_key) */
export function keyProblem(
  name: string,
  taken: readonly string[],
): string | null {
  const allowed = [...name].every((c) => VARIABLE_NAME_CHARACTERS.includes(c))
  if (!name || !VARIABLE_NAME_FIRST.includes(name[0]) || !allowed)
    return 'A key starts with a letter or _ and holds letters, numbers, _ and -.'
  if (name.length > KEY_MAX) return `A key is at most ${KEY_MAX} characters.`
  return taken.includes(name) ? `There is already a key called ${name}.` : null
}

/** The etag a 412 `stale_write` carries (what the server holds now, null
 * when it holds nothing), or undefined for any other failure */
export function staleEtag(error: unknown): string | null | undefined {
  if (!(error instanceof ApiError) || error.status !== 412) return undefined
  if (error.code !== 'stale_write') return undefined
  const etag = error.detail?.etag
  return typeof etag === 'string' ? etag : null
}
