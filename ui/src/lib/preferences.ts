// Model preferences on the run check (#1762 stage B): which `preference:`
// key a workflow composes, and each tier's pick on this host as
// `GET /api/preferences?key=<k>&resolve=true` answers it. The server owns
// resolution (dw/preferences.py); this only reads its answer.
import { PREFERENCE, VARIABLE, referenceName } from './references'
import type {
  PreferenceResolution,
  PreferencesRead,
  WorkflowDefinition,
} from './types'

/** The variable the resolver reads the tier from on the composing
 * workflow (dw/preferences.py: the step's `tier`, else this variable) */
export const TIER_VARIABLE = 'tier'

/** The tiers in their ladder order, cheapest first */
const TIERS = ['draft', 'standard', 'best']

/** A key a step path names: the path itself, or a variable's default the
 * path reads (`variable:<name>`) */
function keyOf(path: unknown, workflow: WorkflowDefinition): string | null {
  if (typeof path !== 'string') return null
  const name = referenceName(path, VARIABLE)
  const value = name ? workflow.variables?.[name] : path
  if (typeof value !== 'string' || !value.startsWith(PREFERENCE)) return null
  return value.slice(PREFERENCE.length) || null
}

/** Every distinct preference key the workflow's own steps compose. A key
 * only a composed child names is not read: its plan line still shows it. */
export function preferenceKeys(workflow: WorkflowDefinition): string[] {
  const keys = new Set<string>()
  for (const step of workflow.steps ?? []) {
    const key = keyOf(step.workflow?.path, workflow)
    if (key) keys.add(key)
  }
  return [...keys]
}

/** The key the run check shows rungs for: the one key the workflow
 * composes, when it declares the tier variable that picks among them;
 * none with no key, two keys, or no tier to choose */
export function rungKey(workflow: WorkflowDefinition): string | null {
  if (!(TIER_VARIABLE in (workflow.variables ?? {}))) return null
  const keys = preferenceKeys(workflow)
  return keys.length === 1 ? keys[0] : null
}

export type Skip = { workflow: string; reason: string }

/** One tier as the form shows it: what this host runs, what it passed over
 * and why, or the refusal a validate at that tier would give */
export type Rung = {
  tier: string
  target: string | null
  skipped: Skip[]
  error: string | null
}

const isSkip = (s: unknown): s is Skip =>
  !!s &&
  typeof (s as Skip).workflow === 'string' &&
  typeof (s as Skip).reason === 'string'

/** The key's rungs from a resolved read, in ladder order; empty when the
 * answer has no such key or no resolutions */
export function rungsOf(read: PreferencesRead, key: string): Rung[] {
  return rungsFrom(read.preferences[key]?.resolutions)
}

/** One key's resolutions as rungs, in ladder order */
export function rungsFrom(
  resolutions: Record<string, PreferenceResolution> = {},
): Rung[] {
  const order = (tier: string) => {
    const i = TIERS.indexOf(tier)
    return i === -1 ? TIERS.length : i
  }
  return Object.keys(resolutions)
    .sort((a, b) => order(a) - order(b) || a.localeCompare(b))
    .map((tier) => {
      const r = resolutions[tier]
      return {
        tier,
        target: r.target ?? null,
        skipped: (r.skipped ?? []).filter(isSkip),
        error: r.error ?? null,
      }
    })
}
