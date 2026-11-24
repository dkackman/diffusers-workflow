import type { ValidationResult } from './types'

/** One thing a validate answer said: an error (`bad`, a Run would be
 * refused) or a warning (`warn`), as text the page prints and never parses
 * further. */
export type Finding = { tone: 'bad' | 'warn'; text: string }

/** The findings a form field shows under itself, by variable name, and the
 * rest, for the general list. */
export type PlacedFindings = {
  byVariable: Record<string, Finding[]>
  /** Errors and the bare refusal, which name no field the form shows */
  general: Finding[]
  /** Warnings that name no shown variable: for whoever wrote the workflow */
  author: Finding[]
}

// `arguments.<name>` or `variables.<name>`, the name as
// validate_variable_name() allows it, then whatever follows it ([i], .key)
const VARIABLE_PATH = /^(?:arguments|variables)\.([A-Za-z_][A-Za-z0-9_-]*)(.*)$/
// A warning is "path: message" or a bare message; only a variable path
// prefix is read off it - a bare message may hold ": " itself
const VARIABLE_WARNING =
  /^((?:arguments|variables)\.[A-Za-z_][A-Za-z0-9_-]*\S*): ([\s\S]*)$/

/** A refusal with no finding list still says why, in `error` */
function bareRefusal(result: ValidationResult): Finding[] {
  const listed = (result.errors ?? []).length > 0
  return !result.valid && !listed && result.error
    ? [{ tone: 'bad', text: result.error }]
    : []
}

/** Split a validate answer into what goes under a field the form shows and
 * what goes in the general list. A finding is placed only by its
 * `arguments.<name>` / `variables.<name>` path prefix; the UI derives
 * nothing else from it and filters nothing out (docs/ARCHITECTURE.md). */
export function placeFindings(
  result: ValidationResult | null,
  shown: Iterable<string>,
): PlacedFindings {
  const names = new Set(shown)
  const placed: PlacedFindings = { byVariable: {}, general: [], author: [] }
  if (!result) return placed

  const place = (
    tone: Finding['tone'],
    path: string | null,
    message: string,
    whole: string,
  ) => {
    const match = path ? VARIABLE_PATH.exec(path) : null
    if (match && names.has(match[1])) {
      // what follows the name (an [i] into a list, a key) stays with the
      // message, so a field holding a list still says which entry
      const rest = match[2].replace(/^\./, '')
      const text = rest ? `${rest}: ${message}` : message
      ;(placed.byVariable[match[1]] ??= []).push({ tone, text })
    } else {
      ;(tone === 'bad' ? placed.general : placed.author).push({
        tone,
        text: whole,
      })
    }
  }

  const errors = result.errors ?? []
  for (const error of errors) {
    const path = error.path ?? null
    place('bad', path, error.message, `${path ?? 'root'}: ${error.message}`)
  }
  placed.general.push(...bareRefusal(result))
  for (const warning of result.warnings ?? []) {
    const match = VARIABLE_WARNING.exec(warning)
    if (match) place('warn', match[1], match[2], warning)
    else placed.author.push({ tone: 'warn', text: warning })
  }
  return placed
}
