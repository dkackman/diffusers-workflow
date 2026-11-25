// What the run form shows first, and what it remembers. The Essentials rule
// is derived from the workflow and its catalog card, not declared by it, so
// a different rule changes `essentials` alone. A draft is what was typed
// here last: per workspace and workflow, in this browser, never sent anywhere.
import { presetView, requiredVariables } from '../editor'
import { PROMPT, VARIABLE, referenceName } from '../references'
import { storageGet, storageRemove, storageSet } from '../storage'
import { WORKFLOW_PREFILL_KEY } from '../chainActions'
import { typedArguments } from './typedArguments'
import type { WorkflowCard, WorkflowDefinition } from '../types'

/** Free text: a string default that is empty, holds whitespace, or is a
 * prompt: reference. A dtype or a backend name is not. */
export function isFreeText(value: unknown): boolean {
  return (
    typeof value === 'string' &&
    (value.trim() === '' || /\s/.test(value) || value.startsWith(PROMPT))
  )
}

/** The arguments a Run sends: the overrides typed as their declared
 * defaults, blanks dropped, so a blank field leaves the workflow's own
 * default in place. Validate asks about exactly these, so its answer
 * predicts the Run click. */
export function runArguments(
  overrides: Record<string, string>,
  declared: Record<string, unknown>,
): Record<string, unknown> {
  return typedArguments(overrides, declared)
}

/** What Enhance starts from: what is typed, else the default - unless that
 * is a prompt: reference, which is not text to expand. */
export function enhanceIdea(
  overrides: Record<string, string>,
  workflow: WorkflowDefinition,
  key: string,
): string {
  const fallback = workflow.variables?.[key]
  const text =
    typeof fallback === 'string' && !fallback.startsWith(PROMPT) ? fallback : ''
  return overrides[key] || text
}

/** The variables a person is expected to set: required ones, the ones that
 * take a file, the ones that move the cost, preset selectors and free text */
export function essentials(
  workflow: WorkflowDefinition,
  card: WorkflowCard | undefined,
): Set<string> {
  const freeText = Object.entries(workflow.variables ?? {})
    .filter(([, value]) => isFreeText(value))
    .map(([key]) => key)
  return new Set([
    ...requiredVariables(workflow),
    ...Object.keys(card?.inputs ?? {}),
    ...Object.keys(card?.cost_drivers ?? {}),
    ...Object.keys(presetView(workflow)),
    ...freeText,
  ])
}

/** The rest, which the form folds. None until the card is known, so a field
 * that takes a file is never hidden while the catalog loads. */
export function advancedVariables(
  workflow: WorkflowDefinition,
  card: WorkflowCard | undefined,
): Set<string> {
  if (!card) return new Set()
  const shown = essentials(workflow, card)
  return new Set(
    Object.keys(workflow.variables ?? {}).filter((k) => !shown.has(k)),
  )
}

/** The variable the workflow's seed names (`seed: "variable:seed"`) */
export function seedVariable(workflow: WorkflowDefinition): string | null {
  return referenceName(workflow.seed, VARIABLE) || null
}

export function draftKey(workspace: string, name: string): string {
  return `run-draft:${workspace}:${name}`
}

/** What was typed here last, minus variables the workflow no longer has and
 * blanks */
export function readDraft(
  key: string,
  declared: string[],
): Record<string, string> {
  const stored = storageGet<Record<string, unknown>>(key, {})
  const kept: Record<string, string> = {}
  for (const name of declared) {
    const value = stored?.[name]
    if (typeof value === 'string' && value !== '') kept[name] = value
  }
  return kept
}

export function writeDraft(
  key: string,
  overrides: Record<string, string>,
): void {
  const kept = Object.fromEntries(
    Object.entries(overrides).filter(([, value]) => value !== ''),
  )
  if (Object.keys(kept).length) storageSet(key, kept)
  else storageRemove(key)
}

/** Writes a draft once typing pauses; `flush` writes what is waiting now */
export class DraftSaver {
  #timer: ReturnType<typeof setTimeout> | undefined
  #waiting: { key: string; values: Record<string, string> } | null = null

  schedule(key: string, values: Record<string, string>): void {
    this.#waiting = { key, values }
    clearTimeout(this.#timer)
    this.#timer = setTimeout(this.flush, 300)
  }

  flush = (): void => {
    clearTimeout(this.#timer)
    if (this.#waiting) writeDraft(this.#waiting.key, this.#waiting.values)
    this.#waiting = null
  }
}

/** Values another page asked this one to open with (Repair this segment,
 * #1476): read once, and only when they name this workflow. Filled in, never
 * submitted - Run is still the user's */
export function takePrefill(name: string): Record<string, string> {
  const raw = sessionStorage.getItem(WORKFLOW_PREFILL_KEY)
  if (raw === null) return {}
  sessionStorage.removeItem(WORKFLOW_PREFILL_KEY)
  try {
    const parsed = JSON.parse(raw)
    if (parsed?.workflow !== name) return {}
    const args = parsed.arguments
    if (!args || typeof args !== 'object') return {}
    return Object.fromEntries(
      Object.entries(args).map(([key, value]) => [key, String(value)]),
    )
  } catch {
    return {}
  }
}

/** What a run form opens with: what was typed here last, then a prefill.
 * `draft` and `prefill` are the two parts, kept to tell them apart. */
export function openingValues(
  workspace: string,
  name: string,
  workflow: WorkflowDefinition,
): {
  restored: boolean
  overrides: Record<string, string>
  draft: Record<string, string>
  prefill: Record<string, string>
} {
  const names = Object.keys(workflow.variables ?? {})
  const draft = readDraft(draftKey(workspace, name), names)
  const prefill = takePrefill(name)
  return {
    restored: Object.keys(draft).length > 0,
    overrides: { ...draft, ...prefill },
    draft,
    prefill,
  }
}

/** What of the form is the person's own to remember: a value still equal
 * to the prefill is the other page's word, not typed here, so it is not
 * stored - the draft it replaced, if any, stays. */
export function draftValues(
  overrides: Record<string, string>,
  opened: { draft: Record<string, string>; prefill: Record<string, string> },
): Record<string, string> {
  const kept: Record<string, string> = {}
  for (const [key, value] of Object.entries(overrides)) {
    if (opened.prefill[key] !== value) kept[key] = value
    else if (opened.draft[key] !== undefined) kept[key] = opened.draft[key]
  }
  return kept
}
