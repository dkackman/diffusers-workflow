import type { EnhancerPreset, ManifestEntry, PromptDefinition } from './types'

/** A blank prompt, the editor's starting point. */
export function emptyPrompt(): PromptDefinition {
  return { text: '' }
}

/** The preset a stored prompt's intended_model preselects - matched
 * case-insensitively so 'MiniMax-H3' and 'minimax-h3' agree. undefined
 * when nothing matches: a wrong preset chosen confidently is worse than
 * leaving the current selection alone. */
export function presetForIntendedModel(
  presets: EnhancerPreset[],
  intendedModel: string | undefined,
): EnhancerPreset | undefined {
  const needle = (intendedModel ?? '').trim().toLowerCase()
  if (!needle) return undefined
  return presets.find((preset) =>
    preset.intended_models.some(
      (model) =>
        needle.includes(model.toLowerCase()) ||
        model.toLowerCase().includes(needle),
    ),
  )
}

/** The text file an enhancement job produced - the first .txt in its
 * manifest, or undefined while it has none. */
export function manifestTextFile(
  manifest: ManifestEntry[] | undefined,
): string | undefined {
  return manifest
    ?.flatMap((entry) => entry.files)
    .find((file) => file.toLowerCase().endsWith('.txt'))
}

/** Comma-separated tag input to the tags array: trimmed, deduplicated,
 * empties dropped. */
export function parseTags(raw: string): string[] {
  return [
    ...new Set(
      raw
        .split(',')
        .map((tag) => tag.trim())
        .filter(Boolean),
    ),
  ]
}

/** The stored text a prompt: reference stands in for - the tooltip shown
 * over an input holding one. undefined for anything else, so it collapses
 * to no title attribute at all. */
export function promptTooltip(
  value: unknown,
  texts: Record<string, string>,
): string | undefined {
  if (typeof value !== 'string' || !value.startsWith('prompt:')) {
    return undefined
  }
  return texts[value.slice('prompt:'.length).trim()] || undefined
}

/** The one datalist of stored-prompt suggestions - declared once per page,
 * referenced everywhere a prompt: reference can be typed. */
export const PROMPT_LIST_ID = 'prompt-references'

/** The datalist id for stored-prompt suggestions, attached only once the
 * value has committed to a prompt: reference - so the dropdown doesn't pop
 * over ordinary text. */
export function promptListId(value: unknown): string | undefined {
  return typeof value === 'string' && value.startsWith('prompt:')
    ? PROMPT_LIST_ID
    : undefined
}

/** Every intended_model value worth suggesting: what the presets know,
 * plus what the library already uses - a value on a card should also be
 * a suggestion in the editor. */
export function knownIntendedModels(
  presets: EnhancerPreset[],
  inUse: Array<string | undefined> = [],
): string[] {
  return [
    ...new Set(
      presets
        .flatMap((preset) => preset.intended_models)
        .concat(inUse.filter((model): model is string => Boolean(model))),
    ),
  ].sort()
}
