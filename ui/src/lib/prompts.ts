import type { EnhancerPreset, ManifestEntry, PromptDefinition } from './types'

/** A blank prompt, the editor's starting point. */
export function emptyPrompt(): PromptDefinition {
  return { text: '' }
}

/** The preset a stored prompt's intended_model preselects - matched
 * case-insensitively so 'MiniMax-H3' and 'minimax-h3' agree - falling
 * back to the first preset when nothing matches. */
export function presetForIntendedModel(
  presets: EnhancerPreset[],
  intendedModel: string | undefined,
): EnhancerPreset | undefined {
  const needle = (intendedModel ?? '').trim().toLowerCase()
  if (needle) {
    const match = presets.find((preset) =>
      preset.intended_models.some(
        (model) =>
          needle.includes(model.toLowerCase()) ||
          model.toLowerCase().includes(needle),
      ),
    )
    if (match) return match
  }
  return presets[0]
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

/** Every intended_model value the presets know, for the editor's datalist. */
export function knownIntendedModels(presets: EnhancerPreset[]): string[] {
  return [
    ...new Set(presets.flatMap((preset) => preset.intended_models)),
  ].sort()
}
