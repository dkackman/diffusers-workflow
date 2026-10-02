/** What an output's embedded metadata says about how it was made, as the
 * job page and the gallery show it. */

/** A prompt argument as displayable text - pipelines accept a list too. */
export function promptText(value: unknown): string {
  if (typeof value === 'string') return value
  if (Array.isArray(value))
    return value.filter((v) => typeof v === 'string').join('\n')
  return ''
}

/** The model, seed and prompts behind an output. The step's realized
 * arguments carry the text the pipeline saw, not the 'variable:'
 * reference in the JSON; the seed falls back to the argument seed. */
export function describeOutputMeta(
  meta: Record<string, unknown> | null | undefined,
) {
  const args = (meta?.arguments as Record<string, unknown> | undefined) ?? null
  return {
    model: typeof meta?.model_name === 'string' ? meta.model_name : '',
    seed: typeof meta?.seed === 'number' ? meta.seed : args?.seed,
    prompt: promptText(args?.prompt),
    negativePrompt: promptText(args?.negative_prompt),
  }
}
