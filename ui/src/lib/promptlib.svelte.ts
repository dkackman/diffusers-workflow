import { api } from './api'

/** The prompt library as the editors share it: names feed the reference
 * datalists, texts feed the tooltip over any input holding a prompt:
 * reference. One store instead of prop-drilling a map through every
 * editor component. names stays undefined until a listing lands, so a
 * not-yet-loaded library is distinguishable from an empty one. */
export const promptLibrary = $state<{
  names: string[] | undefined
  texts: Record<string, string>
}>({ names: undefined, texts: {} })

/** Refresh the store from the server. Always refetches, so a prompt
 * saved moments ago shows up on the next page that loads. */
export async function loadPromptLibrary(): Promise<void> {
  try {
    const result = await api.listPrompts()
    promptLibrary.names = result.prompts
    promptLibrary.texts = Object.fromEntries(
      Object.entries(result.details).map(([name, detail]) => [
        name,
        detail.text,
      ]),
    )
  } catch {
    /* no prompt library - suggestions and tooltips just stay absent */
  }
}
