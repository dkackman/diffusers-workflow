// What a finished job's "run it again" offers read from the server: the
// preset selectors its workflow file declares as it is now. The job's own
// definition has its bundles folded in and its `presets` stripped, so only
// a job run by catalog name can be looked up.
import { api } from '../api'

export type Presets = Record<string, Record<string, unknown>>

/** The `presets` the workflow `name` declares now; null when it declares
 * none, or the job was not run by name, or the file is gone */
export async function workflowPresets(
  name: string | null | undefined,
): Promise<Presets | null> {
  if (!name) return null
  try {
    const { definition } = await api.getWorkflow(name)
    const found = (definition as Record<string, unknown>)?.presets
    return found && typeof found === 'object' ? (found as Presets) : null
  } catch {
    // an inline or deleted workflow: no promotions
    return null
  }
}
