// The LoRA catalog's suggestions for one base model (GET /api/loras,
// catalog-only - never the Hub), best first.
import { api } from '../api'
import type { LoraRow } from '../types'

const STATUS_ORDER: Record<string, number> = { proven: 0, trial: 1 }

/** A catalog read is a local file scan; one still unsettled after this long
 * has hung (a stalled connection, a stub that never answers) and counts as
 * a failure, so the panel never sits on a bare heading (#1501) */
export const CATALOG_TIMEOUT_MS = 8000

/** The catalog's rows for `model` (narrowed to the `workflow` partition
 * when given), rejected ones left out, proven before trial */
export async function catalogFor(
  model: string,
  workflow: string | undefined,
): Promise<LoraRow[]> {
  let giveUp: ReturnType<typeof setTimeout> | undefined
  const hung = new Promise<never>((_, reject) => {
    giveUp = setTimeout(() => reject(new Error('hung')), CATALOG_TIMEOUT_MS)
  })
  const asked = api.listLoras(workflow ? { model, workflow } : { model })
  try {
    const body = await Promise.race([asked, hung])
    return body.loras
      .filter((row) => row.status !== 'rejected')
      .sort(
        (a, b) =>
          (STATUS_ORDER[a.status] ?? 2) - (STATUS_ORDER[b.status] ?? 2) ||
          a.name.localeCompare(b.name),
      )
  } finally {
    clearTimeout(giveUp)
  }
}
