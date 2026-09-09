import type { JobEvent, ManifestEntry } from './types'

/** A run's output files, grouped by the step that produced them. The
 * step_end stream carries the association live; the manifest confirms it
 * at the end - merged here so the grouping never flattens (the old
 * behavior pooled every file into one bag). */
export function groupResultFiles(
  manifest: ManifestEntry[] | undefined,
  events: JobEvent[],
): Array<{ step: string; files: string[]; reused: boolean }> {
  const order: string[] = []
  const byStep = new Map<string, Set<string>>()
  // Only the manifest knows a step was served from the step cache; the
  // live step_end stream carries files but not that flag
  const reused = new Set<string>()
  const add = (step: string, files: string[]) => {
    if (!byStep.has(step)) {
      byStep.set(step, new Set())
      order.push(step)
    }
    const set = byStep.get(step)!
    for (const file of files) set.add(file)
  }
  for (const event of events) {
    if (event.event === 'step_end') {
      add(
        (event.step as string) || '(unnamed)',
        (event.files as string[]) ?? [],
      )
    }
  }
  for (const entry of manifest ?? []) {
    add(entry.step, entry.files)
    if (entry.reused) reused.add(entry.step)
  }
  return order
    .map((step) => ({
      step,
      files: [...byStep.get(step)!],
      reused: reused.has(step),
    }))
    .filter((group) => group.files.length > 0)
}
