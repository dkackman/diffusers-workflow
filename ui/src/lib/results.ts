import type { JobEvent, ManifestEntry, StepEndEvent } from './types'

/** One step's output files, with where in the run directory they landed. */
export interface StepGroup {
  step: string
  files: string[]
  reused: boolean
  /** The step's `result.subfolder` - `''` for the run root. */
  subfolder: string
}

/** A run's output files, grouped by the step that produced them. The
 * step_end stream carries the association live; the manifest confirms it
 * at the end - merged here so the grouping never flattens (the old
 * behavior pooled every file into one bag). */
export function groupResultFiles(
  manifest: ManifestEntry[] | undefined,
  events: JobEvent[],
): StepGroup[] {
  const order: string[] = []
  const byStep = new Map<string, Set<string>>()
  // Only the manifest knows a step was served from the step cache; the
  // live step_end stream carries files but not that flag
  const reused = new Set<string>()
  // Placed by the live step_end, confirmed by the manifest - the manifest
  // is written last, so its value is the one that stands
  const subfolder = new Map<string, string>()
  const add = (step: string, files: string[], where: string | undefined) => {
    if (!byStep.has(step)) {
      byStep.set(step, new Set())
      order.push(step)
    }
    const set = byStep.get(step)!
    for (const file of files) set.add(file)
    if (where !== undefined) subfolder.set(step, where)
  }
  for (const event of events) {
    if (event.event === 'step_end') {
      const end = event as StepEndEvent
      add(end.step || '(unnamed)', end.files ?? [], end.subfolder)
    }
  }
  for (const entry of manifest ?? []) {
    add(entry.step, entry.files, entry.subfolder ?? '')
    if (entry.reused) reused.add(entry.step)
  }
  return order
    .map((step) => ({
      step,
      files: [...byStep.get(step)!],
      reused: reused.has(step),
      subfolder: subfolder.get(step) ?? '',
    }))
    .filter((group) => group.files.length > 0)
}

/** The step groups of one subfolder, under one heading on the job page. */
export interface SubfolderSection {
  subfolder: string
  groups: StepGroup[]
}

/** Step groups sectioned by the subfolder they landed in. A run where no
 * step chose one is a single root section, so the page renders as it did
 * before the field existed. Otherwise sections follow first appearance,
 * except `final` leads: by convention it is the deliverable, the thing the
 * page was opened for. The engine treats no name specially - any other
 * value is a section like the rest. */
export function sectionBySubfolder(groups: StepGroup[]): SubfolderSection[] {
  const order: string[] = []
  const bySubfolder = new Map<string, StepGroup[]>()
  for (const group of groups) {
    if (!bySubfolder.has(group.subfolder)) {
      bySubfolder.set(group.subfolder, [])
      order.push(group.subfolder)
    }
    bySubfolder.get(group.subfolder)!.push(group)
  }
  if (order.every((s) => s === '')) return [{ subfolder: '', groups }]
  const ordered = order.includes('final')
    ? ['final', ...order.filter((s) => s !== 'final')]
    : order
  return ordered.map((subfolder) => ({
    subfolder,
    groups: bySubfolder.get(subfolder)!,
  }))
}
