import { flowNodeName } from './runstate'
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
  // The engine reports a step served from the step cache on the live
  // step_end stream, and again on the manifest
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
      if (end.reused) reused.add(end.step || '(unnamed)')
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

/** One step of a run that finished without writing a file. */
export interface UnsavedStep {
  /** The step's name in the definition. One entry covers a whole for_each
   * group, since that is the one step the workflow declares. */
  node: string
  /** The member names the engine ran it under, one per for_each entry -
   * empty for a step that has no `for_each`, which is its own name. */
  members: string[]
  /** Why nothing was written, read off the definition - null when the
   * definition does not explain it. */
  reason: UnsavedReason | null
}

/** The JSON key a step writes nothing because of, and what that means -
 * kept apart so the page can set the key in the type the engine resolves
 * literally. */
export interface UnsavedReason {
  key: string
  detail: string
}

/** Why a step's result block saves no file: the two ways the engine skips
 * the save, both in `Result.save` (dw/result.py). */
export function unsavedReason(step: Record<string, any>): UnsavedReason | null {
  const result = step.result
  if (result && result.save === false) {
    return {
      key: 'result.save',
      detail: 'is false, so the step is kept in memory and never written',
    }
  }
  if (!result || typeof result.content_type !== 'string') {
    return {
      key: 'result.content_type',
      detail: 'is not declared, so there is no file type to write',
    }
  }
  return null
}

/** The steps that ran and wrote nothing, with the reason the definition
 * gives for each.
 *
 * The page groups a run's files by producing step and drops the empty
 * groups, which made a workflow whose steps deliberately write nothing
 * indistinguishable from a run whose outputs had gone missing: the steps
 * were there, their files were not, and nothing said why. A step counts as
 * having run when the manifest carries its entry - every step that finished
 * gets one - or a top-level step_end named it, so a job still in flight
 * never lists a step it has not reached yet, and a historical job, which
 * has no events at all, reads correctly off the manifest alone. */
export function unsavedSteps(
  manifest: ManifestEntry[] | undefined,
  events: JobEvent[],
  definition: Record<string, any> | null,
): UnsavedStep[] {
  const byNode = new Map<string, Record<string, any>>()
  for (const step of (definition?.steps ?? []) as Array<Record<string, any>>) {
    if (typeof step?.name === 'string') byNode.set(step.name, step)
  }
  if (!byNode.size) return []

  // An inner step of a sub-workflow is no node of the parent's graph, so
  // the entry it rolls up into the manifest is no evidence about anything
  // here - only a definition step, or a member of one, names a node
  const nodeOf = (step: string): string | undefined => {
    const node = flowNodeName(step)
    return node && byNode.has(node) ? node : undefined
  }

  const wrote = new Set<string>()
  for (const group of groupResultFiles(manifest, events)) {
    const node = nodeOf(group.step)
    if (node) wrote.add(node)
  }

  const ran = new Map<string, string[]>()
  const note = (step: string) => {
    const node = nodeOf(step)
    if (!node) return
    const members = ran.get(node) ?? []
    if (!members.includes(step)) members.push(step)
    ran.set(node, members)
  }
  for (const entry of manifest ?? []) note(entry.step)
  for (const event of events) {
    if (event.event === 'step_end' && !event.parent_step && event.step) {
      note(event.step as string)
    }
  }

  return [...ran]
    .filter(([node]) => !wrote.has(node))
    .map(([node, members]) => ({
      node,
      members: members.filter((member) => member !== node),
      reason: unsavedReason(byNode.get(node)!),
    }))
}
