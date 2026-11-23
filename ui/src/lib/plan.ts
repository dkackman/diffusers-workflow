import type {
  Plan,
  PlanEstimate,
  PlanHostMemory,
  RequiredDownload,
} from './types'
import { formatMinutes } from './format'

/** One line of a plan, read for a person: what it says and how loudly.
 * `warn` marks a figure that is not a quote (no cost block, a figure from
 * another accelerator, a low-confidence or withheld one) and weights the box
 * has to fetch first - the things that make "42 minutes" the wrong number to
 * expect. */
export interface PlanLine {
  text: string
  tone: 'plain' | 'warn'
}

const steps = (n: number) => `${n} step${n === 1 ? '' : 's'}`

/** The plan a validate call answered with, as the lines the editor shows
 * under the verdict: the work, the figure with its basis, what the step
 * cache would serve, and what would be downloaded first. Pure, so the
 * wording is testable without a component. */
export function describePlan(plan: Plan): PlanLine[] {
  const lines: PlanLine[] = []

  const lists = Object.entries(plan.list_entries)
    .map(([name, count]) => `${name}: ${count}`)
    .join(', ')
  lines.push({
    text: lists ? `${steps(plan.steps)} (${lists})` : steps(plan.steps),
    tone: 'plain',
  })

  lines.push(figure(plan.estimate, plan.list_entries))

  if (plan.host_memory) lines.push(hostMemory(plan.host_memory))

  if (plan.cached_steps !== null && plan.steps > 0) {
    const cached = plan.cached_steps
    lines.push({
      text:
        cached >= plan.steps
          ? `all ${steps(plan.steps)} cached - a job generates nothing`
          : `${cached} of ${steps(plan.steps)} cached`,
      tone: 'plain',
    })
  }

  const elided = plan.elided_steps ?? []
  if (elided.length) {
    lines.push({
      text: `skipped: ${elided.map((e) => e.step).join(', ')} - nothing reads them`,
      tone: 'warn',
    })
  }

  const downloads = plan.downloads_required.map(download)
  if (downloads.length) {
    lines.push({
      text: `needs download: ${downloads.join(', ')}`,
      tone: 'warn',
    })
  }

  return lines
}

/** One repo or URL to fetch: its size when known, and why the fetch would
 * fail when the hub already said - a gated repo this box's token can't
 * reach, or one that isn't there. */
function download(entry: RequiredDownload): string {
  const name = entry.repo ?? entry.url ?? '?'
  const notes: string[] = []
  if (entry.gb !== null) notes.push(`${entry.gb} GB`)
  if (entry.not_found) notes.push('not found')
  else if (entry.access_blocked)
    notes.push("gated: this box's token can't reach it")
  return notes.length ? `${name} (${notes.join(', ')})` : name
}

/** The host RAM the run is projected to hold at its peak, and what the
 * figure rests on. */
function hostMemory(memory: PlanHostMemory): PlanLine {
  const gb = Math.round((memory.projected_mb / 1024) * 10) / 10
  const basis = {
    observed: 'observed',
    fit: `fit to ${memory.runs} run${memory.runs === 1 ? '' : 's'}`,
    declared: 'declared',
    inherited: 'inherited from the template',
  }[memory.basis]
  return { text: `host RAM about ${gb} GB (${basis})`, tone: 'plain' }
}

/** The qualifiers every priced figure carries: the card it was measured
 * on, what it was scaled along (#866: steps, canvas or frames moved off the
 * defaults it was measured at), and the steps it leaves unpriced. */
function qualifiers(estimate: Plan['estimate']) {
  const { measured_on, partial, unpriced, scaled } = estimate
  return {
    card: measured_on ? `measured on ${measured_on}` : 'measured',
    along: scaled ? ` (${scaled})` : '',
    missing: partial ? `, plus unpriced: ${(unpriced ?? []).join(', ')}` : '',
  }
}

function figure(
  estimate: PlanEstimate,
  listEntries: Plan['list_entries'],
): PlanLine {
  const line = rawFigure(estimate, listEntries)
  const { basis, minutes } = estimate
  // the lines that say no "on X" name the card the run dispatches to last,
  // so their own wording reads as it does without one
  const forCard =
    basis === 'unknown' || minutes === null || basis === 'inherited'
      ? dispatchCard(estimate)
      : ''
  if (basis === 'unknown' || minutes === null) {
    return { text: `${line.text}${forCard}`, tone: line.tone }
  }
  const text = `${line.text}${tempered(estimate)}`
  return estimate.low_confidence
    ? { text: `${text}, low confidence${forCard}`, tone: 'warn' }
    : { text: `${text}${forCard}`, tone: line.tone }
}

/** ` - for <card>` when the card the run dispatches to says more than the
 * backend, else nothing. */
function dispatchCard(estimate: Plan['estimate']): string {
  const { priced_for, device } = estimate
  return priced_for && priced_for !== device ? ` - for ${priced_for}` : ''
}

function rawFigure(
  estimate: PlanEstimate,
  listEntries: Plan['list_entries'],
): PlanLine {
  const { minutes, basis, device, priced_for } = estimate
  if (basis === 'unknown' || minutes === null) return unpriced(estimate)
  // the card the run dispatches to, when it says more than the backend
  const on = priced_for && priced_for !== device ? priced_for : device
  const { card, along, missing } = qualifiers(estimate)
  if (basis === 'other_device') {
    return {
      text: `no figure for ${on} - ${minutes} min was ${card}${along}${missing}`,
      tone: 'warn',
    }
  }
  // another workflow's runs of the same pipeline priced this one (#797)
  if (basis === 'inherited') {
    return {
      text: `${formatMinutes(minutes)}, inherited from ${estimate.inherited_from ?? 'a template'}${along}${missing}`,
      tone: 'plain',
    }
  }
  const forList = Object.entries(listEntries)
    .map(([name, count]) => `${count} ${name}`)
    .join(', ')
  // `derived` is the stored total stretched over a list the caller
  // resized, or scaled along the drivers they moved - an estimate, so it is
  // said to be one rather than quoted
  if (basis === 'derived') {
    const target = forList ? ` for ${forList}` : ''
    return {
      text: `${formatMinutes(minutes)} on ${on} - estimated${target}${along}, from a figure ${card}${missing}`,
      tone: 'warn',
    }
  }
  const repriced = basis === 'per_entry' ? `re-priced for ${forList}, ` : ''
  return {
    text: `${formatMinutes(minutes)} on ${on} - ${repriced}${card}${along}${missing}`,
    tone: 'plain',
  }
}

/** A thin observed history blended toward the curated figure rather than
 * quoted at full authority. */
function tempered(estimate: Plan['estimate']): string {
  const { curated_minutes, observed_minutes, runs } = estimate
  if (!estimate.tempered || curated_minutes === undefined) return ''
  const from = runs === null ? '' : ` from ${runs} run${runs === 1 ? '' : 's'}`
  const observed =
    observed_minutes === undefined ? '' : ` at ${observed_minutes} min`
  return `, blended toward the curated ${curated_minutes} min${from}${observed}`
}

/** No figure: history that could not be re-priced for this workflow, a
 * reason the server gave, or no cost at all. */
function unpriced(estimate: Plan['estimate']): PlanLine {
  if (estimate.withheld) {
    const source = estimate.inherited_from
      ? ` (not priced from ${estimate.inherited_from})`
      : ''
    return { text: `no figure - withheld${source}`, tone: 'warn' }
  }
  if (estimate.unpriced_reason) {
    return { text: `no figure - ${estimate.unpriced_reason}`, tone: 'warn' }
  }
  return { text: 'no measured cost', tone: 'warn' }
}

/** One estimate on its own, read for a person - the figure line a plan
 * shows, for a caller that holds only the `PlanEstimate` (a restart batch's
 * preview prices each member this way, #1473). Null - no plan was built -
 * says so rather than reading as free. */
export function describeEstimate(estimate: PlanEstimate | null): PlanLine {
  if (!estimate) return { text: 'no estimate', tone: 'warn' }
  return figure(estimate, {})
}

/** The sum of several estimates' minutes, with how many had no figure, so a
 * total over a batch never reads as complete when part of it is unpriced. */
export function describeEstimateTotal(
  estimates: (PlanEstimate | null)[],
): PlanLine {
  let minutes = 0
  let priced = 0
  for (const estimate of estimates) {
    if (estimate && estimate.basis !== 'unknown' && estimate.minutes !== null) {
      minutes += estimate.minutes
      priced += 1
    }
  }
  const missing = estimates.length - priced
  const unpricedNote = missing ? `, plus ${missing} with no figure` : ''
  if (!priced) {
    return {
      text: estimates.length ? 'total: no figure' : 'total: nothing to run',
      tone: estimates.length ? 'warn' : 'plain',
    }
  }
  const total = Math.round(minutes * 10) / 10
  return {
    text: `total ${formatMinutes(total)}${unpricedNote}`,
    tone: missing ? 'warn' : 'plain',
  }
}
