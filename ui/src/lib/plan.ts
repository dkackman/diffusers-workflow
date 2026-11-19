import type { Plan, PlanHostMemory, RequiredDownload } from './types'

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

  lines.push(figure(plan))

  if (plan.host_memory) lines.push(hostMemory(plan.host_memory))

  if (plan.cached_steps !== null && plan.steps > 0) {
    const cached = plan.cached_steps
    lines.push({
      text:
        cached >= plan.steps
          ? `all ${steps(plan.steps)} cached - a run generates nothing`
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
  return { text: `host RAM ~${gb} GB (${basis})`, tone: 'plain' }
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

function figure(plan: Plan): PlanLine {
  const line = rawFigure(plan)
  if (plan.estimate.basis === 'unknown' || plan.estimate.minutes === null) {
    return line
  }
  const text = `${line.text}${tempered(plan.estimate)}`
  return plan.estimate.low_confidence
    ? { text: `${text}, low confidence`, tone: 'warn' }
    : { text, tone: line.tone }
}

function rawFigure(plan: Plan): PlanLine {
  const { minutes, basis, device, priced_for } = plan.estimate
  if (basis === 'unknown' || minutes === null) return unpriced(plan.estimate)
  // the card the run dispatches to, when it says more than the backend
  const on = priced_for && priced_for !== device ? priced_for : device
  const { card, along, missing } = qualifiers(plan.estimate)
  if (basis === 'other_device') {
    return {
      text: `no figure for ${on} - ${minutes} min was ${card}${along}${missing}`,
      tone: 'warn',
    }
  }
  // another workflow's runs of the same pipeline priced this one (#797)
  if (basis === 'inherited') {
    return {
      text: `~${minutes} min, inherited from ${plan.estimate.inherited_from ?? 'a template'}${along}${missing}`,
      tone: 'plain',
    }
  }
  const forList = Object.entries(plan.list_entries)
    .map(([name, count]) => `${count} ${name}`)
    .join(', ')
  // `derived` is the stored total stretched over a list the caller
  // resized, or scaled along the drivers they moved - an estimate, so it is
  // said to be one rather than quoted
  if (basis === 'derived') {
    const target = forList ? ` for ${forList}` : ''
    return {
      text: `~${minutes} min on ${on} - estimated${target}${along}, from a figure ${card}${missing}`,
      tone: 'warn',
    }
  }
  const repriced = basis === 'per_entry' ? `re-priced for ${forList}, ` : ''
  return {
    text: `~${minutes} min on ${on} - ${repriced}${card}${along}${missing}`,
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
