import type { Plan } from './types'

/** One line of a plan, read for a person: what it says and how loudly.
 * `warn` marks a figure that is not a quote (no cost block, a figure from
 * another accelerator) and weights the box has to fetch first - the two
 * things that make "42 minutes" the wrong number to expect. */
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

  const downloads = plan.downloads_required.map((entry) => {
    const name = entry.repo ?? entry.url ?? '?'
    return entry.gb === null ? name : `${name} (${entry.gb} GB)`
  })
  if (downloads.length) {
    lines.push({
      text: `needs download: ${downloads.join(', ')}`,
      tone: 'warn',
    })
  }

  return lines
}

function figure(plan: Plan): PlanLine {
  const { minutes, basis, device, measured_on, partial, unpriced } =
    plan.estimate
  const card = measured_on ? `measured on ${measured_on}` : 'measured'
  const missing = partial
    ? `, plus unpriced: ${(unpriced ?? []).join(', ')}`
    : ''
  if (basis === 'unknown' || minutes === null) {
    return { text: 'no measured cost', tone: 'warn' }
  }
  if (basis === 'other_device') {
    return {
      text: `no figure for ${device} - ${minutes} min was ${card}${missing}`,
      tone: 'warn',
    }
  }
  const forList = Object.entries(plan.list_entries)
    .map(([name, count]) => `${count} ${name}`)
    .join(', ')
  // `derived` is the stored total stretched over a list the caller
  // resized - an estimate, so it is said to be one rather than quoted
  if (basis === 'derived') {
    return {
      text: `~${minutes} min on ${device} - estimated for ${forList}, from a figure ${card}${missing}`,
      tone: 'warn',
    }
  }
  const scaled = basis === 'per_entry' ? `re-priced for ${forList}, ` : ''
  return {
    text: `~${minutes} min on ${device} - ${scaled}${card}${missing}`,
    tone: 'plain',
  }
}
