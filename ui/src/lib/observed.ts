/** The server's own history for a workflow - `observed` on a catalog
 * detail (dw/server/observed_cost.py) - said for a person. The curated
 * `cost` is a maintainer's claim about some other card; this is what this
 * box actually took, which is the number that predicts the next run. */
import { formatMinutes } from './format'

type Observed = Record<string, unknown> | null | undefined
const num = (v: unknown) => (typeof v === 'number' ? v : null)
const runs = (n: number) => `${n} ${n === 1 ? 'run' : 'runs'}`

export function observedText(observed: Observed): string | null {
  const minutes = num(observed?.warm_minutes) ?? num(observed?.cold_minutes)
  return minutes === null ? null : `${formatMinutes(minutes)} here`
}

// One clause per measured state, or nothing when that state has no runs
function timing(minutes: number | null, label: string, n: unknown) {
  return minutes === null
    ? []
    : [`${formatMinutes(minutes)} ${label} (${runs(num(n) ?? 0)})`]
}

// Where and since when: the card's name in parentheses, the date it began
function provenance(observed: Record<string, unknown>) {
  const card = observed.name ?? observed.device
  const since =
    typeof observed.since === 'string'
      ? ` since ${observed.since.slice(0, 10)}`
      : ''
  return { card: card ? ` (${card})` : '', since }
}

export function observedTitle(observed: Observed): string | null {
  if (!observed) return null
  const parts = [
    ...timing(
      num(observed.warm_minutes),
      'with the model loaded',
      observed.warm_runs,
    ),
    ...timing(num(observed.cold_minutes), 'from cold', observed.cold_runs),
  ]
  if (!parts.length) return null
  const { card, since } = provenance(observed)
  return `measured on this server${card}: ${parts.join(', ')}; ${runs(num(observed.runs) ?? 0)}${since}`
}
