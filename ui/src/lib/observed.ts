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

export function observedTitle(observed: Observed): string | null {
  if (!observed) return null
  const parts: string[] = []
  const warm = num(observed.warm_minutes)
  const cold = num(observed.cold_minutes)
  if (warm !== null)
    parts.push(
      `${formatMinutes(warm)} with the model loaded (${runs(num(observed.warm_runs) ?? 0)})`,
    )
  if (cold !== null)
    parts.push(
      `${formatMinutes(cold)} from cold (${runs(num(observed.cold_runs) ?? 0)})`,
    )
  if (!parts.length) return null
  const card = observed.name ?? observed.device
  const since =
    typeof observed.since === 'string'
      ? ` since ${observed.since.slice(0, 10)}`
      : ''
  return `measured on this server${card ? ` (${card})` : ''}: ${parts.join(', ')}; ${runs(num(observed.runs) ?? 0)}${since}`
}
