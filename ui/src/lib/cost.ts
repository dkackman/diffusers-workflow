/** The curated cost figure said for a person: who made it and the settings
 * it was measured at. Both read served fields and print them - the card's
 * `cost_drivers` map (each driver's preset-applied default, from
 * dw/server/catalog.py) and the listing's `cost_basis`. Nothing here knows
 * which driver moves the figure: that is dw/cost_scaling.py's, and the
 * per-run answer is validate's plan (docs/ARCHITECTURE.md, the UI row). */
import { formatMinutes } from './format'
import type { WorkflowCost } from './types'

const MAX_VALUE = 40

function driverValue(value: unknown): string {
  if (value === null || value === undefined) return 'unset'
  if (Array.isArray(value)) return `${value.length} entries`
  if (typeof value === 'object') return '{…}'
  const text = String(value)
  return text.length > MAX_VALUE ? `${text.slice(0, MAX_VALUE)}…` : text
}

/** `cost_drivers` as "name=value, …" in the map's order; '' for none. */
export function costMeasuredAt(
  drivers: Record<string, unknown> | null | undefined,
): string {
  if (!drivers) return ''
  return Object.entries(drivers)
    .map(([name, value]) => `${name}=${driverValue(value)}`)
    .join(', ')
}

/** `cost_basis` in words; a basis this UI does not know prints raw. */
export function costBasisWords(basis: string | null | undefined): string {
  if (basis === 'curated') return "a maintainer's figure"
  return basis ?? ''
}

/** "~3 min · 22 GB VRAM" for one measured entry. */
export function costFigure(measured: WorkflowCost): string {
  return `${formatMinutes(measured.minutes)} · ${Number(
    measured.vram_gb.toFixed(1),
  )} GB VRAM`
}

/** "measured on <where>[ at <drivers>]" for the first measured entry, or
 * '' when the card has no cost. */
export function costMeasuredOn(
  cost: WorkflowCost[] | null | undefined,
  drivers: Record<string, unknown> | null | undefined,
): string {
  const measured = cost?.[0]
  if (!measured) return ''
  const where = measured.name ?? measured.device
  const at = costMeasuredAt(drivers)
  return at ? `measured on ${where} at ${at}` : `measured on ${where}`
}
