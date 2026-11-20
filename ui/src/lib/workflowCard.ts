// What a catalog card says on hover, from the fields the listing serves.
import { costBasisWords, costMeasuredOn } from './cost'
import { observedTitle } from './observed'
import type { WorkflowCard } from './types'

/** Everything the card does not have room to print, so hovering it still
 * answers "what is this and what does it need". `costBasis` is the
 * listing's `cost_basis`. */
export function cardTitle(
  name: string,
  detail: WorkflowCard | undefined,
  costBasis: string,
): string {
  if (!detail) return name
  const lines = [name]
  if (detail.description) lines.push(detail.description)
  if (detail.traits?.length) lines.push(detail.traits.join(', '))
  const measured = measuredLine(detail, costBasis)
  if (measured) lines.push(measured)
  const seen = observedTitle(detail.observed)
  if (seen) lines.push(seen)
  if (detail.writable === false)
    lines.push(`read-only, from the ${detail.origin} directory`)
  return lines.join('\n')
}

/** Who made the figure and the settings it holds for: the card's own
 * preset-applied driver values, never the raw definition's defaults. */
function measuredLine(detail: WorkflowCard, costBasis: string): string {
  const measuredOn = costMeasuredOn(detail.cost, detail.cost_drivers)
  const basis = costBasisWords(costBasis)
  return measuredOn && basis ? `${basis}, ${measuredOn}` : measuredOn
}
