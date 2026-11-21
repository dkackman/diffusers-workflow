// The other values of each preset selector a finished job declared: what
// "rerun with changes" offers as one-click reruns.
import { displayValue } from '../editor'

export type Promotion = {
  selector: string
  value: string
  /** The bundle's keys the original run set itself, as `name=value` */
  kept: string[]
}

type Presets = Record<string, Record<string, unknown>>

/** One promotion per bundle of each selector the run declared, the bundle
 * the run used left out. `values` is what the run used for each variable
 * and `args` the arguments it was queued with. */
export function promotionsOf(
  presets: Presets | null,
  declared: Record<string, unknown> | null,
  args: Record<string, unknown>,
  values: Record<string, unknown>,
): Promotion[] {
  if (!presets || !declared) return []
  const found: Promotion[] = []
  for (const [selector, bundles] of Object.entries(presets)) {
    if (!(selector in declared)) continue
    if (!bundles || typeof bundles !== 'object') continue
    const current = displayValue(values[selector])
    for (const [value, bundle] of Object.entries(bundles)) {
      if (value === current) continue
      found.push({ selector, value, kept: keptKeys(bundle, selector, args) })
    }
  }
  return found
}

/** An explicit argument beats the bundle (dw/presets.py), so a key the
 * original set itself stays as it set it. */
function keptKeys(
  bundle: unknown,
  selector: string,
  args: Record<string, unknown>,
): string[] {
  return Object.keys(bundle && typeof bundle === 'object' ? bundle : {})
    .filter((name) => name !== selector && name in args)
    .map((name) => `${name}=${displayValue(args[name])}`)
}

export function promotionTitle(p: Promotion): string {
  return (
    `queue this job again with ${p.selector} = ${p.value} - the bundle's values come from the workflow file as it is now, not as it was when this ran` +
    (p.kept.length ? `; keeps ${p.kept.join(', ')} set explicitly` : '')
  )
}
