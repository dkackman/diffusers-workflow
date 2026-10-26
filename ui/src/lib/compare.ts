/** What a side-by-side compare reads off each file's metadata: the
 * arguments that actually differ between the files. The rest are shared and
 * would only bury the one change the comparison is about. */

export const COMPARE_MAX = 4

type Meta = Record<string, unknown> | null

export function argumentOf(meta: Meta, key: string): unknown {
  if (!meta) return undefined
  if (key === 'model') return meta.model_name
  if (key === 'seed')
    return meta.seed ?? (meta.arguments as Record<string, unknown>)?.seed
  return (meta.arguments as Record<string, unknown> | undefined)?.[key]
}

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)

/** Keys that differ across the files that have metadata - model and seed
 * first, then the arguments alphabetically. A file whose metadata could
 * not be read is unknown, not different. */
export function differingArguments(metas: Meta[]): string[] {
  const known = metas.filter((m): m is Record<string, unknown> => !!m)
  if (known.length < 2) return []
  const keys = new Set<string>()
  for (const m of known)
    for (const k of Object.keys((m.arguments as object) ?? {})) keys.add(k)
  keys.delete('seed')
  const ordered = ['model', 'seed', ...[...keys].sort()]
  return ordered.filter((key) => {
    const first = argumentOf(known[0], key)
    return known.some((m) => !same(argumentOf(m, key), first))
  })
}

export function valueText(value: unknown): string {
  if (value === undefined || value === null) return '—'
  const text = typeof value === 'string' ? value : JSON.stringify(value)
  return text.length > 200 ? `${text.slice(0, 200)}…` : text
}
