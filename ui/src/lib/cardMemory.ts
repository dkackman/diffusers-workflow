import { gbFromMb } from './format'
import type { MemoryInfo } from './types'

/** One card's VRAM reading, flattened from /api/memory. */
export interface CardReading {
  /** the card as `cuda:1`; null for a single-card server's top-level reading */
  device: string | null
  name: string | null
  /** false when the card has never reported, or its backend has no GPU */
  available: boolean
  allocatedMb: number
  /** null on a backend that reports allocated only (MPS) */
  totalMb: number | null
  /** allocated as a share of total, 0-100; null without a total */
  pct: number | null
  live: boolean
  ageSeconds: number | null
}

/** Every card's reading. The server repeats the first card's at the top
 *  level and lists each card in `workers`; an older single-worker server
 *  sends the top level only, so that is the fallback. */
export function cardReadings(memory: MemoryInfo | null): CardReading[] {
  if (!memory) return []
  const entries = memory.workers?.length
    ? memory.workers
    : [{ ...memory, device: memory.device ?? null }]
  return entries.map((entry) => {
    const info = entry.info
    const totalMb = info?.gpu_memory_total_mb || null
    const allocatedMb = info?.gpu_memory_allocated_mb ?? 0
    return {
      device: entry.device ?? null,
      name: info?.gpu_device_name ?? null,
      available: !!info?.gpu_available,
      allocatedMb,
      totalMb,
      pct: totalMb ? Math.min(100, (100 * allocatedMb) / totalMb) : null,
      live: entry.live,
      ageSeconds: entry.age_seconds,
    }
  })
}

/** The header's figure: `18.2/24.0 GB` for one card, `18.2·3.1/24.0 GB`
 *  for cards of one size, each card's pair otherwise. */
export function meterText(cards: CardReading[]): string {
  const shown = cards.filter((c) => c.available && c.totalMb)
  if (!shown.length) return ''
  const totals = new Set(shown.map((c) => c.totalMb))
  if (totals.size === 1)
    return `${shown.map((c) => gbFromMb(c.allocatedMb)).join('·')}/${gbFromMb(shown[0].totalMb!)} GB`
  return (
    shown
      .map((c) => `${gbFromMb(c.allocatedMb)}/${gbFromMb(c.totalMb!)}`)
      .join(' · ') + ' GB'
  )
}

/** One line per card for a tooltip. */
export function meterTitle(cards: CardReading[]): string {
  return cards
    .filter((c) => c.available)
    .map(
      (c) =>
        `${c.name ?? c.device ?? 'GPU'} - ${gbFromMb(c.allocatedMb)}${c.totalMb ? ` of ${gbFromMb(c.totalMb)}` : ''} GB allocated`,
    )
    .join('\n')
}
