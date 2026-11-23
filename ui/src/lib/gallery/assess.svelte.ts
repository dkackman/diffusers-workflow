import { SvelteMap } from 'svelte/reactivity'
import { api } from '../api'
import type { AssessReply } from '../types'

/** One output's Assess: running, its reply, or the server's refusal. A
 * failure keeps the reply before it, so the list shown stays the last one
 * the server measured. */
export type AssessState = {
  running: boolean
  reply: AssessReply | null
  /** The failure as caught, for ErrorLine to describe; null when none */
  error: unknown
}

/** Assess results by output, held for the page's life and never persisted -
 * the route stores nothing either. Keyed by workspace and name, since the
 * same name in another workspace is another file. */
export class AssessCache {
  #entries = new SvelteMap<string, AssessState>()

  static key = (workspace: string, name: string) => `${workspace}\u0000${name}`

  get(key: string): AssessState | undefined {
    return this.#entries.get(key)
  }

  /** Assess one output. A second click while one runs does nothing. */
  async run(key: string, name: string): Promise<void> {
    const before = this.#entries.get(key)
    if (before?.running) return
    const reply = before?.reply ?? null
    this.#entries.set(key, { running: true, reply, error: null })
    try {
      const next = await api.assess(name)
      this.#entries.set(key, { running: false, reply: next, error: null })
    } catch (e) {
      this.#entries.set(key, { running: false, reply, error: e })
    }
  }
}

/** Warn rows first, otherwise in the server's order - what the server sends
 * already is; the stable sort keeps a list from anywhere else honest. */
export const warnFirst = <T extends { severity: string }>(rows: T[]): T[] =>
  [...rows].sort(
    (a, b) => Number(a.severity === 'info') - Number(b.severity === 'info'),
  )

/** Seconds into the file as m:ss.s, or null when the finding has no place. */
export function formatAt(at: unknown): string | null {
  if (typeof at !== 'number' || !Number.isFinite(at) || at < 0) return null
  const tenths = Math.round(at * 10)
  const minutes = Math.floor(tenths / 600)
  const seconds = (tenths % 600) / 10
  return `${minutes}:${seconds.toFixed(1).padStart(4, '0')}`
}

/** The muted line under an assessed list: what was measured, and each probe
 * that could not be, with the server's reason - so "no findings" reads
 * apart from "nothing measured". */
export function measuredLine(reply: AssessReply): string {
  const applied = reply.rules_applied.length
    ? `rules applied: ${reply.rules_applied.join(', ')}`
    : 'no rules applied'
  const skipped = Object.entries(reply.not_applicable ?? {}).map(
    ([probe, why]) => `${probe} not applicable (${why})`,
  )
  return [applied, ...skipped].join(' · ')
}
