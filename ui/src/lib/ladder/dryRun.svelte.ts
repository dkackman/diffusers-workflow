// The live dry run under the chain editor (#1762): 300 ms after the last
// change to a draft, ask the server what each tier would run on this host
// if the draft were saved (`POST /api/preferences/{key}/check`, full=false,
// so no Hub call). Nothing is written. Each check carries a sequence
// number; an answer older than the latest sent is dropped, so a slow check
// never paints over a later draft. A 422 is the draft's shape error, held
// as `refused` for the card to show and to hold Save on; any other failure
// leaves the last answer, and Save, as they were - the PUT is the gate.
import { api } from '../api'
import { ApiError } from '../apiErrors'
import type { PreferenceCheck } from '../types'
import { sendable, type Ladder } from './ladderDraft'

/** The quiet after the last change before checking */
export const DRY_RUN_AFTER_MS = 300

export type DryRun = {
  /** The latest check's answer; null before one lands or for an empty draft */
  readonly answer: PreferenceCheck | null
  /** The latest check sent has not answered */
  readonly checking: boolean
  /** The server's 422 for the draft's shape, as caught; null otherwise */
  readonly refused: unknown
}

export function dryRun(
  key: () => string,
  draft: () => Ladder,
  knownBlocked: () => string[],
): DryRun {
  let answer = $state<PreferenceCheck | null>(null)
  let checking = $state(false)
  let refused = $state<unknown>(null)
  // Not state: which answer is still wanted
  let sent = 0

  async function check(
    seq: number,
    name: string,
    ladder: Ladder,
    blocked: string[],
  ) {
    try {
      const got = await api.checkPreference(name, {
        ladder,
        full: false,
        known_blocked: blocked,
      })
      if (seq !== sent) return
      answer = got
      refused = null
    } catch (e) {
      if (seq !== sent) return
      if (e instanceof ApiError && e.status === 422) {
        answer = null
        refused = e
      }
    } finally {
      if (seq === sent) checking = false
    }
  }

  $effect(() => {
    const name = key()
    const ladder = sendable(draft())
    // read deeply: a move inside a tier is a change
    const body = JSON.parse(JSON.stringify(ladder)) as Ladder
    const blocked = [...knownBlocked()]
    if (!Object.keys(body).length) {
      // nothing to check, and no answer to an earlier draft still stands
      sent++
      answer = null
      refused = null
      checking = false
      return
    }
    const timer = setTimeout(() => {
      checking = true
      void check(++sent, name, body, blocked)
    }, DRY_RUN_AFTER_MS)
    return () => clearTimeout(timer)
  })

  return {
    get answer() {
      return answer
    },
    get checking() {
      return checking
    },
    get refused() {
      return refused
    },
  }
}
