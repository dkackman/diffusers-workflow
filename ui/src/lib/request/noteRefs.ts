// What a note for the agent is about: an item, one of its takes, or one
// file of a take. The server fills in `attempt`, `kind` and `gone` on a
// read; a reference not yet saved is numbered here from the plan.
import type { LedgerNoteRef, LedgerRequest } from '../types'

const same = (a: LedgerNoteRef, b: LedgerNoteRef) =>
  a.item === b.item &&
  (a.job ?? null) === (b.job ?? null) &&
  (a.output ?? null) === (b.output ?? null)

/** `refs` with `ref` added at the end, unless it names the same item, job
 * and file as one already there: then `refs` as it was */
export const addRef = (refs: LedgerNoteRef[], ref: LedgerNoteRef) =>
  refs.some((r) => same(r, ref)) ? refs : [...refs, ref]

/** The 1-based take a reference names: the server's `attempt`, or the
 * job's place among the item's attempts; null for an item alone or a job
 * no longer linked to it */
export function refAttempt(
  ref: LedgerNoteRef,
  request: LedgerRequest,
): number | null {
  if (!ref.job) return null
  if (ref.attempt != null) return ref.attempt
  const item = request.items.find((i) => i.key === ref.item)
  const at = item?.attempts.findIndex((a) => a.job === ref.job) ?? -1
  return at < 0 ? null : at + 1
}

const basename = (path: string) => path.split('/').pop() ?? path

/** The chip's words: "take 2 · shot3.mp4", "take 2", or the item's title */
export function refLabel(ref: LedgerNoteRef, request: LedgerRequest): string {
  if (ref.job) {
    const take = `take ${refAttempt(ref, request) ?? '?'}`
    return ref.output ? `${take} · ${basename(ref.output)}` : take
  }
  return request.items.find((i) => i.key === ref.item)?.title ?? ref.item
}

/** References as a new note sends them: item, job and file only */
export const writtenRefs = (refs: LedgerNoteRef[]) =>
  refs.map(({ item, job, output }) => ({
    item,
    ...(job ? { job } : {}),
    ...(output ? { output } : {}),
  }))
