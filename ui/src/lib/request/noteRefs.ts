// What a note for the agent is about: an item, one of its takes, or one
// file of a take. The server fills in `attempt`, `kind` and `gone` on a
// read; a reference not yet saved is numbered here from the plan.
import type { LedgerNoteRef, LedgerRequest } from '../types'

const same = (a: LedgerNoteRef, b: LedgerNoteRef) =>
  a.item === b.item &&
  (a.job ?? null) === (b.job ?? null) &&
  (a.output ?? null) === (b.output ?? null)

/** Whether `wide` says no more than `narrow` does: the same item, and the
 * same take where it names one, and the same file where it names one. A
 * take's file covers the take, and a take covers its item */
const covers = (narrow: LedgerNoteRef, wide: LedgerNoteRef) =>
  narrow.item === wide.item &&
  (!wide.job || narrow.job === wide.job) &&
  (!wide.output || narrow.output === wide.output)

/** `refs` with `ref` added, unless one already there says as much or more
 * (then `refs` as it was). What the new one covers - its item alone, its
 * take without the file - goes, and it takes the first such place */
export function addRef(refs: LedgerNoteRef[], ref: LedgerNoteRef) {
  if (refs.some((r) => same(r, ref) || covers(r, ref))) return refs
  const at = refs.findIndex((r) => covers(ref, r))
  const kept = refs.filter((r) => !covers(ref, r))
  if (at < 0) return [...kept, ref]
  const place = refs.slice(0, at).filter((r) => !covers(ref, r)).length
  return [...kept.slice(0, place), ref, ...kept.slice(place)]
}

/** A saved note's references as chips: one for a take and its file, one
 * for an item and its take, in the order written */
export const collapseRefs = (refs: LedgerNoteRef[]) =>
  refs.filter(
    (r, i) =>
      !refs.some((other, j) => j !== i && !same(other, r) && covers(other, r)),
  )

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
