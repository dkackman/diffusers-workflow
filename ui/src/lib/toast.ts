import { toast } from 'svelte-sonner'

/** How long an undo toast offers to take the action back */
export const UNDO_MS = 8000
/** How long news with a place to go and see stays */
export const NEWS_MS = 12000

/** All app feedback goes through here, not svelte-sonner directly - one
 * place owns durations and dismissal semantics. Successes announce and
 * leave; errors stay until the user dismisses them. An id makes an error
 * replace its previous occurrence instead of stacking (a re-fired JSON
 * parse error on every blur, say). An undo is a success that can be taken
 * back from the toast for as long as it shows. */
export const notify = {
  success(message: string) {
    toast.success(message)
  },
  error(message: string, id?: string) {
    toast.error(message, { duration: Number.POSITIVE_INFINITY, id })
  },
  /** `id` lets the caller take the toast away (`dismiss`) when the thing
   * it offers to undo turned out not to have happened */
  undo(message: string, onUndo: () => void, id?: string) {
    toast.success(message, {
      duration: UNDO_MS,
      id,
      action: { label: 'Undo', onClick: onUndo },
    })
  },
  /** Something that happened elsewhere, with the button that goes to it */
  news(message: string, action: { label: string; onClick: () => void }) {
    toast.info(message, { duration: NEWS_MS, action })
  },
  dismiss(id: string) {
    toast.dismiss(id)
  },
}
