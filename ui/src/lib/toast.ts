import { toast } from 'svelte-sonner'
import { describeError } from './apiErrors'
import { copyText } from './clipboard'

/** What to do about an action that failed because the server was out of
 * reach - unlike a load, nothing retries it */
export const OFFLINE_ACTION_REMEDY = 'Try again once the server is back.'

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
  error(
    message: string,
    id?: string,
    action?: { label: string; onClick: () => void },
  ) {
    toast.error(message, { duration: Number.POSITIVE_INFINITY, id, action })
  },
  /** A failed action: what was being done, why not, and what to do. The
   * full message is one click away for pasting into a report: Copy details
   * keeps the toast up (svelte-sonner closes one whose action does not
   * prevent the default), and when the copy fails the detail moves into
   * the toast to be selected by hand. */
  failure(doing: string, e: unknown, id?: string) {
    const described = describeError(e)
    const { title, detail } = described
    // A load reloads when the server is back; a failed action does not
    const remedy =
      described.kind === 'offline' ? OFFLINE_ACTION_REMEDY : described.remedy
    let shown: string | number | undefined = id
    const show = (description: string | undefined) => {
      shown = toast.error(`Couldn't ${doing}: ${title}`, {
        description,
        duration: Number.POSITIVE_INFINITY,
        id: shown,
        action: detail
          ? {
              label: 'Copy details',
              onClick: (event: MouseEvent) => {
                event.preventDefault()
                void copyText(detail).then((ok) =>
                  ok
                    ? notify.success('Copied')
                    : show(remedy ? `${remedy}\n${detail}` : detail),
                )
              },
            }
          : undefined,
      })
    }
    show(remedy)
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
