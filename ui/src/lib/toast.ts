import { toast } from 'svelte-sonner'

/** All app feedback goes through here, not svelte-sonner directly - one
 * place owns durations and dismissal semantics. Successes announce and
 * leave; errors stay until the user dismisses them. An id makes an error
 * replace its previous occurrence instead of stacking (a re-fired JSON
 * parse error on every blur, say). */
export const notify = {
  success(message: string) {
    toast.success(message)
  },
  error(message: string, id?: string) {
    toast.error(message, { duration: Number.POSITIVE_INFINITY, id })
  },
  dismiss(id: string) {
    toast.dismiss(id)
  },
}
