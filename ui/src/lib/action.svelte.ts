// One wrapper for the buttons that change things. It owns the three things
// each of them used to forget: a busy guard (a double click does one thing),
// a failure the user can read (notify.failure, never a raw message), and
// the optional question before and the optional undo after.
import { confirmDialog } from './confirm.svelte'
import { notify } from './toast'

export type ConfirmSpec = {
  message: string
  title?: string
  confirmLabel: string
  cancelLabel?: string
  danger?: boolean
}

export interface Action<A extends unknown[], R> {
  /** Undefined without calling fn while busy, when the confirm is
   * declined, or on failure. Never rejects. */
  run(...args: A): Promise<R | undefined>
  readonly busy: boolean
}

type Options<A extends unknown[], R> = {
  doing: string | ((...args: A) => string)
  confirm?: (...args: A) => ConfirmSpec | null
  success?: string | ((r: R) => string)
  undo?: (r: R) => { message: string; revert: () => Promise<unknown> }
}

export function action<A extends unknown[], R>(
  fn: (...args: A) => Promise<R>,
  opts: Options<A, R>,
): Action<A, R> {
  let busy = $state(false)
  const doingOf = (args: A) =>
    typeof opts.doing === 'function' ? opts.doing(...args) : opts.doing

  async function asked(args: A): Promise<boolean> {
    const spec = opts.confirm?.(...args)
    if (!spec) return true
    return confirmDialog(spec.message, {
      confirmLabel: spec.confirmLabel,
      cancelLabel: spec.cancelLabel,
      danger: spec.danger,
      title: spec.title,
    })
  }

  function announce(result: R, doing: string) {
    if (opts.undo) {
      const { message, revert } = opts.undo(result)
      notify.undo(message, () => {
        revert().catch((e) => notify.failure(`undo ${doing}`, e))
      })
    } else if (opts.success !== undefined) {
      notify.success(
        typeof opts.success === 'function'
          ? opts.success(result)
          : opts.success,
      )
    }
  }

  return {
    get busy() {
      return busy
    },
    async run(...args: A) {
      if (busy) return undefined
      busy = true
      try {
        if (!(await asked(args))) return undefined
        const result = await fn(...args)
        announce(result, doingOf(args))
        return result
      } catch (e) {
        notify.failure(doingOf(args), e)
        return undefined
      } finally {
        busy = false
      }
    },
  }
}
