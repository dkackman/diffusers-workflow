// One wrapper for the buttons that change things. It owns the three things
// each of them used to forget: a busy guard (a double click does one thing),
// a failure the user can read (notify.failure, never a raw message), and
// the optional question before and the optional undo after.
import { confirmDialog } from './confirm.svelte'
import { undoToast, type ConfirmSpec, type UndoTarget } from './destructive'
import { notify } from './toast'

export type { ConfirmSpec }

export interface Action<A extends unknown[], R> {
  /** Undefined without calling fn while busy, when the confirm is
   * declined, or on failure; null when fn answers null. Never rejects. */
  run(...args: A): Promise<R | undefined>
  readonly busy: boolean
}

// A fn that asks its own question (a name, Replace or Keep both) answers
// null when it is declined: nothing was done, so nothing is announced
type Options<A extends unknown[], R> = {
  doing: string | ((...args: A) => string)
  confirm?: (...args: A) => ConfirmSpec | null
  success?: string | ((r: NonNullable<R>) => string)
  /** The toast that takes it back (`undoToast`): its revert waits its
   * turn behind the action, and the action behind it */
  undo?: (r: NonNullable<R>) => {
    message: string
    revert: () => Promise<unknown>
    target?: UndoTarget
  }
}

/** One busy flag for an action and the undos it offers: a run while busy
 * is dropped (a double click does one thing), an undo waits its turn */
function turns() {
  let busy = $state(false)
  let running: Promise<unknown> = Promise.resolve()
  function hold<T>(work: () => Promise<T>): Promise<T> {
    busy = true
    const done = work().finally(() => {
      busy = false
    })
    running = done.catch(() => undefined)
    return done
  }
  return {
    get busy() {
      return busy
    },
    hold,
    async after(work: () => Promise<unknown>) {
      while (busy) await running
      return hold(work)
    },
  }
}

export function action<A extends unknown[], R>(
  fn: (...args: A) => Promise<R>,
  opts: Options<A, R>,
): Action<A, R> {
  const turn = turns()
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

  function announce(result: NonNullable<R>) {
    if (opts.undo) {
      const { message, revert, target } = opts.undo(result)
      undoToast(message, () => turn.after(revert), target)
    } else if (opts.success !== undefined) {
      notify.success(
        typeof opts.success === 'function'
          ? opts.success(result)
          : opts.success,
      )
    }
  }

  async function attempt(args: A): Promise<R | undefined> {
    try {
      if (!(await asked(args))) return undefined
      const result = await fn(...args)
      if (result !== null) announce(result as NonNullable<R>)
      return result
    } catch (e) {
      notify.failure(doingOf(args), e)
      return undefined
    }
  }

  return {
    get busy() {
      return turn.busy
    },
    async run(...args: A) {
      if (turn.busy) return undefined
      return turn.hold(() => attempt(args))
    },
  }
}
