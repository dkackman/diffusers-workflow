// A window.confirm replacement that renders through the app's own modal
// styling instead of the browser chrome. One state object plus a single
// <ConfirmDialog> mounted in App.svelte, mirroring the notify()/Toaster
// pair in toast.ts - callers just `await confirmDialog(...)` in place of
// `window.confirm(...)`, or `await chooseDialog(...)` for more than two
// answers. Calls made while a dialog is open wait their turn (FIFO).

export type Choice<T extends string = string> = {
  label: string
  value: T
  danger?: boolean
}

type ConfirmState = {
  open: boolean
  message: string
  // The dialog's accessible name
  title: string
  choices: Choice[]
}

export const confirmState = $state<ConfirmState>({
  open: false,
  message: '',
  title: '',
  choices: [],
})

type Pending = {
  message: string
  title: string
  choices: Choice[]
  resolve: (value: string | null) => void
}

const queue: Pending[] = []

function firstSentence(message: string): string {
  const match = message.trim().match(/^[^.!?\n]*[.!?]?/)
  return match?.[0].trim() || message.trim()
}

function show(head: Pending): void {
  confirmState.message = head.message
  confirmState.title = head.title
  confirmState.choices = head.choices
  confirmState.open = true
}

export function chooseDialog<T extends string>(
  message: string,
  choices: Choice<T>[],
  options?: { title?: string },
): Promise<T | null> {
  return new Promise((resolve) => {
    const pending: Pending = {
      message,
      title: options?.title ?? firstSentence(message),
      choices,
      resolve: resolve as (value: string | null) => void,
    }
    queue.push(pending)
    if (queue.length === 1) show(pending)
  })
}

export async function confirmDialog(
  message: string,
  options?: {
    confirmLabel?: string
    cancelLabel?: string
    // A notice with one button: nothing to choose between, so no Cancel
    alert?: boolean
    danger?: boolean
    title?: string
  },
): Promise<boolean> {
  const choices: Choice[] = []
  if (!options?.alert)
    choices.push({ label: options?.cancelLabel ?? 'Cancel', value: 'cancel' })
  choices.push({
    label: options?.confirmLabel ?? 'OK',
    value: 'confirm',
    danger: options?.danger,
  })
  const answer = await chooseDialog(message, choices, {
    title: options?.title,
  })
  return answer === 'confirm'
}

// Answers the open dialog (null: dismissed) and opens the next in line.
export function resolveChoice(value: string | null): void {
  const head = queue.shift()
  if (!head) return
  head.resolve(value)
  const next = queue[0]
  if (next) show(next)
  else confirmState.open = false
}
