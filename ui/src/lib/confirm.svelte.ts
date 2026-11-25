// A window.confirm replacement that renders through the app's own modal
// styling instead of the browser chrome. One state object plus a single
// <ConfirmDialog> mounted in App.svelte, mirroring the notify()/Toaster
// pair in toast.ts - callers just `await confirmDialog(...)` in place of
// `window.confirm(...)`, or `await chooseDialog(...)` for more than two
// answers, or `await nameDialog(...)` for a name typed into a field (the
// window.prompt replacement). Calls made while a dialog is open wait their
// turn (FIFO), whichever kind they are.

export type Choice<T extends string = string> = {
  label: string
  value: T
  danger?: boolean
  /** The button's title: what this choice will do, when its label alone
   * cannot say (Keep both names the copy it makes) */
  title?: string
}

/** A name dialog's field: the dialog edits `value`, and its confirm is
 * disabled while `validate` answers a message, which is shown under it */
export type NameInput = {
  label: string
  value: string
  validate: (name: string) => string | null
}

type ConfirmState = {
  open: boolean
  message: string
  // The dialog's accessible name
  title: string
  choices: Choice[]
  input: NameInput | null
}

export const confirmState = $state<ConfirmState>({
  open: false,
  message: '',
  title: '',
  choices: [],
  input: null,
})

type Pending = {
  message: string
  title: string
  choices: Choice[]
  input?: NameInput
  resolve: (value: string | null) => void
}

const queue: Pending[] = []

/** The message up to its first sentence end: `. `, `! ` or `? `, whatever
 * case follows (a file's name may start the next sentence), or a newline.
 * A filename's dot, with no space after it, and `e.g.`, `i.e.` or `vs.` do
 * not end it. */
function firstSentence(message: string): string {
  const text = message.trim()
  const end = text.search(/(?<!\b(?:e\.g|i\.e|vs))[.!?](?=\s)|\n/u)
  if (end < 0) return text
  return text.slice(0, text[end] === '\n' ? end : end + 1).trim() || text
}

function show(head: Pending): void {
  confirmState.message = head.message
  confirmState.title = head.title
  confirmState.choices = head.choices
  confirmState.input = head.input ? { ...head.input } : null
  confirmState.open = true
}

export function chooseDialog<T extends string>(
  message: string,
  choices: Choice<T>[],
  options?: { title?: string },
): Promise<T | null> {
  return new Promise((resolve) => {
    enqueue({
      message,
      title: options?.title ?? firstSentence(message),
      choices,
      resolve: resolve as (value: string | null) => void,
    })
  })
}

function enqueue(pending: Pending): void {
  queue.push(pending)
  if (queue.length === 1) show(pending)
}

/** Ask for a name: the typed name on the confirm button (or Enter), null
 * on Cancel or Escape. The confirm stays disabled while `validate` answers
 * a message, and the message says why under the field. */
export function nameDialog(
  question: string,
  opts: {
    title: string
    label: string
    initial?: string
    confirmLabel?: string
    validate: (name: string) => string | null
  },
): Promise<string | null> {
  return new Promise((resolve) => {
    enqueue({
      message: question,
      title: opts.title,
      choices: [
        { label: 'Cancel', value: 'cancel' },
        { label: opts.confirmLabel ?? 'Save', value: 'confirm' },
      ],
      input: {
        label: opts.label,
        value: opts.initial ?? '',
        validate: opts.validate,
      },
      // Read before the next dialog replaces the field. A name that does
      // not validate is never answered, whatever pressed confirm
      resolve: (value) => {
        const name = confirmState.input?.value ?? ''
        const valid = opts.validate(name) === null
        resolve(value === 'confirm' && valid ? name : null)
      },
    })
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

/** Where an overlay portals: the full-screen element when one is showing
 * (the browser draws nothing outside it, so a dialog on <body> would be
 * open and unseen), else <body>. Read when the overlay opens. */
export function overlayHost(): HTMLElement {
  return (document.fullscreenElement as HTMLElement | null) ?? document.body
}

/** An action for a host element that must stay visible (the toaster):
 * it moves into the full-screen element while one shows, and back where it
 * was when full screen ends. */
export function followFullscreen(node: HTMLElement) {
  const home = node.parentElement
  const sync = () => {
    const target = document.fullscreenElement ?? home
    if (target && node.parentElement !== target) target.appendChild(node)
  }
  document.addEventListener('fullscreenchange', sync)
  sync()
  return {
    destroy() {
      document.removeEventListener('fullscreenchange', sync)
    },
  }
}
