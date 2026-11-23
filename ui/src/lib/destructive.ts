// How the UI asks before it destroys something the person made (the P5
// policy): one wording for every delete, which names the thing or counts
// it, says how many of them are starred - the files a person marked to
// keep - and puts the button in danger. Pages pass a subject, never their
// own sentence; ESLint holds it (a Delete or Remove confirmLabel outside
// this module fails lint). A caller runs the spec through action()'s
// `confirm`, or confirmDialog(spec.message, spec). What can be put back
// is not asked about: it acts at once and offers Undo (`undoToast`).
import { formatBytes } from './format'
import { notify } from './toast'

/** A question asked before an action (action()'s `confirm`) */
export type ConfirmSpec = {
  message: string
  title?: string
  confirmLabel: string
  cancelLabel?: string
  danger?: boolean
}

/** What an undo was made for: `now` is read when the undo is offered and
 * again when it is pressed, and a different answer - another workflow,
 * request or workspace open by then - refuses it with `away` */
export type UndoTarget = { now: () => unknown; away: string }

/** A reversible action done at once: the toast that takes it back for
 * UNDO_MS. It runs at most once - a dismissed toast's button still takes a
 * second click while it fades - and only on the thing it was made for. A
 * revert that fails is reported, never thrown. `id` names the toast, so the
 * caller can dismiss it (`notify.dismiss`) */
export function undoToast(
  message: string,
  revert: () => void | Promise<unknown>,
  target?: UndoTarget,
  id?: string,
): void {
  const was = target?.now()
  let used = false
  const run = () => {
    if (used) return
    used = true
    if (target && target.now() !== was) return notify.error(target.away)
    Promise.resolve()
      .then(revert)
      .catch((e) => notify.failure('undo that', e))
  }
  if (id === undefined) notify.undo(message, run)
  else notify.undo(message, run, id)
}

export type DestroySubject = {
  /** One thing, by the name it shows under */
  name?: string
  /** Several things, counted (1 without a name reads "1 <noun>") */
  count?: number
  noun: string
  plural?: string
  /** How many of them are starred */
  starred?: number
  /** The caller's sentence on what else the delete costs */
  consequence?: string
}

const isAre = (n: number) => (n === 1 ? 'is' : 'are')
const counted = (n: number, noun: string, plural = `${noun}s`) =>
  `${n} ${n === 1 ? noun : plural}`

function starredSentence(one: boolean, starred: number): string {
  if (starred <= 0) return ''
  return one
    ? 'It is starred.'
    : `${starred} of them ${isAre(starred)} starred.`
}

export function destroyConfirm(subject: DestroySubject): ConfirmSpec {
  const n = subject.count ?? 1
  const named = subject.name !== undefined
  const one = named || n === 1
  const title = named
    ? `Delete ${subject.name}?`
    : `Delete ${counted(n, subject.noun, subject.plural)}?`
  const message = [
    `This removes ${one ? 'it' : 'them'} on disk.`,
    starredSentence(one, subject.starred ?? 0),
    subject.consequence ?? '',
  ]
    .filter(Boolean)
    .join(' ')
  return { title, message, confirmLabel: 'Delete', danger: true }
}

// The folders a workspace delete counts, in the order the sentence names
// them; a folder the server adds later is named by its own key
const WORKSPACE_FOLDERS: [string, string][] = [
  ['outputs', 'output'],
  ['assets', 'asset'],
  ['workflows', 'workflow'],
]

type Contents = Record<string, { files: number; bytes: number }>

function folderParts(contents: Contents, starred: number): string[] {
  const known = new Map(WORKSPACE_FOLDERS)
  const order = [
    ...WORKSPACE_FOLDERS.map(([key]) => key),
    ...Object.keys(contents).filter((key) => !known.has(key)),
  ]
  return order.flatMap((key) => {
    const files = Number(contents[key]?.files ?? 0)
    if (!files) return []
    const noun = known.get(key) ?? key
    const stars =
      key === 'outputs' && starred > 0 ? ` (${starred} starred)` : ''
    return [`${counted(files, noun, known.has(key) ? undefined : key)}${stars}`]
  })
}

const listed = (parts: string[]) =>
  parts.length < 2
    ? parts.join('')
    : `${parts.slice(0, -1).join(', ')} and ${parts.at(-1)}`

/** The confirm for deleting a workspace: what the server says it holds
 * (its unacknowledged refusal's `contents`), with the starred outputs */
export function workspaceConfirm(
  name: string,
  contents: Contents,
  starred: number,
): ConfirmSpec {
  const parts = folderParts(contents, starred)
  const bytes = Object.values(contents).reduce(
    (sum, folder) => sum + Number(folder?.bytes ?? 0),
    0,
  )
  const removes = parts.length
    ? `This permanently removes ${listed(parts)} (${formatBytes(bytes)}).`
    : 'This permanently removes the workspace, which holds no files.'
  return {
    title: `Delete workspace "${name}"?`,
    message: `${removes} The prompt library and the shared asset library are kept.`,
    confirmLabel: 'Delete workspace',
    danger: true,
  }
}

export type ServerSpec = {
  title: string
  message: string
  confirmLabel: string
}

/** The confirm for something that changes the server rather than destroying
 * a person's file: it says what changes and is never in danger (P5 row 3) */
export function serverConfirm(spec: ServerSpec): ConfirmSpec {
  return { ...spec, danger: false }
}

/** Deleting a repo from the hub cache: its size, and that it comes back */
export function modelDeleteConfirm(repoId: string, gb: string): ConfirmSpec {
  return serverConfirm({
    title: `Delete ${repoId} (${gb} GB) from the model cache?`,
    message: 'The next workflow that needs it downloads it again.',
    confirmLabel: 'Delete model',
  })
}

/** Clearing a dw disk cache: the title names it, the message says how
 * much goes and what the cache says that costs */
export function cacheClearConfirm(
  cache: {
    name: string
    root: string
    files: number
    cost_of_clearing: string
  },
  size: string,
): ConfirmSpec {
  return serverConfirm({
    title: `Clear the ${cache.name} cache at ${cache.root}?`,
    message: `${size}, ${counted(cache.files, 'file')}. ${cache.cost_of_clearing}`,
    confirmLabel: 'Clear',
  })
}

/** Deleting one entry of a dw disk cache; the cache says what it costs */
export function cacheEntryConfirm(
  cache: string,
  what: string,
  size: string,
  cost: string,
): ConfirmSpec {
  return serverConfirm({
    title: `Delete ${what} from the ${cache} cache?`,
    message: `${size}. ${cost}`,
    confirmLabel: 'Delete entry',
  })
}

/** Update diffusers from GitHub: it changes the server's environment */
export const UPDATE_DIFFUSERS_CONFIRM: ConfirmSpec = serverConfirm({
  title: 'Update diffusers from GitHub?',
  message:
    "This changes the server's Python environment and restarts the idle worker.",
  confirmLabel: 'Update diffusers',
})
