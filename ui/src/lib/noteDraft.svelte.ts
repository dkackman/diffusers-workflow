// The draft both note editors share - NoteEditor (the user's note on a
// workspace, asset, run or file) and ItemNote (a note for the agent on a
// request): what is in the box, kept in the tab's sessionStorage so a
// reload or an unmount does not lose it, the keys (Ctrl/Cmd+Enter saves,
// Escape closes and keeps the draft), the one discard question and the
// character count near the limit. What a save does stays the editor's.
import { untrack } from 'svelte'
import { confirmDialog } from './confirm.svelte'
import { draftAtRisk, keepDraft, readDraft } from './request/draft'

/** The server's limit on a note, and where the count starts showing */
export const NOTE_MAX = 4000
const COUNT_FROM = NOTE_MAX * 0.9

/** The draft key of the user's note on `target`, read in `workspace`.
 * Request notes keep their own keys (`draft.ts`: `draftKey`). */
export const noteKey = (workspace: string, target: string) =>
  `note:${workspace}:${target}`

export type NoteDraftOptions = {
  /** Where the draft is kept; a function when it can change, null for
   * nowhere */
  key: string | null | (() => string | null)
  /** The note as saved, '' for none */
  saved: () => string
  /** Store the text: Ctrl/Cmd+Enter calls it */
  save: (text: string) => Promise<void>
  /** The box was closed by Escape */
  onclose?: () => void
  /** What a new key means: `'switch'` (the default) is another note, so
   * the box closes and that note's own draft is read; `'move'` is the same
   * draft under a key that grew a reference, so it goes along */
  rekey?: 'switch' | 'move'
}

export class NoteDraft {
  /** What is in the box */
  text = $state('')
  editing = $state(false)
  /** The box was closed over words that are not saved: still here */
  kept = $state(false)
  #key: () => string | null
  #saved: () => string
  #save: (text: string) => Promise<void>
  #onclose?: () => void

  /** Construct during a component's setup: the storage follows `text` in
   * an effect */
  constructor(options: NoteDraftOptions) {
    const { key } = options
    this.#key = typeof key === 'function' ? key : () => key
    this.#saved = options.saved
    this.#save = options.save
    this.#onclose = options.onclose
    let last = untrack(this.#key)
    untrack(() => this.#restore(last))
    $effect(() => {
      const key = this.#key()
      const keep = (this.editing || this.kept) && this.atRisk
      if (key !== last) {
        const old = last
        last = key
        if (options.rekey !== 'move') return untrack(() => this.#restore(key))
        keepDraft(old, null)
      }
      keepDraft(key, keep ? this.text : null)
    })
  }

  /** A draft left under `key` comes back set aside, unless it is already
   * what is saved */
  #restore(key: string | null) {
    this.editing = false
    const left = readDraft(key)
    const atRisk = left !== null && draftAtRisk(left, this.#saved())
    this.text = atRisk ? left : ''
    this.kept = atRisk
  }

  /** Closing the box now would lose words */
  get atRisk(): boolean {
    return draftAtRisk(this.text, this.#saved())
  }

  /** `<n> / 4000` near the limit, else null */
  get counter(): string | null {
    const n = this.text.length
    return n >= COUNT_FROM ? `${n} / ${NOTE_MAX}` : null
  }

  /** Open the box: a draft set aside comes back, else the saved note */
  open() {
    if (this.editing) return
    if (!this.kept) this.text = this.#saved()
    this.kept = false
    this.editing = true
  }

  /** Close the box, keeping words that are not saved for the next open */
  close() {
    this.kept = this.atRisk
    this.editing = false
    this.#onclose?.()
  }

  /** The save went through: nothing is kept */
  done() {
    this.kept = false
    this.text = ''
    this.editing = false
  }

  /** Discard the draft, asking first when it holds words not saved.
   * False when the user kept editing. */
  async cancel(): Promise<boolean> {
    if (
      this.atRisk &&
      !(await confirmDialog('Discard this draft of your note?', {
        confirmLabel: 'Discard',
        cancelLabel: 'Keep editing',
        danger: true,
      }))
    )
      return false
    this.kept = false
    this.text = ''
    this.editing = false
    return true
  }

  keydown(event: KeyboardEvent) {
    if (event.key === 'Escape') {
      event.stopPropagation()
      this.close()
    } else if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault()
      void this.#save(this.text)
    }
  }
}
