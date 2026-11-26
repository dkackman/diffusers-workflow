// The draft both note editors share - NoteEditor (the user's note on a
// workspace, asset, run or file) and ItemNote (a note for the agent on a
// request): what is in the box, kept in the tab's sessionStorage so a
// reload or an unmount does not lose it, the keys (Ctrl/Cmd+Enter saves,
// Escape closes and keeps the draft), the one discard question and the
// character count near the limit. Beside a kept draft goes the note it
// answered (`base`), so an editor can tell a draft resumed over a newer
// note. What a save does stays the editor's.
import { untrack } from 'svelte'
import { confirmDialog } from './confirm.svelte'
import { draftAtRisk, keepDraft, readDraft } from './draftStorage'

/** The server's limit on a note, and where the count starts showing */
export const NOTE_MAX = 4000
const COUNT_FROM = NOTE_MAX * 0.9

/** The draft key of the user's note on `target`, read in `workspace`.
 * Request notes keep their own keys (`request/draft.ts`: `draftKey`). */
export const noteKey = (workspace: string, target: string) =>
  `note:${workspace}:${target}`

/** Where the note a kept draft answered is kept, beside the draft */
const baseKey = (key: string | null) => (key ? `${key}#base` : null)

function keep(key: string | null, text: string | null, base: string | null) {
  keepDraft(key, text)
  keepDraft(baseKey(key), text === null ? null : base)
}

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
  /** The box was closed over words that are not saved, and the draft is
   * set aside */
  #aside = $state(false)
  /** The saved note a draft set aside answered, as it was when the box
   * closed */
  #base = $state<string | null>(null)
  #key: () => string | null
  #saved: () => string
  #save: (text: string) => Promise<void>
  #onclose?: () => void
  /** Set aside and still not the note as saved now: a draft equal to it,
   * after a save went through, is not kept */
  readonly kept: boolean

  /** Construct during a component's setup: the storage follows `text` in
   * an effect */
  constructor(options: NoteDraftOptions) {
    const { key } = options
    this.#key = typeof key === 'function' ? key : () => key
    this.#saved = options.saved
    this.#save = options.save
    this.#onclose = options.onclose
    this.kept = $derived(this.#aside && draftAtRisk(this.text, this.#saved()))
    let last = untrack(this.#key)
    untrack(() => this.#restore(last))
    $effect(() => {
      const key = this.#key()
      const kept = (this.editing || this.kept) && this.atRisk
      if (key !== last) {
        const old = last
        last = key
        if (options.rekey !== 'move') return untrack(() => this.#restore(key))
        keep(old, null, null)
      }
      keep(key, kept ? this.text : null, this.base)
    })
  }

  /** A draft left under `key` comes back set aside, unless it is already
   * what is saved */
  #restore(key: string | null) {
    this.editing = false
    const left = readDraft(key)
    const atRisk = left !== null && draftAtRisk(left, this.#saved())
    this.text = atRisk ? left : ''
    this.#aside = atRisk
    this.#base = atRisk ? readDraft(baseKey(key)) : null
  }

  /** The saved note the draft answers: the note as saved while the box is
   * open; for a draft set aside, the note as it was when the box closed -
   * null when that is not known (nothing is set aside, or the draft was
   * kept before its note was) */
  get base(): string | null {
    return this.editing ? this.#saved() : this.#base
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
    this.#aside = false
    this.editing = true
  }

  /** Close the box, keeping words that are not saved for the next open */
  close() {
    this.#base = this.atRisk ? this.#saved() : null
    this.#aside = this.atRisk
    this.editing = false
    this.#onclose?.()
  }

  /** The save of `sent` went through: nothing is kept - unless words were
   * typed (or Escape pressed) while it was out, which stay as the draft
   * over the note now saved. True when the box was emptied */
  done(sent: string): boolean {
    if (this.text !== sent) {
      // A draft set aside meanwhile now answers the note just saved
      this.#base = sent.trim() ? sent : ''
      return false
    }
    this.#aside = false
    this.#base = null
    this.text = ''
    this.editing = false
    return true
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
    this.#aside = false
    this.#base = null
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
