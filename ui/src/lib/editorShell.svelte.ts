// What the workflow and prompt editors share about the document they edit:
// the document and the baseline its dirty flag compares against, the JSON
// view's draft, the view picked (remembered per editor), and where a save
// lands. Also whether the document loaded, the ETag it was read at and
// the preconditions a save sends from them, and the flushers a page's
// debounced inputs register so a save, a view switch or a leave sees every
// keystroke - with the field still focused inside `root`, whose change
// commits only on blur.
// Constructed during a page's component init, since it registers the
// JSON-mirror, tab-close and leave-guard effects there.
import { tick } from 'svelte'
import type { WritePreconditions } from './apiRequest'
import { confirmDialog } from './confirm.svelte'
import { isNameSegment } from './names'
import { addLeaveGuard } from './router.svelte'
import { notify } from './toast'

export type EditorView = 'form' | 'split' | 'json' | 'flow'

/** 'new' is a blank or handed-over document, 'failed' one that never
 * arrived - which must not be saved over the file it stands in for. */
export type LoadState = 'new' | 'loading' | 'loaded' | 'failed'

export type LoadOptions = {
  etag?: string | null
  path?: string | null
  writable?: boolean
}

// The views that show the JSON draft, which a parse failure keeps open
const JSON_VIEWS: readonly EditorView[] = ['json', 'split']

/** The folder picker's value for "a folder that does not exist yet". */
export const NEW_FOLDER = '__new__'

function storedView(
  key: string,
  views: readonly EditorView[],
  legacyView?: () => EditorView | null,
): EditorView {
  try {
    const stored = localStorage.getItem(key) as EditorView | null
    if (stored && views.includes(stored)) return stored
    return legacyView?.() ?? views[0]
  } catch {
    return views[0]
  }
}

type TextField = HTMLInputElement | HTMLTextAreaElement

function isTextField(el: HTMLElement): el is TextField {
  return el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement
}

function caretOf(el: HTMLElement): [number | null, number | null] | null {
  return isTextField(el) ? [el.selectionStart, el.selectionEnd] : null
}

/** The field a re-render put in place of `was`: the same id, else name. */
function successorOf(was: HTMLElement, root: HTMLElement | null) {
  const name = was.getAttribute('name')
  const selector = was.id
    ? `#${CSS.escape(was.id)}`
    : name
      ? `[name="${CSS.escape(name)}"]`
      : ''
  return selector ? root?.querySelector<HTMLElement>(selector) : null
}

export class DocumentEditor {
  doc = $state<Record<string, any>>({})
  // The document as last loaded or saved, serialized; '' until one loads,
  // so nothing reads as dirty before there is anything to compare
  baseline = $state('')
  jsonDraft = $state('')
  jsonParseFailed = $state(false)
  view = $state<EditorView>('form')
  busy = $state(false)
  saveName = $state('')
  folder = $state('')
  newFolder = $state('')
  loadState = $state<LoadState>('new')
  loadError = $state<unknown>(null)
  // The ETag the document was read (or last saved) at, the library path it
  // came from, and whether that origin takes writes in place
  etag = $state<string | null>(null)
  loadedPath = $state<string | null>(null)
  writable = $state(true)

  readonly canWrite = $derived(
    this.loadState === 'new' || this.loadState === 'loaded',
  )

  readonly dirty = $derived(
    this.baseline !== '' &&
      JSON.stringify($state.snapshot(this.doc)) !== this.baseline,
  )

  // Not reactive: nothing renders from it
  #root: HTMLElement | null = null
  #composing = false

  /** The page's editing surface: a flush commits the field focused in it.
   * Setting it follows the element's IME composition, which a blur ends. */
  get root(): HTMLElement | null {
    return this.#root
  }
  set root(el: HTMLElement | null) {
    this.#root?.removeEventListener('compositionstart', this.#compose)
    this.#root?.removeEventListener('compositionend', this.#compose)
    this.#root?.removeEventListener('focusout', this.#compose)
    this.#root = el
    this.#composing = false
    el?.addEventListener('compositionstart', this.#compose)
    el?.addEventListener('compositionend', this.#compose)
    // A field that leaves mid-composition (removed, or tabbed away) ends it
    el?.addEventListener('focusout', this.#compose)
  }

  #compose = (event: Event) => {
    this.#composing = event.type === 'compositionstart'
  }

  #viewKey: string
  // Not reactive: nothing renders from it
  #flushers: Array<() => void> = []

  constructor({
    viewKey,
    views,
    legacyView,
    label,
  }: {
    viewKey: string
    views: readonly EditorView[]
    legacyView?: () => EditorView | null
    /** The file being edited, as the leave dialog names it */
    label: () => string
  }) {
    this.#viewKey = viewKey
    this.view = storedView(viewKey, views, legacyView)

    // Mirror the document into the JSON surfaces. A failed parse pins the
    // raw text so a broken edit isn't regenerated out from under the user
    // before they can fix it
    $effect(() => {
      const pretty = JSON.stringify($state.snapshot(this.doc), null, 2)
      if (!this.jsonParseFailed) this.jsonDraft = pretty
    })

    // Unsaved edits should survive an accidental tab close. They are read
    // inside the handler only, so the listener registers exactly once
    $effect(() => {
      const guard = (event: BeforeUnloadEvent) => {
        if (this.unsaved()) event.preventDefault()
      }
      window.addEventListener('beforeunload', guard)
      return () => window.removeEventListener('beforeunload', guard)
    })

    // Every in-app exit (sidebar, Back, a link) asks while edits are unsaved
    $effect(() =>
      addLeaveGuard(
        () =>
          !this.unsaved() ||
          confirmDialog(`Leave without saving your changes to ${label()}?`, {
            confirmLabel: 'Leave',
            cancelLabel: 'Keep editing',
            danger: true,
          }),
      ),
    )
  }

  /** Run `fn` on every flush; returns its removal. */
  registerFlush(fn: () => void): () => void {
    this.#flushers.push(fn)
    return () => {
      this.#flushers = this.#flushers.filter((f) => f !== fn)
    }
  }

  /** Push every pending edit into the document now: the focused field's,
   * which a browser commits on blur (its `change`), then every debounced
   * one. The field gets its focus back once the commit has rendered, so
   * typing carries on; mid-composition it is not blurred at all. */
  flush() {
    const active = document.activeElement
    if (
      !this.#composing &&
      active instanceof HTMLElement &&
      this.#root?.contains(active)
    ) {
      active.blur()
      void this.#refocus(active)
    }
    for (const fn of [...this.#flushers]) fn()
  }

  async #refocus(was: HTMLElement) {
    const caret = caretOf(was)
    await tick()
    const node = was.isConnected ? was : successorOf(was, this.#root)
    if (!node) return
    // Focus that moved on meanwhile (a dialog, a click) is not ours to take
    const now = document.activeElement
    if (now && now !== document.body && now !== was) return
    node.focus({ preventScroll: true })
    if (caret && isTextField(node)) {
      try {
        node.setSelectionRange(caret[0], caret[1])
      } catch {
        /* a type with no selection */
      }
    }
  }

  /** Whether leaving now would lose anything: an edit not yet saved, once
   * every pending one is in, or a JSON draft that never parsed. */
  unsaved(): boolean {
    this.flush()
    return this.dirty || this.jsonParseFailed
  }

  setView(next: EditorView) {
    this.flush()
    if (this.jsonParseFailed && !JSON_VIEWS.includes(next)) {
      notify.error('Fix the JSON before switching view', 'json-parse')
      return
    }
    this.view = next
    try {
      localStorage.setItem(this.#viewKey, next)
    } catch {
      /* session only */
    }
  }

  /** Take `raw` as the document if it parses; `quiet` pins a failure
   * without the toast, for a draft still being typed. */
  applyJson(raw: string, { quiet = false } = {}) {
    this.jsonDraft = raw
    try {
      this.doc = JSON.parse(raw)
      this.jsonParseFailed = false
      notify.dismiss('json-parse')
    } catch (e) {
      this.jsonParseFailed = true
      if (!quiet)
        notify.error(
          `JSON: ${e instanceof Error ? e.message : e}`,
          'json-parse',
        )
    }
  }

  /** A read of the document has started. */
  beginLoad() {
    this.loadState = 'loading'
    this.loadError = null
  }

  /** Open `doc` as the unedited document, read at `etag` from `path`. */
  load(
    doc: Record<string, any>,
    { etag = null, path = null, writable = true }: LoadOptions = {},
  ) {
    this.doc = doc
    this.baseline = JSON.stringify(doc)
    // The last document's broken draft is not this one's: the JSON view
    // mirrors the document just opened
    this.jsonParseFailed = false
    notify.dismiss('json-parse')
    this.loadState = 'loaded'
    this.loadError = null
    this.etag = etag
    this.loadedPath = path
    this.writable = writable
  }

  /** The read failed: nothing to edit, and nothing a save may write. */
  failLoad(error: unknown) {
    this.loadState = 'failed'
    this.loadError = error
  }

  /** Open `doc` as a document no file holds yet. */
  startNew(doc: Record<string, any>) {
    this.load(doc)
    this.loadState = 'new'
  }

  /** What a save to `path` insists on: the file still at the ETag this
   * document was read at when it is that file and its origin takes writes,
   * else that nothing is there yet - a new name, or a read-only origin's
   * document whose save lands as a new copy in front of it. */
  preconditions(path: string): WritePreconditions {
    if (path !== this.loadedPath || !this.writable) return { createOnly: true }
    return this.etag ? { ifMatch: this.etag } : {}
  }

  /** The document as it is now, serialized as the baseline is. */
  serialized(): string {
    return JSON.stringify($state.snapshot(this.doc))
  }

  /** Record `baseline` (the document as it is now) as the saved one. */
  markSaved(baseline: string = this.serialized()) {
    this.baseline = baseline
  }

  /** The folder a save lands in - a new folder's typed name, trimmed. */
  directory(): string {
    return this.folder === NEW_FOLDER ? this.newFolder.trim() : this.folder
  }

  /** Where a save lands, or null while the name or a new folder's name is
   * missing or not one path segment. Pure: it renders in the save bar. */
  savePath(): string | null {
    if (!this.saveName) return null
    const directory = this.directory()
    if (this.folder === NEW_FOLDER && !isNameSegment(directory)) return null
    return directory ? `${directory}/${this.saveName}` : this.saveName
  }

  /** After a save into a new folder, that folder is the one picked. */
  commitNewFolder() {
    if (this.folder !== NEW_FOLDER) return
    this.folder = this.newFolder.trim()
    this.newFolder = ''
  }

  /** The one-shot hand-off another page left in session storage - a
   * duplicate, or a definition from image metadata - and the folder it
   * came from. Both keys are cleared; null when there is no readable one,
   * so a plain "New" stays a blank slate. */
  takeImport(importKey: string, folderKey: string): Record<string, any> | null {
    this.folder = sessionStorage.getItem(folderKey) ?? ''
    sessionStorage.removeItem(folderKey)
    const imported = sessionStorage.getItem(importKey)
    if (!imported) return null
    sessionStorage.removeItem(importKey)
    try {
      return JSON.parse(imported)
    } catch {
      return null
    }
  }
}
