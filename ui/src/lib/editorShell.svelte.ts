// What the workflow and prompt editors share about the document they edit:
// the document and the baseline its dirty flag compares against, the JSON
// view's draft, the view picked (remembered per editor), and where a save
// lands. Constructed during a page's component init, since it registers
// the JSON-mirror and tab-close effects there.
import { isNameSegment } from './names'
import { notify } from './toast'

export type EditorView = 'form' | 'split' | 'json' | 'flow'

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

  readonly dirty = $derived(
    this.baseline !== '' &&
      JSON.stringify($state.snapshot(this.doc)) !== this.baseline,
  )

  #viewKey: string

  constructor({
    viewKey,
    views,
    legacyView,
  }: {
    viewKey: string
    views: readonly EditorView[]
    legacyView?: () => EditorView | null
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

    // Unsaved edits should survive an accidental tab close. dirty is read
    // inside the handler only, so the listener registers exactly once
    $effect(() => {
      const guard = (event: BeforeUnloadEvent) => {
        if (this.dirty) event.preventDefault()
      }
      window.addEventListener('beforeunload', guard)
      return () => window.removeEventListener('beforeunload', guard)
    })
  }

  setView(next: EditorView) {
    this.view = next
    try {
      localStorage.setItem(this.#viewKey, next)
    } catch {
      /* session only */
    }
  }

  applyJson(raw: string) {
    this.jsonDraft = raw
    try {
      this.doc = JSON.parse(raw)
      this.jsonParseFailed = false
      notify.dismiss('json-parse')
    } catch (e) {
      this.jsonParseFailed = true
      notify.error(`JSON: ${e instanceof Error ? e.message : e}`, 'json-parse')
    }
  }

  /** Open `doc` as the unedited document. */
  load(doc: Record<string, any>) {
    this.doc = doc
    this.baseline = JSON.stringify(doc)
  }

  /** Record the document as it is now as the saved one. */
  markSaved() {
    this.baseline = JSON.stringify($state.snapshot(this.doc))
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
