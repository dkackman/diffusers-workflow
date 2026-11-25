// A document editor's place in the route: which file the URL names, which
// one is on screen, and the load that brings the second to the first. The
// workflow and prompt editors share it, so the rule that only the route
// starts a load - never a read made along the way - is written once.
import { untrack } from 'svelte'
import type { DocumentEditor } from './editorShell.svelte'
import { groupOf, leafOf } from './grouping'
import { renameRoute } from './router.svelte'

export interface DocumentRouteOptions {
  /** The file the route names; empty for a new document */
  name: () => string
  /** What a file belongs to beyond its path - a workflow's workspace. A
   * change of it reads the route's file again, or takes the draft back up. */
  scope?: () => string
  /** The hash of `path`, for the URL to follow a save */
  href: (path: string) => string
  /** Read `path` into the editor: false when the read failed, or `stale`
   * says the route moved on while it was out. */
  open: (path: string, stale: () => boolean) => Promise<boolean>
  /** Begin a new document: a hand-off's, else the blank starter */
  startDraft: () => void
  /** Run before every look at the route, whether or not it loads */
  refresh?: () => void
}

export class DocumentRoute {
  #editor: DocumentEditor
  #options: DocumentRouteOptions
  // The scope and file of the document on screen, so the route catching up
  // with a save does not read it again
  #opened = ''
  #reads = 0

  /** The path being edited: the route's, until a save gives the document
   * another (or its first) - the URL follows a beat later */
  readonly current: string = $derived.by(() => {
    const { loadState, loadedPath } = this.#editor
    const name = this.#options.name()
    return loadState === 'loading' || loadState === 'failed'
      ? name
      : (loadedPath ?? name)
  })

  /** Made in a component's setup: its effect lives as long as the page. */
  constructor(editor: DocumentEditor, options: DocumentRouteOptions) {
    this.#editor = editor
    this.#options = options
    $effect(() => {
      // Only the route and the scope start a load. The rest is untracked:
      // what a load reads on the way - the editor's own path among it -
      // would otherwise run this again when a first save names the
      // document, and start the draft over the one just saved
      const name = options.name()
      const key = this.#key(name)
      untrack(() => this.#arrive(name, key))
    })
  }

  #key(path: string): string {
    return `${this.#options.scope?.() ?? ''}:${path}`
  }

  #arrive(name: string, key: string) {
    this.#options.refresh?.()
    // the route catching up with a save: this is the document already
    if (name && this.#opened === key) return
    // a new read has begun: until it lands nothing is opened, so a return
    // to the previous file reads it again instead of trusting the key
    this.#opened = ''
    this.#reads++
    if (!name) {
      this.#options.startDraft()
      return
    }
    this.#editor.saveName = leafOf(name)
    this.#editor.folder = groupOf(name)
    void this.#open(name, key)
  }

  async #open(path: string, key: string) {
    const read = ++this.#reads
    const stale = () => path !== this.#options.name()
    const landed = await this.#options.open(path, stale)
    // the route may have moved on in the beat after the read landed: only
    // the latest read says what is on screen
    if (landed && read === this.#reads) this.#opened = key
  }

  /** Read the route's file again: a failed load's Retry */
  retry(): Promise<void> {
    const name = this.#options.name()
    this.#opened = ''
    return this.#open(name, this.#key(name))
  }

  /** A save wrote `path`: it is the document on screen, and the URL names
   * it - a first save, or a save as. */
  saved(path: string) {
    this.#opened = this.#key(path)
    if (path !== this.#options.name()) renameRoute(this.#options.href(path))
  }
}
