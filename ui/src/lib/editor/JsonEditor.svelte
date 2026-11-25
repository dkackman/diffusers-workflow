<script module lang="ts">
  // Quiet pause after the last keystroke before the text applies
  export const JSON_APPLY_MS = 300
</script>

<script lang="ts">
  import type * as Monaco from 'monaco-editor'
  import { describeError } from '../apiErrors'
  import type { Flusher } from '../editorShell.svelte'
  import Details from '../ui/Details.svelte'

  let {
    value,
    onchange,
    register,
    readonly = false,
    height = '460px',
    schema = 'workflow',
  }: {
    value: string
    onchange?: (value: string, options?: { quiet?: boolean }) => void
    register?: (flush: Flusher) => () => void
    readonly?: boolean
    height?: string
    schema?: 'workflow' | 'prompt'
  } = $props()

  let container: HTMLDivElement
  let editor = $state<Monaco.editor.IStandaloneCodeEditor | null>(null)
  let loading = $state(true)
  // Monaco's chunk never arrived - a tab opened before the server was
  // upgraded asks for one the new build no longer has
  let failed = $state<unknown>(null)
  let timer: ReturnType<typeof setTimeout> | null = null
  // The last text handed to onchange (or received from outside): an
  // unchanged value is never applied again
  let applied = ''
  // Whether that text went out quietly: a quiet apply of broken JSON shows
  // no toast, so the real apply on blur or flush must still run
  let appliedQuiet = false

  function apply(quiet: boolean) {
    if (timer) clearTimeout(timer)
    timer = null
    if (readonly || !onchange || !editor) return
    const text = editor.getValue()
    if (text === applied && (quiet || !appliedQuiet)) return
    applied = text
    appliedQuiet = quiet
    if (quiet) onchange(text, { quiet: true })
    else onchange(text)
  }

  $effect(() => {
    if (readonly || !register) return
    // A quiet flush (a re-read checking for edits) never toasts a draft
    return register((options) => apply(options?.quiet === true))
  })

  // Follow the app theme after mount: the header toggle stamps data-theme
  // on <html>, and "system" tracks the OS preference. Without this the
  // editor kept whatever theme it was created with
  function followTheme(retheme: () => void): () => void {
    const observer = new MutationObserver(retheme)
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    })
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    media.addEventListener('change', retheme)
    return () => {
      observer.disconnect()
      media.removeEventListener('change', retheme)
    }
  }

  function watchEdits(created: Monaco.editor.IStandaloneCodeEditor) {
    applied = value
    appliedQuiet = false
    // Changes apply quietly once typing pauses, and for real on blur or a
    // flush (save, view switch)
    created.onDidChangeModelContent(() => {
      if (readonly || !onchange) return
      if (timer) clearTimeout(timer)
      timer = setTimeout(() => apply(true), JSON_APPLY_MS)
    })
    created.onDidBlurEditorWidget(() => apply(false))
  }

  $effect(() => {
    let disposed = false
    let model: Monaco.editor.ITextModel | null = null
    let untheme: (() => void) | null = null
    import('../monaco')
      .then(async ({ setupMonaco, currentTheme }) => {
        const monaco = await setupMonaco()
        if (disposed) return
        loading = false
        untheme = followTheme(() => monaco.editor.setTheme(currentTheme()))
        // The model's name is what routes the schema to it - setupMonaco
        // registers one schema per prefix. The random suffix keeps two open
        // editors (the split view) from fighting over one URI
        model = monaco.editor.createModel(
          value,
          'json',
          monaco.Uri.parse(
            `inmemory://dw/${schema}-${Math.random().toString(36).slice(2)}.json`,
          ),
        )
        editor = monaco.editor.create(container, {
          model,
          theme: currentTheme(),
          readOnly: readonly,
          minimap: { enabled: false },
          guides: { indentation: false },
          automaticLayout: true,
          scrollBeyondLastLine: false,
          wordWrap: 'on',
          fontSize: 13,
          tabSize: 2,
          fixedOverflowWidgets: true,
        })
        watchEdits(editor)
      })
      .catch((e: unknown) => {
        if (disposed) return
        loading = false
        failed = e
      })
    return () => {
      disposed = true
      if (timer) clearTimeout(timer)
      timer = null
      untheme?.()
      editor?.dispose()
      editor = null
      model?.dispose()
      model = null
    }
  })

  // The outside world replaced the document (form edits, new workflow).
  // setValue resets scroll and cursor, which is jarring for the split
  // view's constant refreshes - restore the view state around it.
  $effect(() => {
    if (editor && editor.getValue() !== value && !editor.hasTextFocus()) {
      const viewState = editor.saveViewState()
      editor.setValue(value)
      applied = value
      appliedQuiet = false
      if (viewState) editor.restoreViewState(viewState)
    }
  })
</script>

{#if failed}
  {@const described = describeError(failed)}
  <div class="failed" role="alert">
    <p>
      The JSON editor couldn't load. Reload the page; if that doesn't help,
      check that the server is running.
    </p>
    <Details
      summary="What failed"
      detail={described.detail ?? described.title}
    />
  </div>
{/if}
<div class="jsoneditor" style:height hidden={!!failed} bind:this={container}>
  {#if loading}<span class="muted loading">loading editor…</span>{/if}
</div>

<style>
  .jsoneditor {
    border: 1px solid var(--line);
    border-radius: 6px;
    overflow: hidden;
    background: var(--panel);
  }
  .loading {
    display: block;
    padding: 1rem;
  }
  .failed {
    color: var(--bad);
  }
  .failed p {
    margin: 0;
  }
</style>
