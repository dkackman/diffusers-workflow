<script module lang="ts">
  // Quiet pause after the last keystroke before the text applies
  export const JSON_APPLY_MS = 300
</script>

<script lang="ts">
  import type * as Monaco from 'monaco-editor'

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
    register?: (flush: () => void) => () => void
    readonly?: boolean
    height?: string
    schema?: 'workflow' | 'prompt'
  } = $props()

  let container: HTMLDivElement
  let editor = $state<Monaco.editor.IStandaloneCodeEditor | null>(null)
  let loading = $state(true)
  let timer: ReturnType<typeof setTimeout> | null = null
  // The last text handed to onchange (or received from outside): an
  // unchanged value is never applied again
  let applied = ''

  function apply(quiet: boolean) {
    if (timer) clearTimeout(timer)
    timer = null
    if (readonly || !onchange || !editor) return
    const text = editor.getValue()
    if (text === applied) return
    applied = text
    if (quiet) onchange(text, { quiet: true })
    else onchange(text)
  }

  $effect(() => {
    if (readonly || !register) return
    return register(() => apply(false))
  })

  $effect(() => {
    let disposed = false
    let model: Monaco.editor.ITextModel | null = null
    let untheme: (() => void) | null = null
    import('../monaco').then(async ({ setupMonaco, currentTheme }) => {
      const monaco = await setupMonaco()
      if (disposed) return
      loading = false
      // Follow the app theme after mount: the header toggle stamps
      // data-theme on <html>, and "system" tracks the OS preference. Without
      // this the editor kept whatever theme it was created with
      const retheme = () => monaco.editor.setTheme(currentTheme())
      const observer = new MutationObserver(retheme)
      observer.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ['data-theme'],
      })
      const media = window.matchMedia('(prefers-color-scheme: dark)')
      media.addEventListener('change', retheme)
      untheme = () => {
        observer.disconnect()
        media.removeEventListener('change', retheme)
      }
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
      applied = value
      // Changes apply quietly once typing pauses, and for real on blur or a
      // flush (save, view switch)
      editor.onDidChangeModelContent(() => {
        if (readonly || !onchange) return
        if (timer) clearTimeout(timer)
        timer = setTimeout(() => apply(true), JSON_APPLY_MS)
      })
      editor.onDidBlurEditorWidget(() => apply(false))
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
      if (viewState) editor.restoreViewState(viewState)
    }
  })
</script>

<div class="jsoneditor" style:height bind:this={container}>
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
</style>
