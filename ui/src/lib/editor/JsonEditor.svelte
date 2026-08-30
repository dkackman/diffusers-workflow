<script lang="ts">
  import type * as Monaco from 'monaco-editor'

  let {
    value,
    onchange,
    readonly = false,
    height = '460px',
    schema = 'workflow',
  }: {
    value: string
    onchange?: (value: string) => void
    readonly?: boolean
    height?: string
    schema?: 'workflow' | 'prompt'
  } = $props()

  let container: HTMLDivElement
  let editor = $state<Monaco.editor.IStandaloneCodeEditor | null>(null)
  let loading = $state(true)

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
        fontSize: 13,
        tabSize: 2,
        fixedOverflowWidgets: true,
      })
      // Changes apply on blur, matching the form/JSON handoff everywhere else
      editor.onDidBlurEditorWidget(() => {
        if (!readonly && onchange) onchange(editor!.getValue())
      })
    })
    return () => {
      disposed = true
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
