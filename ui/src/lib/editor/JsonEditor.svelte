<script lang="ts">
  import type * as Monaco from 'monaco-editor'

  let {
    value,
    onchange,
    readonly = false,
    height = '460px',
  }: {
    value: string
    onchange?: (value: string) => void
    readonly?: boolean
    height?: string
  } = $props()

  let container: HTMLDivElement
  let editor = $state<Monaco.editor.IStandaloneCodeEditor | null>(null)
  let loading = $state(true)

  $effect(() => {
    let disposed = false
    import('../monaco').then(async ({ setupMonaco, currentTheme }) => {
      const monaco = await setupMonaco()
      if (disposed) return
      loading = false
      editor = monaco.editor.create(container, {
        value,
        language: 'json',
        theme: currentTheme(),
        readOnly: readonly,
        minimap: { enabled: false },
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
      editor?.dispose()
      editor = null
    }
  })

  // The outside world replaced the document (form edits, new workflow)
  $effect(() => {
    if (editor && editor.getValue() !== value && !editor.hasTextFocus()) {
      editor.setValue(value)
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
