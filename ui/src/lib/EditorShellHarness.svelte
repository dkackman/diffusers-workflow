<script lang="ts">
  // Test-only: constructs a DocumentEditor during component init, as the
  // editor pages do, and hands it to the test
  import { DocumentEditor, type EditorView } from './editorShell.svelte'

  let {
    viewKey,
    views,
    legacyView = undefined,
    onready,
  }: {
    viewKey: string
    views: readonly EditorView[]
    legacyView?: () => EditorView | null
    onready: (shell: DocumentEditor) => void
  } = $props()

  // svelte-ignore state_referenced_locally
  onready(new DocumentEditor({ viewKey, views, legacyView }))
</script>
