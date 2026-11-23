<script lang="ts">
  // Test-only: constructs a DocumentEditor during component init, as the
  // editor pages do, and hands it to the test
  import { DocumentEditor, type EditorView } from './editorShell.svelte'

  let {
    viewKey,
    views,
    legacyView = undefined,
    label,
    remote = undefined,
    onready,
  }: {
    viewKey: string
    views: readonly EditorView[]
    legacyView?: () => EditorView | null
    label: () => string
    /** Passed to watchRemote, as a page's open path does */
    remote?: Parameters<DocumentEditor['watchRemote']>[0]
    onready: (shell: DocumentEditor) => void
  } = $props()

  // svelte-ignore state_referenced_locally
  const shell = new DocumentEditor({ viewKey, views, legacyView, label })
  // svelte-ignore state_referenced_locally
  if (remote) shell.watchRemote(remote)
  // svelte-ignore state_referenced_locally
  onready(shell)
</script>
