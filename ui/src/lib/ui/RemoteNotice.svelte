<script lang="ts">
  import { REMOTE_RELOADED } from '../editorShell.svelte'

  // The word a view that is not an editor gives when another writer (an
  // agent, another tab) changed what it shows: the editors' toast copy,
  // inline, wearing the changed-row bar that fades over CHANGED_MARK_MS.
  // `shown` is the view's resource saying so (`changed`), so the notice
  // lasts as long as the mark on every page that uses it
  let { shown, inline = false }: { shown: boolean; inline?: boolean } = $props()
</script>

{#if shown}
  <svelte:element
    this={inline ? 'span' : 'p'}
    class="remotenote row-changed"
    role="status"
  >
    {REMOTE_RELOADED}
  </svelte:element>
{/if}

<style>
  .remotenote {
    margin: 0.4rem 0;
    padding-left: 0.6rem;
  }
</style>
