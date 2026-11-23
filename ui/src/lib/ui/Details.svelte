<script lang="ts">
  // The engine's own text behind one line of plain words: the summary shows,
  // the detail opens on demand, and Copy puts all of it on the clipboard
  import CopyButton from '../CopyButton.svelte'

  let {
    summary,
    detail,
    open = false,
  }: { summary: string; detail: string; open?: boolean } = $props()
</script>

{#if detail}
  <details class="engine-detail" {open}>
    <summary>{summary}</summary>
    <div class="body">
      <pre>{detail}</pre>
      <CopyButton text={detail} title="copy {summary.toLowerCase()}" />
    </div>
  </details>
{/if}

<style>
  .engine-detail {
    margin: 0.25rem 0 0;
    font-size: 0.85em;
  }
  summary {
    cursor: pointer;
    color: var(--muted);
  }
  .body {
    display: flex;
    gap: 0.4rem;
    align-items: flex-start;
  }
  pre {
    flex: 1;
    min-width: 0;
    margin: 0.25rem 0 0;
    font-family: var(--mono, ui-monospace, monospace);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    max-height: 14rem;
    overflow: auto;
  }
</style>
