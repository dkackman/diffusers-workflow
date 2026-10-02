<script lang="ts">
  import type { DigestLine } from '../digest'

  // A step's compact view: one line per area it sets, each a jump to that
  // area in the full view
  let {
    lines,
    onopen,
  }: {
    lines: DigestLine[]
    onopen: (section: string) => void
  } = $props()
</script>

<div class="digest">
  {#each lines as line (line.section)}
    <button
      class="digestline"
      onclick={() => onopen(line.section)}
      title="edit in full view"
    >
      <span class="digestsection muted">{line.section}</span>
      <span class="digesttext">{line.text}</span>
    </button>
  {:else}
    <div class="muted hint">nothing set yet - switch to full to edit</div>
  {/each}
</div>

<style>
  .digest {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    margin-top: var(--space-2);
  }
  .digestline {
    display: flex;
    gap: var(--space-2);
    align-items: baseline;
    background: transparent;
    border: 0;
    color: var(--ink);
    font-weight: 400;
    text-align: left;
    padding: 0.2rem 0.3rem;
    border-radius: var(--radius-1);
    font-size: 0.85rem;
  }
  .digestline:hover {
    background: var(--panel-2);
    filter: none;
  }
  .digestsection {
    font-size: 0.7rem;
    text-transform: none;
    flex: none;
    width: 90px;
  }
  .digesttext {
    overflow-wrap: anywhere;
  }
  .hint {
    font-size: 0.75rem;
    margin-top: 0.3rem;
  }
</style>
