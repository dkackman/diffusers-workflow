<script lang="ts">
  import type { CardReading } from './cardMemory'
  import { gbFromMb } from './format'

  // One line per card's VRAM reading. Shared by the status popover and the
  // server page; a reading taken while a job held the card is marked stale
  // with its age, since it is the run's last report rather than a measure
  let { cards }: { cards: CardReading[] } = $props()
  const gb = gbFromMb
</script>

{#if cards.some((c) => c.available)}
  <ul>
    {#each cards as c, i (c.device ?? i)}
      <li>
        <span class="card" title={c.name ?? ''}
          >{c.device ?? c.name ?? 'GPU'}</span
        >
        {#if c.available}
          <span class="num"
            >{gb(c.allocatedMb)}{#if c.totalMb}&nbsp;/ {gb(c.totalMb)}{/if} GB</span
          >
          {#if !c.totalMb}
            <span class="muted">(this backend reports allocated only)</span>
          {/if}
          {#if !c.live && c.ageSeconds != null}
            <span class="muted" title="the last reading; the card is busy"
              >{Math.round(c.ageSeconds)}s ago</span
            >
          {/if}
        {:else}
          <span class="muted">no reading yet</span>
        {/if}
      </li>
    {/each}
  </ul>
{:else}
  <span class="muted">no reading yet</span>
{/if}

<style>
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: var(--space-1);
  }
  li {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0 var(--space-3);
  }
  .card {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
  }
</style>
