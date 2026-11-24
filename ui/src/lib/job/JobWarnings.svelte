<script lang="ts">
  // A job's warnings: one about an argument the job was given stays in view,
  // the rest are for the workflow's author and fold away
  import { TriangleAlert } from '@lucide/svelte'
  import { placeFindings } from '../findings'
  import Details from '../ui/Details.svelte'

  let { warnings, given }: { warnings: string[]; given: string[] } = $props()

  const placed = $derived(
    placeFindings({ valid: true, errors: [], warnings, error: null }, given),
  )
  const warned = $derived(Object.values(placed.byVariable).flat())
</script>

{#if warned.length}
  <div class="panel warnings warn-edge">
    {#each warned as warning, i (i)}
      <div class="warnrow"><TriangleAlert size={14} /> {warning.text}</div>
    {/each}
  </div>
{/if}
{#if placed.author.length}
  <Details
    summary="{placed.author.length} {placed.author.length === 1
      ? 'note'
      : 'notes'} for the workflow's author"
    detail={placed.author.map((f) => f.text).join('\n')}
  />
{/if}

<style>
  .warnings {
    color: var(--warn);
  }
  .warnrow {
    display: flex;
    align-items: center;
    gap: 0.45rem;
  }
</style>
