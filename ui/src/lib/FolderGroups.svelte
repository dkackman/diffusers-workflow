<script lang="ts">
  import type { Snippet } from 'svelte'
  import { Plus } from '@lucide/svelte'
  import { collapsedState } from './collapsed.svelte'
  import GroupHeading from './ui/GroupHeading.svelte'
  import { groupNames, groupOf as defaultGroupOf } from './grouping'

  let {
    names,
    collapseKey,
    filterActive,
    newHref = undefined,
    onnewingroup = undefined,
    minColumn = '210px',
    groupOf = defaultGroupOf,
    card,
  }: {
    names: string[]
    collapseKey: string
    filterActive: boolean
    newHref?: string
    onnewingroup?: (group: string) => void
    minColumn?: string
    /** Which folder a name belongs to. The gallery passes its own: its
     * names carry a run id that the server has already folded into `folder`. */
    groupOf?: (name: string) => string
    card: Snippet<[string]>
  } = $props()

  // svelte-ignore state_referenced_locally
  const collapsed = collapsedState(collapseKey)

  const grouped = $derived(groupNames(names, groupOf))
  const groups = $derived([...grouped.keys()])
  const inGroup = (group: string) => grouped.get(group) ?? []
  // While filtering, everything stays visible - a collapsed folder hiding
  // matches would make the filter look broken
  const isOpen = (group: string) => filterActive || !collapsed.has(group)
</script>

{#each groups as group (group)}
  {#if group}
    <div class="grouprow">
      <GroupHeading
        label="{group}/"
        count={inGroup(group).length}
        collapsed={!isOpen(group)}
        ontoggle={() => collapsed.toggle(group)}
        title={isOpen(group) ? 'collapse this folder' : 'expand this folder'}
      />
      {#if newHref}
        <a
          class="groupnew"
          href={newHref}
          onclick={() => onnewingroup?.(group)}
          title="new in {group}/"
          aria-label="new in {group}/"
        >
          <Plus size={13} />
        </a>
      {/if}
    </div>
  {/if}
  {#if isOpen(group)}
    <div
      class="grid"
      style="grid-template-columns: repeat(auto-fill, minmax({minColumn}, 1fr))"
    >
      {#each inGroup(group) as name (name)}
        {@render card(name)}
      {/each}
    </div>
  {/if}
{/each}

<style>
  .grouprow {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin: 1.2rem 0 var(--space-2);
  }
  .groupnew {
    display: inline-flex;
    align-items: center;
    padding: 0.15rem;
    color: var(--muted);
    border: 1px solid transparent;
    border-radius: 4px;
    opacity: 0;
    transition: opacity 0.15s ease;
  }
  .grouprow:hover .groupnew {
    opacity: 1;
  }
  .groupnew:hover {
    color: var(--ink);
    border-color: var(--line);
  }
  /* Cards are deliberately not equal height - one that has produced
     something carries a picture and one that has not does not - so rows
     align to the top rather than stretching the short ones */
  .grid {
    display: grid;
    align-items: start;
    gap: 0.6rem;
  }
</style>
