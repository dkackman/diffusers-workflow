<script lang="ts">
  import type { Snippet } from 'svelte'
  import { ChevronDown, ChevronRight, Plus } from '@lucide/svelte'
  import { storageGet, storageSet } from './storage'

  let {
    names,
    collapseKey,
    filterActive,
    newHref = undefined,
    onnewingroup = undefined,
    card,
  }: {
    names: string[]
    collapseKey: string
    filterActive: boolean
    newHref?: string
    onnewingroup?: (group: string) => void
    card: Snippet<[string]>
  } = $props()

  let collapsed = $state<Record<string, boolean>>(storageGet(collapseKey, {}))

  function toggle(group: string) {
    collapsed[group] = !collapsed[group]
    storageSet(collapseKey, $state.snapshot(collapsed))
  }

  const groupOf = (name: string) =>
    name.includes('/') ? name.split('/')[0] : ''
  const groups = $derived(
    [...new Set(names.map(groupOf))].sort((a, b) => a.localeCompare(b)),
  )
  const inGroup = (group: string) =>
    names.filter((name) => groupOf(name) === group)
  // While filtering, everything stays visible - a collapsed folder hiding
  // matches would make the filter look broken
  const isOpen = (group: string) => filterActive || !collapsed[group]
</script>

{#each groups as group (group)}
  {#if group}
    <div class="grouprow">
      <button
        class="group"
        onclick={() => toggle(group)}
        title={isOpen(group) ? 'collapse this folder' : 'expand this folder'}
      >
        {#if isOpen(group)}<ChevronDown size={14} />{:else}<ChevronRight
            size={14}
          />{/if}
        {group}/ <span class="muted">({inGroup(group).length})</span>
      </button>
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
    <div class="grid">
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
    color: var(--accent);
    border-color: var(--line);
  }
  .group {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    background: none;
    border: none;
    color: var(--muted);
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0;
    margin: 0;
    cursor: pointer;
  }
  .group:hover {
    color: var(--ink);
    filter: none;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 0.6rem;
  }
</style>
