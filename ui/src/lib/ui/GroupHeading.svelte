<script lang="ts">
  import type { Snippet } from 'svelte'
  import { ChevronDown, ChevronRight } from '@lucide/svelte'

  // The heading of a section that can be shut: a button that says whether
  // it is open. A folder name is a path segment the engine resolves, so it
  // is mono like every other name; a `library` is the heavier top level.
  let {
    label,
    count,
    collapsed,
    ontoggle,
    title,
    size = 'folder',
    children,
  }: {
    label: string
    count?: number
    collapsed: boolean
    ontoggle: () => void
    title?: string
    size?: 'folder' | 'library'
    children?: Snippet
  } = $props()
</script>

<button
  class="group {size}"
  aria-expanded={!collapsed}
  onclick={ontoggle}
  title={title ?? (collapsed ? 'expand this section' : 'collapse this section')}
>
  {#if collapsed}<ChevronRight size={14} />{:else}<ChevronDown size={14} />{/if}
  {label}
  {#if count != null}<span class="muted">({count})</span>{/if}
  {@render children?.()}
</button>

<style>
  .group {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    background: none;
    border: none;
    color: var(--muted);
    font-family: var(--font-mono);
    font-weight: 600;
    font-size: var(--t-sm);
    letter-spacing: -0.01em;
    padding: 0;
    margin: 0;
    cursor: pointer;
  }
  .group:hover {
    color: var(--ink);
    filter: none;
  }
  .group.library {
    color: var(--ink);
    font-size: var(--t-md);
    letter-spacing: normal;
  }
  .group.library:hover {
    color: var(--accent);
  }
</style>
