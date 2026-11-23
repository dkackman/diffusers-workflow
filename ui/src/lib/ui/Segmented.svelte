<script lang="ts">
  import type { Component } from 'svelte'

  // One choice from a few, side by side: the editor's view, a step's detail
  // level, the shape filter, a picker's tabs. The chosen option is the accent
  // outline and says so to a screen reader (aria-pressed). A chip that
  // several can hold at once is `.togglechip` (patterns.css), not this.
  let {
    options,
    value,
    onselect,
    label,
    size = 'md',
  }: {
    options: {
      value: string
      label: string
      title?: string
      icon?: Component<{ size?: number }>
    }[]
    value: string
    onselect: (value: string) => void
    label: string
    size?: 'sm' | 'md'
  } = $props()
</script>

<div class="segmented" class:sm={size === 'sm'} role="group" aria-label={label}>
  {#each options as option (option.value)}
    <button
      class="quiet withicon"
      aria-pressed={value === option.value}
      onclick={() => onselect(option.value)}
      title={option.title}
    >
      {#if option.icon}<option.icon size={14} />{/if}{option.label}
    </button>
  {/each}
</div>

<style>
  .segmented {
    display: inline-flex;
    flex-wrap: wrap;
    max-width: 100%;
  }
  .segmented button {
    border-radius: 0;
  }
  .segmented.sm button {
    padding: 0.2rem 0.55rem;
    font-size: var(--t-xs);
  }
  .segmented button:first-child {
    border-radius: var(--radius-1) 0 0 var(--radius-1);
  }
  .segmented button:last-child {
    border-radius: 0 var(--radius-1) var(--radius-1) 0;
  }
  .segmented button + button {
    margin-left: -1px;
  }
  .segmented button[aria-pressed='true'] {
    border-color: var(--accent);
    color: var(--accent);
    position: relative;
    z-index: 1;
  }
</style>
