<script lang="ts">
  import { Ellipsis } from '@lucide/svelte'
  import Popover from '../ui/Popover.svelte'
  import type { MenuEntry } from './itemMenu'

  // The "more actions" button of a row, or of the request's header, and
  // the list it opens. What the list holds is the caller's (itemMenu.ts)
  let {
    label,
    entries,
    onselect,
  }: {
    /** Names the button and the list: "More actions for {title}" */
    label: string
    entries: MenuEntry[]
    /** An action entry was chosen, by its id */
    onselect: (id: string) => void
  } = $props()

  let open = $state(false)
  let trigger = $state<HTMLButtonElement | null>(null)

  function choose(id: string) {
    open = false
    onselect(id)
  }
</script>

<button
  type="button"
  class="bare icon more"
  bind:this={trigger}
  aria-label={label}
  title={label}
  aria-haspopup="dialog"
  aria-expanded={open}
  onclick={() => (open = !open)}
>
  <Ellipsis size={16} aria-hidden="true" />
</button>
<Popover bind:open {label} anchor={trigger} align="end" menu>
  <ul class="entries">
    {#each entries as entry, i (i)}
      {#if entry.kind === 'rule'}
        <li class="rule" role="separator"></li>
      {:else if entry.kind === 'link'}
        <li>
          <a class="plain" href={entry.href} onclick={() => (open = false)}
            >{entry.label}</a
          >
        </li>
      {:else}
        <li>
          <button
            type="button"
            class="bare"
            class:danger={entry.danger}
            disabled={entry.disabled}
            onclick={() => choose(entry.id)}>{entry.label}</button
          >
        </li>
      {/if}
    {/each}
  </ul>
</Popover>

<style>
  .more {
    color: var(--muted);
  }
  @media (pointer: coarse) {
    .more {
      min-width: 44px;
      min-height: 44px;
    }
  }
  .entries {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
  }
  .entries button,
  .entries a {
    display: block;
    width: 100%;
    padding: 0.35rem 0.6rem;
    border-radius: var(--radius-1);
    color: var(--ink);
    font-size: var(--t-sm);
    font-weight: 500;
    text-align: left;
  }
  .entries a:hover {
    background: var(--panel-2);
  }
  .rule {
    border-top: 1px solid var(--line);
    margin: 0.25rem 0;
  }
  @media (pointer: coarse) {
    .entries button,
    .entries a {
      padding-block: 0.7rem;
    }
  }
</style>
