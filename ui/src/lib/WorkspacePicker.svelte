<script lang="ts">
  import { onMount } from 'svelte'
  import { loadWorkspaces, workspace } from './workspace.svelte'
  import { go, route } from './router.svelte'

  onMount(loadWorkspaces)

  // Only worth showing once there is a choice to make: a server with one
  // workspace should not grow a control that can only pick it
  const choices = $derived(workspace.names ?? [])
</script>

{#if choices.length > 1}
  <label class="picker">
    <span class="muted">workspace</span>
    <select
      value={workspace.current}
      onchange={(event) =>
        go(
          'ws',
          (event.currentTarget as HTMLSelectElement).value,
          route.view.kind === 'ws' ? route.view.section : 'overview',
        )}
      title="which workspace's workflows and outputs to show"
    >
      {#each choices as name (name)}
        <option value={name}>{name}</option>
      {/each}
    </select>
  </label>
{/if}

<style>
  .picker {
    display: inline-flex;
    align-items: center;
    gap: 0.4em;
    font-size: 0.9em;
  }
</style>
