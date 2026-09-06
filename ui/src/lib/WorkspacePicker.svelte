<script lang="ts">
  import { onMount } from 'svelte'
  import {
    loadWorkspaces,
    selectWorkspace,
    workspace,
  } from './workspace.svelte'

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
        selectWorkspace((event.currentTarget as HTMLSelectElement).value)}
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
