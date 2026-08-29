<script lang="ts">
  import { api } from '../api'

  let workflows = $state<string[]>([])
  let workflowDir = $state('')
  let filter = $state('')
  let error = $state('')

  $effect(() => {
    api
      .listWorkflows()
      .then((result) => {
        workflows = result.workflows
        workflowDir = result.workflow_dir
      })
      .catch((e) => (error = e.message))
  })

  const visible = $derived(
    workflows.filter((name) => name.toLowerCase().includes(filter.toLowerCase())),
  )
  const groupOf = (name: string) =>
    name.includes('/') ? name.split('/')[0] : ''
  const groups = $derived(
    [...new Set(visible.map(groupOf))].sort((a, b) => a.localeCompare(b)),
  )
</script>

<div class="head">
  <h1>Workflows</h1>
  <span class="muted">{workflowDir}</span>
  <input placeholder="filter…" bind:value={filter} class="filter" />
</div>

{#if error}
  <p class="muted">Could not load workflows: {error}</p>
{/if}

{#each groups as group}
  {#if group}<h2 class="group">{group}/</h2>{/if}
  <div class="grid">
    {#each visible.filter((name) => groupOf(name) === group) as name}
      <a class="card panel" href={'#/workflows/' + name.split('/').map(encodeURIComponent).join('/')}>
        {name.includes('/') ? name.split('/').slice(1).join('/') : name}
      </a>
    {/each}
  </div>
{/each}

<style>
  .head { display: flex; align-items: baseline; gap: 1rem; margin-bottom: 1rem; }
  .filter { max-width: 220px; margin-left: auto; }
  .group { margin: 1.2rem 0 0.5rem; color: var(--muted); }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 0.6rem; }
  .card { color: var(--ink); font-weight: 600; padding: 0.7rem 0.9rem; }
  .card:hover { border-color: var(--accent); }
</style>
