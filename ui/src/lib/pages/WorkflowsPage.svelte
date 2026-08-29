<script lang="ts">
  import { ChevronDown, ChevronRight, Plus, X } from 'lucide-svelte'
  import { api } from '../api'

  let workflows = $state<string[]>([])
  let workflowDir = $state('')
  let filter = $state('')
  let error = $state('')

  const STORAGE_KEY = 'dw-collapsed-folders'

  function readCollapsed(): Record<string, boolean> {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}')
    } catch {
      return {}
    }
  }

  let collapsed = $state<Record<string, boolean>>(readCollapsed())

  let showHint = $state(
    (() => {
      try {
        return localStorage.getItem('dw-hint-dismissed') !== '1'
      } catch {
        return true
      }
    })(),
  )

  function dismissHint() {
    showHint = false
    try {
      localStorage.setItem('dw-hint-dismissed', '1')
    } catch {
      /* session-only dismissal */
    }
  }

  function toggle(group: string) {
    collapsed[group] = !collapsed[group]
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(collapsed))
    } catch {
      /* private mode etc. - collapse still works for the session */
    }
  }

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
  const groupOf = (name: string) => (name.includes('/') ? name.split('/')[0] : '')
  const groups = $derived(
    [...new Set(visible.map(groupOf))].sort((a, b) => a.localeCompare(b)),
  )
  const inGroup = (group: string) => visible.filter((name) => groupOf(name) === group)
  // While filtering, everything stays visible - a collapsed folder hiding
  // matches would make the filter look broken
  const isOpen = (group: string) => filter !== '' || !collapsed[group]

  const href = (name: string) =>
    '#/workflows/' + name.split('/').map(encodeURIComponent).join('/')
</script>

<div class="head">
  <h1>Workflows</h1>
  <span class="muted">{workflowDir}</span>
  <input placeholder="filter…" bind:value={filter} class="filter" />
  <a class="newlink" href="#/edit" title="new workflow"><Plus size={15} /></a>
</div>

{#if error}
  <p class="muted">Could not load workflows: {error}</p>
{/if}

{#if showHint}
  <div class="hintbar muted">
    <span>Pick a workflow → tweak its arguments → Run. Every image saves its recipe — reopen it from the Gallery.</span>
    <button class="quiet icon" onclick={dismissHint} title="dismiss" aria-label="dismiss this hint">
      <X size={13} />
    </button>
  </div>
{/if}

{#each groups as group (group)}
  {#if group}
    <button
      class="group"
      onclick={() => toggle(group)}
      title={isOpen(group) ? 'collapse this folder' : 'expand this folder'}
    >
      {#if isOpen(group)}<ChevronDown size={14} />{:else}<ChevronRight size={14} />{/if}
      {group}/ <span class="muted">({inGroup(group).length})</span>
    </button>
  {/if}
  {#if isOpen(group)}
    <div class="grid">
      {#each inGroup(group) as name (name)}
        <a class="card panel" href={href(name)}>
          {group ? name.split('/').slice(1).join('/') : name}
        </a>
      {/each}
    </div>
  {/if}
{/each}

<style>
  .head { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }
  .filter { max-width: 220px; margin-left: auto; }
  .newlink {
    display: inline-flex; align-items: center; padding: 0.4rem;
    border: 1px solid var(--line); border-radius: 6px; color: var(--muted);
  }
  .newlink:hover { border-color: var(--accent); color: var(--accent); }
  .hintbar {
    display: flex; align-items: center; gap: 0.8rem;
    border: 1px dashed var(--line); border-radius: 6px;
    padding: 0.45rem 0.7rem; font-size: 0.85rem; margin-bottom: 1rem;
  }
  .hintbar span { flex: 1; }
  .hintbar .icon { display: inline-flex; padding: 0.2rem 0.3rem; }
  .group {
    display: flex; align-items: center; gap: 0.35rem;
    background: none; border: none; color: var(--muted); font-weight: 600;
    font-size: 0.95rem; padding: 0; margin: 1.2rem 0 0.5rem; cursor: pointer;
  }
  .group:hover { color: var(--ink); filter: none; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 0.6rem; }
  .card { color: var(--ink); font-weight: 600; padding: 0.7rem 0.9rem; }
  .card:hover { border-color: var(--accent); }
</style>
