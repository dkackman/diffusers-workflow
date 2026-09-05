<script lang="ts">
  import { Film, Image, Layers, Music, Plus } from '@lucide/svelte'
  import { api } from '../api'
  import Empty from '../Empty.svelte'
  import FolderGroups from '../FolderGroups.svelte'
  import HintBar from '../HintBar.svelte'

  let workflows = $state<string[]>([])
  let details = $state<
    Record<
      string,
      {
        kinds: string[]
        steps?: number
        variables: number
        description: string
        /** Which source the workflow was read from. */
        origin?: string
        /** False for a read-only source - an examples directory. */
        writable?: boolean
      }
    >
  >({})
  let workflowDir = $state('')
  let filter = $state('')
  let error = $state('')
  let loaded = $state(false)

  $effect(() => {
    api
      .listWorkflows()
      .then((result) => {
        workflows = result.workflows
        details = result.details ?? {}
        workflowDir = result.workflow_dir
        loaded = true
      })
      .catch((e) => (error = e.message))
  })

  const visible = $derived(
    workflows.filter((name) => {
      const needle = filter.toLowerCase()
      return (
        name.toLowerCase().includes(needle) ||
        (details[name]?.description ?? '').toLowerCase().includes(needle)
      )
    }),
  )

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

<HintBar storageKey="hint-dismissed">
  Pick a workflow → tweak its variables → Run. Every image saves its recipe —
  reopen it from the Gallery.
</HintBar>

<FolderGroups
  names={visible}
  collapseKey="collapsed-folders"
  filterActive={filter !== ''}
  newHref="#/edit"
  onnewingroup={(group) => sessionStorage.setItem('dw-editor-folder', group)}
>
  {#snippet card(name)}
    {@const detail = details[name]}
    {@const group = name.includes('/') ? name.split('/')[0] : ''}
    <a
      class="card panel"
      href={href(name)}
      title={detail?.description || undefined}
    >
      <span class="cardtop">
        <span class="cardname"
          >{group ? name.split('/').slice(1).join('/') : name}</span
        >
        <span class="cardmeta muted">
          {#if detail?.writable === false}<span
              title="read-only: from the {detail.origin} directory"
              >{detail.origin}</span
            >{/if}
          {#if detail?.kinds.includes('image')}<Image size={13} />{/if}
          {#if detail?.kinds.includes('video')}<Film size={13} />{/if}
          {#if detail?.kinds.includes('audio')}<Music size={13} />{/if}
          {#if (detail?.steps ?? 0) > 1}<span
              title="{detail.steps} steps run in sequence"
              >{detail.steps} steps</span
            >{/if}
          {#if detail?.variables}<span
              title="{detail.variables} variables to tweak"
              >{detail.variables} vars</span
            >{/if}
        </span>
      </span>
      {#if detail?.description}
        <span class="carddesc muted">{detail.description}</span>
      {/if}
    </a>
  {/snippet}
</FolderGroups>

{#if loaded && workflows.length === 0}
  <Empty>
    {#snippet icon()}<Layers size={36} strokeWidth={1.5} />{/snippet}
    No workflows yet — the + above creates the first one.
  </Empty>
{:else if loaded && visible.length === 0}
  <p class="muted">Nothing matches "{filter}".</p>
{/if}

<style>
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 1rem;
    margin-bottom: 1rem;
  }
  .filter {
    max-width: 220px;
    margin-left: auto;
  }
  .newlink {
    display: inline-flex;
    align-items: center;
    padding: 0.4rem;
    border: 1px solid var(--line);
    border-radius: 6px;
    color: var(--muted);
  }
  .newlink:hover {
    border-color: var(--accent);
    color: var(--accent);
  }
  .card {
    color: var(--ink);
    font-weight: 600;
    padding: 0.7rem 0.9rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .card:hover {
    border-color: var(--accent);
  }
  .cardtop {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
  }
  .cardname {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .carddesc {
    font-weight: 400;
    font-size: 0.78rem;
    line-height: 1.35;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .cardmeta {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.72rem;
    font-weight: 500;
    flex-shrink: 0;
  }
</style>
