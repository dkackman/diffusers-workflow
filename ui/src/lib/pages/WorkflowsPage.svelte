<script lang="ts">
  import { Film, Image, Layers, Music, Plus } from '@lucide/svelte'
  import { api } from '../api'
  import Empty from '../Empty.svelte'
  import FolderGroups from '../FolderGroups.svelte'
  import { leafOf } from '../grouping'
  import HintBar from '../HintBar.svelte'
  import WorkspacePicker from '../WorkspacePicker.svelte'
  import { workspace } from '../workspace.svelte'
  import {
    WORKFLOW_SHAPES,
    WORKFLOW_TRAITS,
    type WorkflowCost,
    type WorkflowShape,
    type WorkflowTrait,
  } from '../types'

  type Detail = {
    kinds: string[]
    steps?: number
    variables: number
    description: string
    /** For a model config: the template it configures. */
    configures?: string
    /** What the workflow makes - the server derives it. */
    shape?: WorkflowShape
    /** Sorted facts about how it is made or what it needs. */
    traits?: WorkflowTrait[]
    /** The description's first sentence, clipped by the server. */
    summary?: string
    /** Measured runs; null or absent means nobody has measured it. */
    cost?: WorkflowCost[] | null
    /** Which source the workflow was read from. */
    origin?: string
    /** False for a read-only source - an examples directory. */
    writable?: boolean
  }

  let workflows = $state<string[]>([])
  let details = $state<Record<string, Detail>>({})
  let workflowDir = $state('')
  let filter = $state('')
  let shape = $state('')
  let traits = $state<WorkflowTrait[]>([])
  let error = $state('')
  let loaded = $state(false)

  $effect(() => {
    // Read inside the effect so switching workspaces refetches the listing
    void workspace.current
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

  // Only the shapes and traits the listing actually has: the vocabulary is
  // fixed, but a workspace holding no video has no use for a `shot` option
  const shapesPresent = $derived(
    WORKFLOW_SHAPES.filter((value) =>
      workflows.some((name) => details[name]?.shape === value),
    ),
  )
  const traitsPresent = $derived(
    WORKFLOW_TRAITS.filter((value) =>
      workflows.some((name) => details[name]?.traits?.includes(value)),
    ),
  )

  const toggleTrait = (trait: WorkflowTrait) => {
    traits = traits.includes(trait)
      ? traits.filter((value) => value !== trait)
      : [...traits, trait]
  }

  // Client-side over the listing the page already holds - shape, then every
  // selected trait (AND, not OR: the chips narrow), then the text filter
  const visible = $derived(
    workflows.filter((name) => {
      const detail = details[name]
      if (shape && detail?.shape !== shape) return false
      const has = detail?.traits ?? []
      if (!traits.every((trait) => has.includes(trait))) return false
      const needle = filter.toLowerCase()
      return (
        name.toLowerCase().includes(needle) ||
        (detail?.description ?? '').toLowerCase().includes(needle)
      )
    }),
  )

  const filterActive = $derived(
    filter !== '' || shape !== '' || traits.length > 0,
  )

  /** Minutes for a person: under one is '<1 min', anything else keeps the
   * measured value ('~3 min', '~1.5 min') rather than rounding it away. */
  const formatMinutes = (minutes: number) =>
    minutes < 1 ? '<1 min' : `~${Number(minutes.toFixed(2))} min`

  /** The one measurement a card shows: the first the maintainer recorded.
   * '~3 min · 22 GB (RTX 4090)', with the accelerator dropped when the entry
   * did not name one. */
  const formatCost = (cost?: WorkflowCost[] | null) => {
    const measured = cost?.[0]
    if (!measured) return ''
    const where = measured.name ? ` (${measured.name})` : ''
    return `${formatMinutes(measured.minutes)} · ${Number(
      measured.vram_gb.toFixed(2),
    )} GB${where}`
  }

  const href = (name: string) =>
    '#/workflows/' + name.split('/').map(encodeURIComponent).join('/')
</script>

<div class="head">
  <h1>Workflows</h1>
  <WorkspacePicker />
  <span class="muted">{workflowDir}</span>
  <div class="filters">
    <select bind:value={shape} class="shape" aria-label="shape">
      <option value="">any shape</option>
      {#each shapesPresent as value (value)}
        <option {value}>{value}</option>
      {/each}
    </select>
    <input placeholder="filter…" bind:value={filter} class="filter" />
  </div>
  <a class="newlink" href="#/edit" title="new workflow"><Plus size={15} /></a>
</div>

{#if error}
  <p class="muted">Could not load workflows: {error}</p>
{/if}

{#if traitsPresent.length}
  <div class="chips">
    {#each traitsPresent as trait (trait)}
      <button
        class="chip"
        class:on={traits.includes(trait)}
        aria-pressed={traits.includes(trait)}
        onclick={() => toggleTrait(trait)}
        title="show only workflows that are {trait}"
      >
        {trait}
      </button>
    {/each}
  </div>
{/if}

<HintBar storageKey="hint-dismissed">
  Pick a workflow → tweak its variables → Run. Every image saves its recipe —
  reopen it from the Gallery.
</HintBar>

<FolderGroups
  names={visible}
  collapseKey="collapsed-folders"
  {filterActive}
  newHref="#/edit"
  onnewingroup={(group) => sessionStorage.setItem('dw-editor-folder', group)}
>
  {#snippet card(name)}
    {@const detail = details[name]}
    <a
      class="card panel"
      href={href(name)}
      title={detail?.description || undefined}
    >
      <span class="cardtop">
        <span class="cardname">{leafOf(name)}</span>
      </span>
      {#if detail?.shape || detail?.traits?.length}
        <span class="cardbadges muted">
          {#if detail?.shape}<span
              class="badge"
              title="what this workflow makes">{detail.shape}</span
            >{/if}
          {#each detail?.traits ?? [] as trait (trait)}<span
              class="badge trait"
              title="trait: {trait}">{trait}</span
            >{/each}
        </span>
      {/if}
      <span class="cardmeta muted">
        {#if detail?.configures}<span
            class="configures"
            title="a tuned configuration of {detail.configures}"
            >configures {leafOf(detail.configures)}</span
          >{/if}
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
        {#if detail?.cost?.length}<span
            class="cost"
            title="measured on {detail.cost[0].name ?? detail.cost[0].device}"
            >{formatCost(detail.cost)}</span
          >{/if}
      </span>
      {#if detail?.summary || detail?.description}
        <span class="carddesc muted"
          >{detail.summary || detail.description}</span
        >
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
  .filters {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-left: auto;
  }
  .filter {
    max-width: 220px;
  }
  .shape {
    font-size: 0.8rem;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-bottom: 0.8rem;
  }
  /* A chip narrows the listing the page already holds - no refetch, so the
     selected set reads as part of the filter row rather than a mode */
  .chip {
    background: none;
    border: 1px solid var(--line);
    border-radius: 999px;
    color: var(--muted);
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.1rem 0.5rem;
    cursor: pointer;
  }
  .chip:hover {
    color: var(--ink);
    filter: none;
  }
  .chip.on {
    border-color: var(--accent);
    color: var(--accent);
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
    width: 100%;
  }
  .cardname {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 0.95rem;
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
  /* Marks a model config as a tuned instance of a template rather than a
     pattern to copy - the one distinction the two-tree catalog turns on */
  .configures {
    border: 1px solid currentColor;
    border-radius: 3px;
    padding: 0 0.25rem;
    opacity: 0.75;
    white-space: nowrap;
  }

  /* shape and traits: what the catalog is chosen by, so they sit with the
     kind icons rather than in the description */
  .badge {
    border: 1px solid currentColor;
    border-radius: 3px;
    padding: 0 0.25rem;
    opacity: 0.75;
    white-space: nowrap;
  }
  .badge.trait {
    opacity: 0.6;
  }
  .cost {
    white-space: nowrap;
  }

  /* Shape and traits on a row of their own: sharing the meta line with the
     origin, icons and counts wrapped at a different point on every card */
  .cardbadges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    font-size: 0.7rem;
    font-weight: 500;
  }
  .cardmeta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.72rem;
    font-weight: 500;
  }
</style>
