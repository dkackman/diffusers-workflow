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
    type GalleryFile,
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
  /** The newest output each workflow has produced, by workflow identity.
   * A run writes to <outputs>/<identity>/<run id>/, and the gallery listing
   * reports that identity as an entry's `folder` already stripped of the
   * run id - so a workflow's name matches its folder directly. */
  let proofs = $state<Record<string, GalleryFile>>({})

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

  $effect(() => {
    void workspace.current
    // Deliberately not awaited with the listing: the catalog is useful the
    // moment names arrive, and the proofs fill in behind it. A workspace
    // with no outputs yet simply never populates this.
    api
      .gallery()
      .then((result) => {
        // Entries arrive newest first, so the first one seen for a folder
        // is that workflow's latest. Images win over video because only
        // images have a thumbnail endpoint; a video-only workflow falls
        // back to its video, which renders its first frame.
        const latest: Record<string, GalleryFile> = {}
        for (const file of result.files) {
          if (!file.folder) continue
          const held = latest[file.folder]
          if (!held) latest[file.folder] = file
          else if (held.kind !== 'image' && file.kind === 'image')
            latest[file.folder] = file
        }
        proofs = latest
      })
      .catch(() => {
        /* the catalog reads fine without proofs */
      })
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

  /** Within a folder, workflows that have produced something come first,
   * then alphabetical as before. Two reasons, and both matter: the ones you
   * have actually used are the ones you come back to, and only those carry
   * a picture - so grouping them keeps the taller cards together instead of
   * punching holes through an otherwise even grid. */
  const ordered = $derived(
    [...visible].sort((left, right) => {
      const leftRun = proofs[left] ? 0 : 1
      const rightRun = proofs[right] ? 0 : 1
      return leftRun - rightRun || left.localeCompare(right)
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
   * '~3 min · 22 GB', with the accelerator moved to the tooltip - the card
   * has room for the numbers, not for the hardware they came from. */
  const formatCost = (cost?: WorkflowCost[] | null) => {
    const measured = cost?.[0]
    if (!measured) return ''
    return `${formatMinutes(measured.minutes)} · ${Number(
      measured.vram_gb.toFixed(1),
    )} GB VRAM`
  }

  /** Everything the card does not have room to print, so hovering it still
   * answers "what is this and what does it need". */
  const cardTitle = (name: string, detail?: Detail) => {
    const lines = [name]
    if (detail?.description) lines.push(detail.description)
    if (detail?.traits?.length) lines.push(detail.traits.join(', '))
    if (detail?.cost?.[0]) {
      const measured = detail.cost[0]
      const where = measured.name ?? measured.device
      lines.push(`measured on ${where}`)
    }
    if (detail?.writable === false)
      lines.push(`read-only, from the ${detail.origin} directory`)
    return lines.join('\n')
  }

  const href = (name: string) =>
    '#/workflows/' + name.split('/').map(encodeURIComponent).join('/')
</script>

<div class="head">
  <h1>Workflows</h1>
  <span class="count num muted">{workflows.length}</span>
  <WorkspacePicker />
  <span class="flex"></span>
  <input placeholder="filter…" bind:value={filter} class="filter" />
  <a class="newlink plain" href="#/edit" title="new workflow"
    ><Plus size={15} /></a
  >
</div>

{#if error}
  <p class="muted">Could not load workflows: {error}</p>
{/if}

<!-- Shape is what people choose by, so the vocabulary is visible rather
     than folded into a select nobody opens -->
{#if shapesPresent.length}
  <div class="shapes" role="group" aria-label="filter by shape">
    <button
      class="shapebtn"
      class:on={shape === ''}
      aria-pressed={shape === ''}
      onclick={() => (shape = '')}>all</button
    >
    {#each shapesPresent as value (value)}
      <button
        class="shapebtn"
        class:on={shape === value}
        aria-pressed={shape === value}
        onclick={() => (shape = shape === value ? '' : value)}
        title="show only workflows that make {value}">{value}</button
      >
    {/each}
  </div>
{/if}

{#if traitsPresent.length}
  <div class="chips">
    {#each traitsPresent as trait (trait)}
      <button
        class="traitchip"
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
  names={ordered}
  collapseKey="collapsed-folders"
  {filterActive}
  newHref="#/edit"
  minColumn="200px"
  onnewingroup={(group) => sessionStorage.setItem('dw-editor-folder', group)}
>
  {#snippet card(name)}
    {@const detail = details[name]}
    {@const proof = proofs[name]}
    <a class="card" href={href(name)} title={cardTitle(name, detail)}>
      <!-- The proof: what this workflow actually made last time. In a tool
           whose entire output is pictures, the picture is the description.
           A workflow that has never run gets no frame at all rather than a
           grey placeholder - a fresh workspace would otherwise be a wall of
           empty plates, and the extra height is worth spending only where
           there is something to look at. Card height ends up saying which
           workflows you have actually used. -->
      {#if proof}
        <span class="cardframe">
          {#if proof.kind === 'image'}
            <img
              src={api.galleryThumbnailUrl(proof.name)}
              alt=""
              loading="lazy"
              decoding="async"
            />
          {:else}
            <video src={proof.url} muted playsinline preload="metadata"></video>
          {/if}
        </span>
      {/if}
      <span class="caption">
        <span class="cardname">{leafOf(name)}</span>
        <span class="cardmeta muted">
          {#if detail?.shape}<span class="shape">{detail.shape}</span>{/if}
          {#if detail?.kinds.includes('image')}<Image size={12} />{/if}
          {#if detail?.kinds.includes('video')}<Film size={12} />{/if}
          {#if detail?.kinds.includes('audio')}<Music size={12} />{/if}
          <span class="flex"></span>
          {#if (detail?.steps ?? 0) > 1}
            <span class="num">{detail?.steps} steps</span>
          {:else if detail?.variables}
            <span class="num">{detail.variables} vars</span>
          {/if}
        </span>
        <!-- A measured run is the most concrete thing on the card - what it
             will cost you to press Run - so it gets its own line instead of
             competing for the meta row's leftovers -->
        {#if detail?.cost?.length}
          <span class="cost num muted">{formatCost(detail.cost)}</span>
        {/if}
        {#if detail?.summary || detail?.description}
          <span class="carddesc muted"
            >{detail.summary || detail.description}</span
          >
        {/if}
        {#if detail?.configures || detail?.writable === false}
          <span class="cardfoot muted">
            {#if detail?.configures}configures {leafOf(
                detail.configures,
              )}{/if}{#if detail?.configures && detail?.writable === false}
              ·
            {/if}{#if detail?.writable === false}{detail.origin}, read-only{/if}
          </span>
        {/if}
      </span>
    </a>
  {/snippet}
</FolderGroups>

{#if loaded && workflows.length === 0}
  <Empty>
    {#snippet icon()}<Layers size={36} strokeWidth={1.5} />{/snippet}
    No workflows yet — the + above creates the first one.
  </Empty>
{:else if loaded && visible.length === 0}
  <p class="muted">Nothing matches those filters.</p>
{/if}

<!-- Where the files live is worth knowing and not worth the subtitle slot
     beside the title, which is the best position on the page -->
{#if workflowDir}
  <p class="dir muted">
    read from <span class="path">{workflowDir}</span>
  </p>
{/if}

<style>
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.8rem;
    margin-bottom: var(--space-4);
  }
  .head .flex {
    flex: 1;
  }
  .count {
    font-size: var(--t-sm);
  }
  .filter {
    max-width: 220px;
  }
  .newlink {
    display: inline-flex;
    align-items: center;
    padding: 0.4rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-1);
    color: var(--muted);
  }
  .newlink:hover {
    border-color: var(--ink);
    color: var(--ink);
  }

  /* Shape: the primary cut through the catalog, so it reads as a row of
     choices rather than a set of tags */
  .shapes {
    display: flex;
    flex-wrap: wrap;
    gap: 0.2rem;
    margin-bottom: var(--space-2);
    border-bottom: 1px solid var(--line);
    padding-bottom: var(--space-2);
  }
  .shapebtn {
    background: none;
    border: 1px solid transparent;
    border-radius: var(--radius-1);
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: var(--t-sm);
    font-weight: 500;
    padding: 0.2rem 0.55rem;
    cursor: pointer;
  }
  .shapebtn:hover {
    color: var(--ink);
    background: var(--panel-2);
    filter: none;
  }
  .shapebtn.on {
    background: var(--ink);
    color: var(--panel);
    border-color: var(--ink);
    font-weight: 600;
  }

  /* Traits narrow further, so they are quieter than shape and read as a
     second-order refinement */
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-bottom: var(--space-4);
  }
  .traitchip {
    background: none;
    border: 1px solid var(--line);
    border-radius: 999px;
    color: var(--muted);
    font-size: var(--t-xs);
    font-weight: 500;
    padding: 0.05rem 0.5rem;
    cursor: pointer;
  }
  .traitchip:hover {
    color: var(--ink);
    border-color: var(--muted);
    filter: none;
  }
  .traitchip.on {
    background: var(--ink);
    border-color: var(--ink);
    color: var(--panel);
  }

  /* A frame on a contact sheet: the picture bleeds to the top edge, the
     caption is a separate strip below it */
  .card {
    display: flex;
    flex-direction: column;
    color: var(--ink);
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-2);
    overflow: hidden;
  }
  .card:hover {
    border-color: var(--ink);
  }
  /* A banner across the top of a card rather than a free-standing frame:
     the card's own border surrounds it, so it contributes only the rule
     that separates the picture from its caption */
  .cardframe {
    display: block;
    aspect-ratio: 4 / 3;
    background: var(--panel-2);
    border-bottom: 1px solid var(--line);
    overflow: hidden;
  }
  .cardframe img,
  .cardframe video {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
  .caption {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    padding: 0.55rem 0.7rem 0.65rem;
  }
  .cardname {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
    font-weight: 600;
    letter-spacing: -0.01em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .cardmeta {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-size: var(--t-xs);
  }
  .cardmeta .flex {
    flex: 1;
    min-width: 0.4rem;
  }
  .shape {
    font-family: var(--font-mono);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .cost {
    font-size: var(--t-xs);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .carddesc {
    font-size: var(--t-xs);
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .cardfoot {
    font-size: var(--t-xs);
    opacity: 0.8;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dir {
    margin-top: 2.5rem;
    font-size: var(--t-xs);
  }
  .path {
    font-family: var(--font-mono);
    word-break: break-all;
  }
</style>
