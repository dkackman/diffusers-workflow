<script lang="ts">
  import { Film, Image, Layers, Music, Plus } from '@lucide/svelte'
  import { api } from '../api'
  import Segmented from '../ui/Segmented.svelte'
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import { writableRoot } from '../libraries'
  import FolderGroups from '../FolderGroups.svelte'
  import { groupOf, leafOf } from '../grouping'
  import { compareNames } from '../names'
  import HintBar from '../HintBar.svelte'
  import { costFigure } from '../cost'
  import { observedText } from '../observed'
  import { latestProofs } from '../proofs'
  import { outputMedia } from '../media'
  import OutputTile from '../ui/OutputTile.svelte'
  import ListSort, { listSort, newestFirst } from '../ui/ListSort.svelte'
  import Time from '../ui/Time.svelte'
  import { sharedHref, wsHref } from '../routes'
  import { cardTitle } from '../workflowCard'
  import { catalogMatches, queryWords } from '../workflow/catalogSearch'
  import { workspace } from '../workspace.svelte'
  import { codec, hashState } from '../router.svelte'
  import {
    WORKFLOW_SHAPES,
    WORKFLOW_TRAITS,
    type GalleryFile,
    type WorkflowCard,
    type WorkflowCost,
    type WorkflowTrait,
  } from '../types'

  type Detail = WorkflowCard

  let { examples = false }: { examples?: boolean } = $props()

  // The filters live in the hash, so opening a workflow and pressing Back -
  // or sharing the link - keeps the narrowed view
  const filter = hashState('q', '')
  const shape = hashState(
    'shape',
    '',
    codec.oneOf<'' | (typeof WORKFLOW_SHAPES)[number]>([
      '',
      ...WORKFLOW_SHAPES,
    ]),
  )
  const traits = hashState<WorkflowTrait[]>(
    'traits',
    [],
    codec.someOf<WorkflowTrait>(WORKFLOW_TRAITS, []),
  )
  const sort = listSort()

  const list = resource(() => api.listWorkflows({ inputs: true }), {
    watch: ['workflows'],
    rows: (data) => data.workflows.map((name) => [name, data.details[name]]),
  })
  const workflows = $derived(list.data?.workflows ?? [])
  const details = $derived<Record<string, Detail>>(list.data?.details ?? {})
  /** What the listing says a card's `cost` is - "curated" today. */
  const costBasis = $derived(list.data?.cost_basis ?? '')
  const workflowDir = $derived(writableRoot(list.data?.libraries ?? []))
  // The newest output each workflow has produced, by workflow identity. A
  // run writes to <outputs>/<identity>/<run id>/, and the gallery listing
  // reports that identity as an entry's `folder` already stripped of the run
  // id - so a workflow's name matches its folder directly. Not rendered
  // through LoadState: the catalog reads fine without proofs, so a failed
  // read shows none
  const proofing = resource(
    async () => {
      const { files, workspace: home } = await api.gallery()
      return { workspace: home, proofs: latestProofs(files) }
    },
    { watch: ['gallery'] },
  )
  const proofs = $derived<Record<string, GalleryFile>>(
    proofing.data?.proofs ?? {},
  )
  // The proofs' own workspace, which their thumbnails name
  const proofHome = $derived(proofing.data?.workspace ?? workspace.current)

  // The population this page is over: the workspace's own workflows (any
  // writable library), or - under `examples` - only the read-only ones. The
  // templates stay one link away rather than mixed in among what the
  // workspace made. Everything below reads this rather than `workflows`
  // directly, so the count, the shape/trait vocabulary and the empty states
  // all agree with what the cards show.
  const readOnly = (name: string) => details[name]?.writable === false
  const listed = $derived(
    workflows.filter((name) => readOnly(name) === examples),
  )
  const exampleCount = $derived(workflows.filter(readOnly).length)

  // Only the shapes and traits the listing actually has: the vocabulary is
  // fixed, but a workspace holding no video has no use for a `shot` option
  const shapesPresent = $derived(
    WORKFLOW_SHAPES.filter((value) =>
      listed.some((name) => details[name]?.shape === value),
    ),
  )
  const traitsPresent = $derived(
    WORKFLOW_TRAITS.filter((value) =>
      listed.some((name) => details[name]?.traits?.includes(value)),
    ),
  )

  const toggleTrait = (trait: WorkflowTrait) => {
    traits.value = traits.value.includes(trait)
      ? traits.value.filter((value) => value !== trait)
      : [...traits.value, trait]
  }

  // Client-side over the listing the page already holds - shape, then every
  // selected trait (AND, not OR: the chips narrow), then the words: the best
  // matches, best first
  const visible = $derived(
    catalogMatches(
      listed.filter((name) => {
        const detail = details[name]
        if (shape.value && detail?.shape !== shape.value) return false
        const has = detail?.traits ?? []
        return traits.value.every((trait) => has.includes(trait))
      }),
      details,
      filter.value,
    ),
  )
  // Words put the best match first, ahead of what has run before
  const searching = $derived(queryWords(filter.value).length > 0)
  // Newest first is one list across the folders, as a search is
  const flat = $derived(searching || sort.value === 'newest')

  /** Within a folder, workflows that have produced something come first,
   * then alphabetical as before. Two reasons, and both matter: the ones you
   * have actually used are the ones you come back to, and only those carry
   * a picture - so grouping them keeps the taller cards together instead of
   * punching holes through an otherwise even grid. */
  // Newest orders a search's matches too; Name leaves them best first
  const ordered = $derived(
    sort.value === 'newest'
      ? newestFirst(visible, (name) => details[name]?.mtime)
      : searching
        ? visible
        : [...visible].sort((left, right) => {
            const leftRun = proofs[left] ? 0 : 1
            const rightRun = proofs[right] ? 0 : 1
            return leftRun - rightRun || compareNames(left, right)
          }),
  )

  const filterActive = $derived(
    filter.value !== '' || shape.value !== '' || traits.value.length > 0,
  )

  const clearFilters = () => {
    filter.value = ''
    shape.value = ''
    traits.value = []
  }

  /** The one measurement a card shows: the first the maintainer recorded.
   * 'about 3 min · 22 GB VRAM', with the accelerator moved to the tooltip - the card
   * has room for the numbers, not for the hardware they came from. */
  const formatCost = (cost?: WorkflowCost[] | null) => {
    const measured = cost?.[0]
    return measured ? costFigure(measured) : ''
  }

  /** The variables a workflow asks for, by name, or '' when it asks for none. */
  const needs = (detail?: Detail) =>
    Object.keys(detail?.inputs ?? {}).join(', ')

  const href = (name: string) =>
    examples
      ? sharedHref('examples', ...name.split('/'))
      : wsHref(workspace.current, 'workflows', ...name.split('/'))
</script>

{#snippet search()}
  <input
    placeholder="search workflows…"
    bind:value={filter.value}
    class="filter"
  />
{/snippet}
<PageHeader
  title={examples ? 'Examples' : 'Workflows'}
  count={{
    n: visible.length,
    of: listed.length,
    noun: examples ? 'example' : 'workflow',
  }}
  filter={search}
>
  {#snippet actions()}
    <ListSort {sort} />
    {#if !examples && exampleCount}
      <a
        class="sidelink plain muted"
        href={sharedHref('examples')}
        title="the read-only templates every workspace can start from"
      >
        examples <span class="num">{exampleCount}</span>
      </a>
    {/if}
    {#if !examples}
      <a
        class="quiet icon plain"
        href={wsHref(workspace.current, 'edit')}
        title="new workflow"
        aria-label="new workflow"
      >
        <Plus size={15} />
      </a>
    {/if}
  {/snippet}
</PageHeader>

<!-- Shape is what people choose by, so the vocabulary is visible rather
   than folded into a select nobody opens -->
{#if shapesPresent.length}
  <div class="shapes">
    <Segmented
      label="filter by shape"
      value={shape.value}
      options={[
        { value: '', label: 'all' },
        ...shapesPresent.map((value) => ({
          value,
          label: value,
          title:
            value === shape.value
              ? 'press again to show every shape'
              : `show only workflows that make ${value}`,
        })),
      ]}
      onselect={(value) => (shape.value = value === shape.value ? '' : value)}
    />
  </div>
{/if}

{#if traitsPresent.length}
  <div class="chips">
    {#each traitsPresent as trait (trait)}
      <button
        class="togglechip"
        class:on={traits.value.includes(trait)}
        aria-pressed={traits.value.includes(trait)}
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

<!-- The controls above stay put when they filter the grid to nothing -->
<LoadState
  of={list}
  label="workflows"
  empty={visible.length === 0}
  filtered={filterActive && listed.length > 0}
  onclear={clearFilters}
>
  {#snippet emptyIcon()}<Layers size={36} strokeWidth={1.5} />{/snippet}
  {#snippet emptyText()}
    {#if examples}
      <span>No examples on this server.</span><br /><span>
        Start dw.serve with --examples-dir &lt;folder&gt; to list some.
      </span>
    {:else}
      No workflows yet — the + above creates the first one{#if exampleCount}, or
        start from one of the <a href={sharedHref('examples')}>
          {exampleCount} examples
        </a>{/if}.
    {/if}
  {/snippet}
  <FolderGroups
    names={ordered}
    collapseKey="collapsed-folders"
    {filterActive}
    newHref={examples ? undefined : wsHref(workspace.current, 'edit')}
    minColumn="200px"
    groupOf={flat ? () => '' : undefined}
    onnewingroup={(group) => sessionStorage.setItem('dw-editor-folder', group)}
  >
    {#snippet card(name)}
      {@const detail = details[name]}
      {@const proof = proofs[name]}
      <a
        class="card"
        class:row-changed={list.changed.has(name)}
        href={href(name)}
        title={cardTitle(name, detail, costBasis)}
      >
        <!-- The proof: what this workflow actually made last time. In a tool
           whose entire output is pictures, the picture is the description.
           A workflow that has never run gets no frame at all rather than a
           grey placeholder - a fresh workspace would otherwise be a wall of
           empty plates, and the extra height is worth spending only where
           there is something to look at. Card height ends up saying which
           workflows you have actually used. -->
        {#if proof}
          <OutputTile
            file={outputMedia(proof, proofHome)}
            size="card"
            caption="none"
            alt=""
          />
        {/if}
        <span class="caption">
          <span class="cardname">{leafOf(name)}</span>
          <!-- Searching flattens the folders, so the folder moves onto the
             card: three text-to-image workflows must not look alike -->
          {#if flat && groupOf(name)}
            <span class="cardfolder muted">{groupOf(name)}</span>
          {/if}
          <span class="cardmeta muted">
            {#if detail?.shape}<span class="shape">{detail.shape}</span>{/if}
            {#if detail?.kinds.includes('image')}<Image size={12} />{/if}
            {#if detail?.kinds.includes('video')}<Film size={12} />{/if}
            {#if detail?.kinds.includes('audio')}<Music size={12} />{/if}
            <span class="flex"></span>
            {#if (detail?.steps ?? 0) > 1}
              <span class="num">{detail?.steps} steps</span>
            {/if}
          </span>
          {#if needs(detail)}
            <span class="needs muted">needs {needs(detail)}</span>
          {/if}
          <!-- A measured run is the most concrete thing on the card - what it
             will cost you to press Run - so it gets its own line instead of
             competing for the meta row's leftovers -->
          {#if detail?.cost?.length}
            <span class="cost num muted">{formatCost(detail.cost)}</span>
          {/if}
          {#if !detail?.cost?.length && observedText(detail?.observed)}
            <span class="cost num muted">{observedText(detail?.observed)}</span>
          {/if}
          {#if detail?.summary || detail?.description}
            <span class="carddesc muted">
              {detail.summary || detail.description}
            </span>
          {/if}
          {#if detail?.mtime}
            <span class="cardfoot muted">
              <Time at={detail.mtime} label="changed" />
            </span>
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
</LoadState>

<!-- Where the files live is worth knowing and not worth the subtitle slot
   beside the title, which is the best position on the page. Outside
   LoadState, so an empty, filtered or failed page still says it -->
{#if workflowDir}
  <p class="dir muted">
    read from <span class="path">{workflowDir}</span>
  </p>
{/if}

<style>
  /* Shape: the primary cut through the catalog, so it reads as a row of
     choices rather than a set of tags */
  .shapes {
    margin-bottom: var(--space-2);
    border-bottom: 1px solid var(--line);
    padding-bottom: var(--space-2);
  }

  /* Traits narrow further, so they are quieter than shape and read as a
     second-order refinement */
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-bottom: var(--space-4);
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
  .card :global(.filetile) {
    border-width: 0 0 1px;
    border-radius: 0;
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
  .needs {
    font-size: var(--t-xs);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
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
