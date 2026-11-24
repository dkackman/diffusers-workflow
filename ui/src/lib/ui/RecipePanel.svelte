<script lang="ts">
  // "What made this": one read of a file's metadata, and of the job that
  // made it when the file embeds nothing (video embeds no recipe), shown
  // open, every argument listed. A page adds its own lines (notes,
  // findings) through `children`, which is handed the same read, so the
  // page makes no second metadata call. Links name the file's own
  // workspace, never the one the picker is on (#1600)
  import type { Snippet } from 'svelte'
  import { untrack } from 'svelte'
  import { FolderOpen } from '@lucide/svelte'
  import CopyButton from '../CopyButton.svelte'
  import { api } from '../api'
  import { outputUrl } from '../apiUrls'
  import { recipeRows } from '../recipe'
  import { ASSET, OUTPUT, reference, referenceName } from '../references'
  import { resource } from '../resource.svelte'
  import { go } from '../router.svelte'
  import { wsHref } from '../routes'
  import type { GalleryInput, GalleryMetadata, JobDetail } from '../types'
  import ErrorLine from './ErrorLine.svelte'
  import LoadState from './LoadState.svelte'
  import ReferenceChip from './ReferenceChip.svelte'

  let {
    name,
    source,
    workspace,
    children: aside,
  }: {
    /** The file's name: an output's path, or an asset's name unprefixed */
    name: string
    source: 'output' | 'asset'
    /** The workspace the file belongs to */
    workspace: string
    /** The page's own lines, under the recipe, from the same read */
    children?: Snippet<[GalleryMetadata]>
  } = $props()

  const own = () => workspace
  // An output moves with the gallery, an asset (read by its reference)
  // with the library; the source is fixed for the panel's life
  const meta =
    untrack(() => source) === 'output'
      ? resource(() => api.galleryMetadata(name, workspace), {
          watch: ['gallery'],
          workspace: own,
        })
      : resource(() => api.galleryMetadata(reference(ASSET, name), workspace), {
          watch: ['assets'],
          workspace: own,
        })

  // The job, read only when the file embeds nothing and names one that
  // history still holds; a finished job's arguments never change
  const jobId = $derived.by(() => {
    const m = meta.data
    if (!m || m.metadata || !m.job || m.job.status === 'forgotten') return null
    return m.job.id
  })
  const job = resource(
    () => {
      const id = jobId
      return id ? api.getJob(id) : Promise.resolve(null)
    },
    { watch: [], workspace: own },
  )
  const jobRead = $derived(
    jobId && job.data?.id === jobId ? (job.data as JobDetail) : null,
  )

  const inputName = (input: GalleryInput) =>
    referenceName(input.reference, input.kind === 'asset' ? ASSET : OUTPUT) ??
    input.reference
  // Where an input leads: an earlier output opens its file in a new tab,
  // an asset the library
  const inputLink = (input: GalleryInput) =>
    input.kind === 'asset'
      ? {
          href: wsHref(workspace, 'assets'),
          target: undefined,
          title: 'open the asset library',
          text: 'Open Assets',
        }
      : {
          href: outputUrl(inputName(input), undefined, workspace),
          target: '_blank',
          title: 'open this earlier output',
          text: 'Open',
        }
  // The folder the file sits in, as the engine wrote it
  const location = $derived(
    name.includes('/') ? name.slice(0, name.lastIndexOf('/')) : '',
  )

  const embeddedWorkflow = (m: GalleryMetadata) =>
    (m.metadata?.workflow as Record<string, unknown> | undefined) ?? null

  function openAsWorkflow(m: GalleryMetadata) {
    const embedded = embeddedWorkflow(m)
    if (!embedded) return
    // Pin the run's seed into the definition so reopening reproduces this
    // exact image; delete the seed field in the editor to re-randomize
    const definition = { ...embedded }
    if (typeof m.metadata?.seed === 'number') definition.seed = m.metadata.seed
    sessionStorage.setItem('dw-editor-import', JSON.stringify(definition))
    go('ws', workspace, 'edit')
  }
</script>

<section class="recipe">
  <h3>What made this</h3>
  <LoadState of={meta} label="what made this file">
    {#snippet children(m)}
      {@const rows = recipeRows(m, jobRead, workspace)}
      {@const head = rows.filter((r) => r.key !== 'argument')}
      {@const args = rows.filter((r) => r.key === 'argument')}
      {#if head.length}
        <dl class="formgrid top">
          {#each head as row (row.key)}
            <dt>{row.label}</dt>
            <dd class:long={row.key === 'prompt' || row.key === 'negative'}>
              {#if row.href}
                <a href={row.href} title="open the job that made this file">
                  {row.value}
                </a>
              {:else}
                <span class="value">{row.value}</span>
              {/if}
              {#if row.copy}
                <CopyButton
                  text={row.value}
                  title={`Copy the ${row.label.toLowerCase()}`}
                />
              {/if}
            </dd>
          {/each}
        </dl>
      {/if}
      {#if jobId && job.status === 'error'}
        <ErrorLine error={job.error} doing="load the job that made this file" />
      {/if}
      {#if embeddedWorkflow(m)}
        <button
          type="button"
          class="quiet withicon"
          onclick={() => openAsWorkflow(m)}
          title="open the embedded workflow definition in the editor"
        >
          <FolderOpen size={14} />Open as workflow
        </button>
      {/if}
      {#if args.length}
        <h4>Arguments</h4>
        <dl class="formgrid top">
          {#each args as row (row.label)}
            <dt>{row.label}</dt>
            <dd><code class="value">{row.value}</code></dd>
          {/each}
        </dl>
      {/if}
      {#if m.inputs?.length || location}
        <dl class="formgrid top">
          {#if m.inputs?.length}
            <dt>Made from</dt>
            <dd>
              <ul class="inputs">
                {#each m.inputs as input (input.kind + input.reference)}
                  {@const link = inputLink(input)}
                  <li>
                    <ReferenceChip
                      prefix={input.kind === 'asset' ? ASSET : OUTPUT}
                      name={inputName(input)}
                    />
                    <a href={link.href} target={link.target} title={link.title}>
                      {link.text}
                    </a>
                  </li>
                {/each}
              </ul>
            </dd>
          {/if}
          {#if location}
            <dt>Location</dt>
            <dd><code class="value">{location}/</code></dd>
          {/if}
        </dl>
      {/if}
      {#if !m.metadata && !m.job}
        <p class="none">No recipe is saved with this file.</p>
        <p class="hint">
          A step saves one when its result sets embed_metadata.
        </p>
      {/if}
      {@render aside?.(m)}
    {/snippet}
  </LoadState>
</section>

<style>
  .recipe {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    font-size: var(--t-sm);
    min-width: 0;
  }
  h3,
  h4 {
    margin: 0;
  }
  h4 {
    color: var(--muted);
    font-size: var(--t-sm);
  }
  dl {
    margin: 0;
  }
  dt {
    font-weight: 600;
    color: var(--muted);
    overflow-wrap: anywhere;
  }
  dd {
    margin: 0;
    display: flex;
    align-items: flex-start;
    gap: var(--space-1);
    min-width: 0;
  }
  .value {
    overflow-wrap: anywhere;
  }
  dd.long .value {
    white-space: pre-wrap;
  }
  .inputs {
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .inputs li {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  .none,
  .hint {
    margin: 0;
  }
</style>
