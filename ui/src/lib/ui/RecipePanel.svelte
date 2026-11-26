<script lang="ts">
  // "What made this": one read of a file's metadata, and of the job that
  // made it when it is still on record (it names the workflow by its file,
  // and a video embeds no recipe), shown open, every argument listed. A page adds its own lines (notes,
  // findings) through `children`, which is handed the same read and a way
  // to read it again (after a note is saved), so the page makes no
  // metadata call of its own. Links name the file's own
  // workspace, never the one the picker is on (#1600)
  import type { Snippet } from 'svelte'
  import { untrack } from 'svelte'
  import { FolderOpen } from '@lucide/svelte'
  import CopyButton from '../CopyButton.svelte'
  import { api } from '../api'
  import { ApiError } from '../apiErrors'
  import { outputUrl } from '../apiUrls'
  import { recipeRows } from '../recipe'
  import { ASSET, OUTPUT, reference, referenceName } from '../references'
  import { resource } from '../resource.svelte'
  import { goCarrying } from '../router.svelte'
  import { wsHref } from '../routes'
  import type { GalleryInput, GalleryMetadata } from '../types'
  import ErrorLine from './ErrorLine.svelte'
  import LoadState from './LoadState.svelte'
  import ReferenceChip from './ReferenceChip.svelte'

  let {
    name,
    source,
    workspace,
    children: aside,
    onread,
    stamp,
  }: {
    /** The file's name: an output's path, or an asset's name unprefixed */
    name: string
    source: 'output' | 'asset'
    /** The workspace the file belongs to */
    workspace: string
    /** The page's own lines, under the recipe, from the same read, and
     * the read again (after a note is saved beside it) */
    children?: Snippet<[GalleryMetadata, () => Promise<GalleryMetadata | null>]>
    /** Handed each read as it lands, for a control beside the panel whose
     * state the reply holds (the job viewer's star) */
    onread?: (m: GalleryMetadata, asked: number) => void
    /** Asked as each read starts, and handed to `onread` with its reply, so
     * a page can tell a reply older than a write of its own */
    stamp?: () => number
  } = $props()

  // Read through the name and workspace alone: a page handing in a new
  // file object for the same file (a note saved beside it) reads nothing.
  // The name is the read's subject: another file's recipe is never shown
  // while this one's loads
  const file = $derived(name)
  const ws = $derived(workspace)
  const own = () => ws
  // What `stamp` said when each reply was asked for, read as the request
  // starts (nothing can write between the call and this)
  const asked = new WeakMap<GalleryMetadata, number>()
  async function stamped(read: Promise<GalleryMetadata>) {
    const at = untrack(() => stamp?.()) ?? 0
    const m = await read
    asked.set(m, at)
    return m
  }
  // An output moves with the gallery, an asset (read by its reference)
  // with the library; the source is fixed for the panel's life
  const meta =
    untrack(() => source) === 'output'
      ? resource(() => stamped(api.galleryMetadata(file, ws)), {
          watch: ['gallery'],
          workspace: own,
          subject: () => file,
        })
      : resource(
          () => stamped(api.galleryMetadata(reference(ASSET, file), ws)),
          {
            watch: ['assets'],
            workspace: own,
            subject: () => file,
          },
        )

  $effect(() => {
    const m = meta.status === 'ready' ? meta.data : null
    if (m) untrack(() => onread?.(m, asked.get(m) ?? 0))
  })

  // Read the file again: the fresh read, or null when it failed or a
  // later read overtook it (still loading), so nothing from before is
  // handed on
  async function reread(): Promise<GalleryMetadata | null> {
    await meta.reload()
    return meta.status === 'ready' ? (meta.data ?? null) : null
  }

  // The job, read whenever the file names one that history still holds: its
  // file name is the workflow's name, and a clip's arguments come from it.
  // Embedded arguments still win for the rows; a finished job never changes
  const jobId = $derived.by(() => {
    const m = meta.data
    if (!m || !m.job || m.job.status === 'forgotten') return null
    return m.job.id
  })
  const job = resource(
    () => {
      const id = jobId
      return id ? api.getJob(id) : Promise.resolve(null)
    },
    { watch: [], workspace: own },
  )
  const jobRead = $derived(jobId && job.data?.id === jobId ? job.data : null)
  // The job's read found nothing: it left history since the file was listed
  const jobGone = $derived(
    job.status === 'error' &&
      job.error instanceof ApiError &&
      job.error.status === 404,
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
    // 'workflow_seed' is the run's seed when this image's own differs from it
    const seed = m.metadata?.workflow_seed ?? m.metadata?.seed
    if (typeof seed === 'number') definition.seed = seed
    goCarrying(['ws', workspace, 'edit'], {
      'dw-editor-import': JSON.stringify(definition),
    })
  }
</script>

<section class="recipe">
  <h3>What made this</h3>
  <LoadState of={meta} label="what made this file">
    {#snippet children(m)}
      {@const rows = recipeRows(m, jobRead, workspace, { jobGone })}
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
      <!-- An image's recipe is what it embeds: its job's read is not missed -->
      {#if jobId && job.status === 'error' && !m.metadata}
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
      {@render aside?.(m, reread)}
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
