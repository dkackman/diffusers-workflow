<script lang="ts">
  import { Trash2, X } from '@lucide/svelte'
  import CopyButton from '../CopyButton.svelte'
  import { api } from '../api'
  import NoteEditor from '../ui/NoteEditor.svelte'
  import { ASSET } from '../references'
  import type { AssetFile, GalleryMetadata } from '../types'
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import Skeleton from '../ui/Skeleton.svelte'
  import { formatBytes, formatMtime } from '../format'
  import { workflowName } from '../terms'
  import { wsHref } from '../routes'
  import { workspace } from '../workspace.svelte'

  // The popout for the one asset picked: its reference to copy, the file,
  // and a delete unless a read-only library brought it
  let {
    asset,
    busy,
    onremove,
    onclose,
    onnotesaved,
  }: {
    asset: AssetFile
    busy: boolean
    onremove: (asset: AssetFile) => void
    onclose: () => void
    /** The note the server stored, so the page can carry it on the asset */
    onnotesaved?: (note: string | null) => void
  } = $props()

  // Where a kept asset came from, when keep_output recorded it: the
  // workflow and run version of the job that wrote the file
  // Read through the reference alone, so a note saved (a new `asset`, the
  // same file) does not read it again
  const reference = $derived(asset.reference)
  const source = resource(() => api.galleryMetadata(reference), {
    watch: ['assets'],
  })
  const keptFrom = (r: GalleryMetadata) => {
    if (!r.job) return ''
    const text = (v: unknown) => (typeof v === 'string' ? v : null)
    const named = text(r.job.workflow_name) || text(r.job.workflow)
    const workflow = named ? workflowName({ workflow_name: named }) : r.job.id
    return `kept from ${workflow}${r.version ? ` v${r.version}` : ''}`
  }
</script>

<div class="detail panel">
  <div class="bar">
    <strong class="selname">{asset.reference}</strong>
    <CopyButton
      text={asset.reference}
      title="copy the reference a workflow argument carries"
    />
    <span class="flex"></span>
    <a
      href={asset.url}
      target="_blank"
      class="muted"
      title="open the file itself in a new tab"
    >
      open file
    </a>
    <span class="num muted">
      {asset.kind} · {formatBytes(asset.size)} · {formatMtime(asset.mtime)}
    </span>
    {#if asset.origin === 'examples'}
      <span class="muted" title="read-only: an examples library brought it">
        read-only
      </span>
    {:else}
      <button
        class="quiet icon danger"
        onclick={() => onremove(asset)}
        disabled={busy}
        title="delete this asset from the library"
        aria-label="delete this asset from the library"
      >
        <Trash2 size={14} />
      </button>
    {/if}
    <span class="flex"></span>
    <button
      class="quiet icon"
      onclick={onclose}
      title="close details"
      aria-label="close details"
    >
      <X size={14} />
    </button>
  </div>
  <LoadState of={source} label="where this asset came from">
    {#snippet skeleton()}<Skeleton
        lines={1}
        label="loading where this asset came from"
      />{/snippet}
    {#snippet children(r)}
      {#if keptFrom(r)}<p class="usedby muted">{keptFrom(r)}</p>{/if}
    {/snippet}
  </LoadState>
  {#if asset.used_by?.length}
    <p class="usedby muted">
      used by
      {#each asset.used_by as name, i (name)}{i ? ', ' : ''}<a
          href={wsHref(workspace.current, 'workflows', ...name.split('/'))}
        >
          {name}
        </a>{/each}
    </p>
  {/if}
  <NoteEditor
    note={asset.note}
    target={`${ASSET}${asset.name}`}
    onsaved={(note) => onnotesaved?.(note)}
  />
  <div class="body">
    {#if asset.kind === 'image'}
      <img src={asset.url} alt={asset.name} />
    {:else if asset.kind === 'video'}
      <!-- svelte-ignore a11y_media_has_caption -->
      <video src={asset.url} controls loop></video>
    {:else if asset.kind === 'audio'}
      <audio src={asset.url} controls></audio>
    {:else}
      <a href={asset.url} target="_blank" rel="noreferrer">{asset.name}</a>
    {/if}
  </div>
</div>

<style>
  .usedby {
    margin: 0 0 var(--space-2);
    font-size: var(--t-sm);
  }
  /* The gallery's popout: it rides the bottom of the viewport while the
     grid scrolls behind it. Sitting at the end of the document instead
     would put it off screen for any click above the fold */
  .detail {
    position: sticky;
    bottom: 1rem;
    z-index: 2;
    margin-top: var(--space-4);
    padding: 0.6rem 0.8rem;
    box-shadow: 0 6px 24px rgb(0 0 0 / 0.35);
  }
  .bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
  }
  .selname {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
    word-break: break-all;
  }
  .body {
    margin-top: 0.6rem;
  }
  .body img,
  .body video {
    max-width: 100%;
    max-height: 60vh;
    border-radius: var(--radius-frame);
    background: var(--sunk);
  }
  .body audio {
    width: 100%;
  }
</style>
