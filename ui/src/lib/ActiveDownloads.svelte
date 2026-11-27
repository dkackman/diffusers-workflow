<script lang="ts">
  import { X } from '@lucide/svelte'
  import { api } from './api'
  import { action } from './action.svelte'
  import { gbFromBytes as gb } from './format'
  import { resource } from './resource.svelte'
  import LoadState from './ui/LoadState.svelte'
  import type { ModelDownload } from './types'

  // Downloads follow the models feed (one started or cancelled, here or by
  // an agent); one in flight also keeps a 2 s poll while it runs
  const downloadsRead = resource(() => api.listDownloads(), {
    watch: ['models'],
    alsoLive: (): number | null => (active ? 2000 : null),
  })
  const downloads = $derived(downloadsRead.data?.downloads ?? [])
  const active = $derived(
    downloads.filter((d) => d.status === 'downloading').length,
  )

  const cancelDownload = action(
    async (id: string) => {
      await api.cancelDownload(id)
      await downloadsRead.reload()
    },
    { doing: 'cancel the download' },
  )

  const pct = (download: ModelDownload) =>
    download.total
      ? Math.min(100, Math.round((100 * download.downloaded) / download.total))
      : null
</script>

<LoadState of={downloadsRead} label="downloads" quiet>
  {#each downloads.filter((d) => d.status === 'downloading' || (d.finished_at && Date.now() / 1000 - d.finished_at < 300)) as download (download.id)}
    <div class="dl" class:failed={download.status === 'failed'}>
      <span class="dlrepo">{download.repo_id}</span>
      {#if download.status === 'downloading'}
        {#if pct(download) != null}
          <span class="dlbar">
            <span style="width: {pct(download)}%"></span>
          </span>
          <span class="muted dlnum">
            {gb(download.downloaded)} / {gb(download.total ?? 0)} GB
          </span>
        {:else}
          <span class="muted dlnum">{gb(download.downloaded)} GB…</span>
        {/if}
        <button
          class="quiet icon"
          onclick={() => cancelDownload.run(download.id)}
          disabled={cancelDownload.busy}
          title="cancel this download - partial files resume on retry"
          aria-label="cancel downloading {download.repo_id}"
        >
          <X size={14} />
        </button>
      {:else if download.status === 'failed'}
        <span class="bad">{download.error}</span>
      {:else}
        <span class="muted">{download.status}</span>
      {/if}
    </div>
  {/each}
</LoadState>

<style>
  .dl {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem 0.7rem;
    font-size: 0.88rem;
    margin-bottom: 0.4rem;
  }
  .dl.failed .bad {
    font-size: 0.82rem;
  }
  .dlrepo {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
    font-weight: 500;
  }
  .dlbar {
    width: 180px;
    height: 7px;
    border-radius: 4px;
    background: var(--line);
    overflow: hidden;
    display: inline-block;
  }
  /* A download in flight - the machine working, so the signal colour, the
     same as a running job and a filling progress bar */
  .dlbar > span {
    display: block;
    height: 100%;
    background: var(--live);
  }
  .dlnum {
    font-variant-numeric: tabular-nums;
  }
</style>
