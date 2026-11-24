<script lang="ts" module>
  // The server checks a download with huggingface_hub's validate_repo_id
  // (dw/hub_cache.py); this is that rule, so the field never refuses a repo
  // the server takes nor sends one it refuses. Pinned by
  // tests/test_ui_twins.py. A part is 1-96 word, "-" or "." characters,
  // opening and closing on a word character; the namespace is optional
  const PART = String.raw`[A-Za-z0-9_](?:[A-Za-z0-9_.-]{0,94}[A-Za-z0-9_])?`
  const REPO_ID = new RegExp(`^(?:${PART}/)?${PART}$`)

  const REPO_ID_HINT =
    'A repo id is org/name, like black-forest-labs/FLUX.1-dev.'

  /** Why a download field's value cannot be sent, or null when it is empty
   * or a repo id the server accepts */
  export function repoIdHint(value: string): string | null {
    const repo = value.trim()
    if (!repo) return null
    const valid =
      REPO_ID.test(repo) &&
      !repo.includes('--') &&
      !repo.includes('..') &&
      !repo.endsWith('.git')
    return valid ? null : REPO_ID_HINT
  }
</script>

<script lang="ts">
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import DiskMeter from '../ui/DiskMeter.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import { formatBytes, formatDay, gbFromBytes } from '../format'
  import {
    ChevronDown,
    ChevronRight,
    Download,
    ExternalLink,
    RefreshCw,
    Trash2,
    X,
  } from '@lucide/svelte'
  import { api } from '../api'
  import { action } from '../action.svelte'
  import { modelDeleteConfirm, UPDATE_DIFFUSERS_CONFIRM } from '../destructive'
  import CopyButton from '../CopyButton.svelte'
  import LastUsedCell, { LAST_USED_TIP } from '../LastUsedCell.svelte'
  import type { ModelDownload, ModelRepo } from '../types'

  let expanded = $state<Record<string, boolean>>({})
  let repoInput = $state('')
  const repoHint = $derived(repoIdHint(repoInput))

  // The hub cache and downloads follow the models feed (a delete, a
  // download started or cancelled, here or by an agent); a download or a
  // diffusers update in flight also keeps a 2 s poll while it runs
  const cacheRead = resource(() => api.listModels(), { watch: ['models'] })
  const cache = $derived(cacheRead.data ?? null)
  const downloadsRead = resource(() => api.listDownloads(), {
    watch: ['models'],
    alsoLive: (): number | null => (active ? 2000 : null),
  })
  const downloads = $derived(downloadsRead.data?.downloads ?? [])
  const active = $derived(
    downloads.filter((d) => d.status === 'downloading').length,
  )
  // A download finishing re-reads the repo list, so it appears with its
  // real size
  let wasActive = 0
  $effect(() => {
    const now = active
    if (now < wasActive) void cacheRead.reload()
    wasActive = now
  })
  // An update started or finished elsewhere (another tab, an agent) moves
  // models too
  const diffusersRead = resource(() => api.diffusersStatus(), {
    watch: ['models'],
    alsoLive: (): number | null => (updating ? 2000 : null),
  })
  const diffusers = $derived(diffusersRead.data)
  // Polled while pip runs so the badge flips when it finishes
  const updating = $derived(diffusers?.status === 'running')

  const update = action(
    async () => {
      await api.updateDiffusers()
      await diffusersRead.reload()
    },
    {
      doing: 'start the diffusers update',
      confirm: () => UPDATE_DIFFUSERS_CONFIRM,
    },
  )

  const startDownload = action(
    async (repo: string) => {
      await api.startDownload(repo)
      repoInput = ''
      await downloadsRead.reload()
    },
    { doing: (repo) => `download ${repo}` },
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

  const remove = action(
    async (repo: ModelRepo) => {
      await api.deleteModel(repo.repo_id)
      await cacheRead.reload()
    },
    {
      doing: (repo) => `delete ${repo.repo_id}`,
      confirm: (repo) =>
        modelDeleteConfirm(repo.repo_id, gbFromBytes(repo.size_on_disk)),
    },
  )

  const gb = gbFromBytes
  const day = (stamp: number | null) => (stamp ? formatDay(stamp) : '—')
  const hubUrl = (repo: ModelRepo) =>
    'https://huggingface.co/' +
    (repo.repo_type === 'model' ? '' : repo.repo_type + 's/') +
    repo.repo_id

  // The size bar scales against the largest repo so relative cost reads at
  // a glance
  const largest = $derived(
    Math.max(1, ...(cache?.repos ?? []).map((r) => r.size_on_disk)),
  )
</script>

<PageHeader title="Models" baseline>
  {#if cache}
    <span class="muted" title={cache.cache_dir}>
      {cache.repos.length} cached · {gb(cache.size_on_disk)} GB
    </span>
  {/if}
  {#snippet actions()}
    {#if cache && cache.disk_total && cache.disk_free != null}
      <DiskMeter
        freeBytes={cache.disk_free}
        totalBytes={cache.disk_total}
        title="disk holding the hub cache"
        icon
      />
    {/if}
  {/snippet}
</PageHeader>

<div class="toolrow">
  <form
    class="dlform"
    onsubmit={(e) => {
      e.preventDefault()
      const repo = repoInput.trim()
      if (repo && !repoIdHint(repo)) void startDownload.run(repo)
    }}
  >
    <input
      aria-label="Hugging Face repo to download"
      placeholder="org/name, e.g. black-forest-labs/FLUX.1-dev"
      bind:value={repoInput}
      aria-invalid={repoHint ? 'true' : undefined}
    />
    <button
      class="withicon"
      type="submit"
      disabled={!repoInput.trim() || !!repoHint || startDownload.busy}
      title="download this repo into the hub cache"
    >
      <Download size={14} />Download
    </button>
    {#if repoHint}<span class="hint">{repoHint}</span>{/if}
  </form>

  <LoadState of={diffusersRead} label="the diffusers version" quiet>
    {#if diffusers}
      <div class="engine">
        <span class="enginename">diffusers</span>
        <code>{diffusers.version ?? 'not installed'}</code>
        {#if diffusers.commit}
          <span
            class="muted"
            title="installed from git commit {diffusers.commit}"
          >
            @{diffusers.commit.slice(0, 9)}
          </span>
        {/if}
        {#if updating}
          <span class="muted busy">
            <RefreshCw size={14} class="spin" />updating from GitHub…
          </span>
        {:else}
          <button
            class="quiet withicon"
            onclick={() => update.run()}
            disabled={update.busy}
            title="install the latest diffusers from GitHub - new model pipelines usually land there before a release. The worker restarts when idle so the next job uses it"
          >
            <RefreshCw size={14} />Update from GitHub
          </button>
          {#if diffusers.status === 'failed'}
            <span class="bad" title={diffusers.log ?? ''}>
              update failed: {diffusers.error} (hover for pip output)
            </span>
          {:else if diffusers.status === 'succeeded'}
            <span class="updated">updated - the next job uses it</span>
          {/if}
        {/if}
      </div>
    {/if}
  </LoadState>
</div>

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

{#if cache?.warnings.length}
  <p class="muted skipped" title={cache.warnings.join('\n')}>
    {cache.warnings.length} cache
    {cache.warnings.length === 1 ? 'entry' : 'entries'} skipped - not model repos
    (hover for detail)
  </p>
{/if}

<LoadState
  of={cacheRead}
  label="the hub cache"
  empty={cache !== null && cache.repos.length === 0}
>
  {#snippet emptyText()}The hub cache is empty - models land here the first time
    a workflow downloads them.{/snippet}
  {#if cache}
    <div class="tablewrap">
      <table class="datatable">
        <thead>
          <tr>
            <th></th>
            <th>repo</th>
            <th class="num">size</th>
            <th class="num wideonly">files</th>
            <th class="wideonly" title={LAST_USED_TIP}>last used</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {#each cache.repos as repo (repo.repo_id)}
            <tr>
              <td>
                <button
                  class="quiet icon"
                  onclick={() =>
                    (expanded[repo.repo_id] = !expanded[repo.repo_id])}
                  title={expanded[repo.repo_id]
                    ? 'hide revisions'
                    : 'show cached revisions'}
                  aria-label="toggle revisions for {repo.repo_id}"
                >
                  {#if expanded[repo.repo_id]}<ChevronDown
                      size={14}
                    />{:else}<ChevronRight size={14} />{/if}
                </button>
              </td>
              <td class="repo">
                {repo.repo_id}
                {#if repo.repo_type !== 'model'}
                  <span class="badge">{repo.repo_type}</span>
                {/if}
                <CopyButton
                  text={repo.repo_id}
                  title="copy model id to clipboard"
                  class="copymodel"
                />
                <a
                  href={hubUrl(repo)}
                  target="_blank"
                  rel="noreferrer"
                  class="muted hublink"
                  title="open on huggingface.co"
                >
                  <ExternalLink size={12} />
                </a>
              </td>
              <td class="num size">
                <span
                  class="sizebar"
                  style="width: {(100 * repo.size_on_disk) / largest}%"
                ></span>
                <span class="sizenum">{formatBytes(repo.size_on_disk)}</span>
              </td>
              <td class="num muted wideonly">{repo.nb_files}</td>
              <LastUsedCell
                class="wideonly"
                lastJobUse={repo.last_job_use}
                lastAccessed={repo.last_accessed}
              />
              <td>
                <button
                  class="quiet icon danger"
                  onclick={() => remove.run(repo)}
                  disabled={remove.busy}
                  title="delete every cached revision of this repo"
                  aria-label="delete {repo.repo_id} from the cache"
                >
                  <Trash2 size={14} />
                </button>
              </td>
            </tr>
            {#if expanded[repo.repo_id]}
              {#each repo.revisions as revision (revision.commit_hash)}
                <tr class="revision">
                  <td></td>
                  <td class="muted">
                    <code>{revision.commit_hash.slice(0, 12)}</code>
                    {#if revision.refs.length}({revision.refs.join(', ')}){/if}
                  </td>
                  <td class="num muted">
                    {formatBytes(revision.size_on_disk)}
                  </td>
                  <td class="wideonly"></td>
                  <td class="muted lastused wideonly">
                    {day(revision.last_modified)}
                  </td>
                  <td></td>
                </tr>
              {/each}
            {/if}
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</LoadState>

<style>
  .engine {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-left: auto;
    font-size: 0.9rem;
  }
  .enginename {
    color: var(--muted);
  }
  .busy {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  .engine :global(.spin) {
    animation: spin 1.2s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .engine :global(.spin) {
      animation: none;
    }
  }
  .updated {
    color: var(--accent);
  }
  .bad {
    font-size: 0.9rem;
  }
  .toolrow {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.8rem;
  }
  .dlform {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
  }
  .dlform input {
    max-width: 320px;
    min-width: 0;
  }
  @media (max-width: 640px) {
    .toolrow {
      flex-direction: column;
      align-items: stretch;
    }
    .engine {
      margin-left: 0;
    }
    .dlform input {
      flex: 1;
      max-width: none;
    }
    /* repo and size are what a phone needs; the rest is on a wider screen */
    .datatable :global(.wideonly) {
      display: none;
    }
  }
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
  .skipped {
    font-size: 0.82rem;
    margin: 0 0 0.6rem;
  }
  /* Six columns of repo ids and sizes have a floor well above a phone: the
     table scrolls itself (.tablewrap) rather than the whole document */
  table {
    font-size: 0.9rem;
  }
  .num {
    text-align: right;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  /* A repo id is what you paste into a workflow for the engine to resolve */
  .repo {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
    font-weight: 500;
  }
  .lastused {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: var(--t-xs);
  }
  .badge {
    font-size: 0.7rem;
    color: var(--muted);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 0 0.3rem;
    margin-left: 0.3rem;
  }
  .hublink {
    margin-left: 0.3rem;
    vertical-align: middle;
  }
  :global(.copymodel) {
    margin-left: 0.3rem;
    vertical-align: middle;
  }
  .size {
    position: relative;
    min-width: 140px;
  }
  .sizebar {
    position: absolute;
    right: 0.5rem;
    top: 15%;
    height: 70%;
    background: var(--accent);
    opacity: 0.12;
    border-radius: 3px;
    pointer-events: none;
  }
  .sizenum {
    position: relative;
  }
  .revision td {
    border-top: none;
    padding-top: 0.1rem;
    padding-bottom: 0.1rem;
    font-size: 0.82rem;
  }
</style>
