<script lang="ts">
  import {
    ChevronDown,
    ChevronRight,
    Download,
    ExternalLink,
    HardDrive,
    RefreshCw,
    Trash2,
    X,
  } from '@lucide/svelte'
  import { api } from '../api'
  import { notify } from '../toast'
  import type {
    DiffusersStatus,
    ModelCache,
    ModelDownload,
    ModelRepo,
  } from '../types'

  let cache = $state<ModelCache | null>(null)
  let error = $state('')
  let deleting = $state<string | null>(null)
  let expanded = $state<Record<string, boolean>>({})

  let downloads = $state<ModelDownload[]>([])
  let repoInput = $state('')

  async function refresh() {
    try {
      cache = await api.listModels()
      error = ''
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    }
  }
  async function refreshDownloads() {
    try {
      downloads = (await api.listDownloads()).downloads
    } catch {
      /* the models fetch reports connectivity problems */
    }
  }

  let diffusers = $state<DiffusersStatus | null>(null)
  async function refreshDiffusers() {
    try {
      diffusers = await api.diffusersStatus()
    } catch {
      /* the models fetch reports connectivity problems */
    }
  }
  $effect(() => {
    refresh()
    refreshDownloads()
    refreshDiffusers()
  })

  // Poll while pip runs so the badge flips when it finishes
  const updating = $derived(diffusers?.status === 'running')
  $effect(() => {
    if (!updating) return
    const timer = setInterval(refreshDiffusers, 2000)
    return () => clearInterval(timer)
  })

  async function startDiffusersUpdate() {
    try {
      diffusers = await api.updateDiffusers()
      error = ''
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      notify.error(msg)
    }
  }

  // While anything is downloading, poll progress - and refresh the repo
  // list when a download finishes so it appears with its real size
  const anyActive = $derived(downloads.some((d) => d.status === 'downloading'))
  $effect(() => {
    if (!anyActive) return
    const timer = setInterval(async () => {
      const before = downloads.filter((d) => d.status === 'downloading').length
      await refreshDownloads()
      const after = downloads.filter((d) => d.status === 'downloading').length
      if (after < before) refresh()
    }, 2000)
    return () => clearInterval(timer)
  })

  async function startDownload() {
    const repo = repoInput.trim()
    if (!repo) return
    try {
      await api.startDownload(repo)
      repoInput = ''
      error = ''
      await refreshDownloads()
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      notify.error(msg)
    }
  }

  async function cancelDownload(id: string) {
    try {
      await api.cancelDownload(id)
      await refreshDownloads()
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      notify.error(msg)
    }
  }

  const pct = (download: ModelDownload) =>
    download.total
      ? Math.min(100, Math.round((100 * download.downloaded) / download.total))
      : null

  async function remove(repo: ModelRepo) {
    if (
      !window.confirm(
        `Delete ${repo.repo_id} (${gb(repo.size_on_disk)} GB) from the hub cache?\n` +
          'The next workflow that needs it will download it again.',
      )
    )
      return
    deleting = repo.repo_id
    try {
      await api.deleteModel(repo.repo_id)
      await refresh()
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      notify.error(msg)
    } finally {
      deleting = null
    }
  }

  const gb = (bytes: number) => (bytes / 1024 ** 3).toFixed(1)
  const size = (bytes: number) =>
    bytes >= 1024 ** 3
      ? gb(bytes) + ' GB'
      : Math.max(1, Math.round(bytes / 1024 ** 2)) + ' MB'
  const day = (stamp: number | null) =>
    stamp ? new Date(stamp * 1000).toLocaleDateString() : '—'
  const hubUrl = (repo: ModelRepo) =>
    'https://huggingface.co/' +
    (repo.repo_type === 'model' ? '' : repo.repo_type + 's/') +
    repo.repo_id

  // The size bar scales against the largest repo so relative cost reads at
  // a glance
  const largest = $derived(
    Math.max(1, ...(cache?.repos ?? []).map((r) => r.size_on_disk)),
  )
  const usedPct = $derived.by(() => {
    if (!cache?.disk_total || cache.disk_free == null) return null
    return Math.round(
      (100 * (cache.disk_total - cache.disk_free)) / cache.disk_total,
    )
  })
</script>

<div class="head">
  <h1>Models</h1>
  {#if cache}
    <span class="muted" title={cache.cache_dir}>
      {cache.repos.length} cached · {gb(cache.size_on_disk)} GB
    </span>
    {#if usedPct != null}
      <span class="disk" title="disk holding the hub cache">
        <HardDrive size={14} />
        <span class="diskbar"><span style="width: {usedPct}%"></span></span>
        {gb(cache.disk_free ?? 0)} GB free
      </span>
    {/if}
  {/if}
</div>

<div class="toolrow">
  <form
    class="dlform"
    onsubmit={(e) => {
      e.preventDefault()
      startDownload()
    }}
  >
    <input
      placeholder="download a model… (org/name)"
      bind:value={repoInput}
      title="a Hugging Face repo id, e.g. black-forest-labs/FLUX.1-dev"
    />
    <button
      class="withicon"
      type="submit"
      disabled={!repoInput.trim()}
      title="download this repo into the hub cache"
    >
      <Download size={14} />Download
    </button>
  </form>

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
          class="withicon"
          onclick={startDiffusersUpdate}
          title="install the latest diffusers from GitHub - new model pipelines usually land there before a release. The worker restarts when idle so the next job uses it"
        >
          <RefreshCw size={14} />Update from GitHub
        </button>
        {#if diffusers.status === 'failed'}
          <span class="warn" title={diffusers.log ?? ''}>
            update failed: {diffusers.error} (hover for pip output)
          </span>
        {:else if diffusers.status === 'succeeded'}
          <span class="updated">updated - the next job uses it</span>
        {/if}
      {/if}
    </div>
  {/if}
</div>

{#if error}<p class="warn">{error}</p>{/if}

{#each downloads.filter((d) => d.status === 'downloading' || (d.finished_at && Date.now() / 1000 - d.finished_at < 300)) as download (download.id)}
  <div class="dl" class:failed={download.status === 'failed'}>
    <span class="dlrepo">{download.repo_id}</span>
    {#if download.status === 'downloading'}
      {#if pct(download) != null}
        <span class="dlbar"><span style="width: {pct(download)}%"></span></span>
        <span class="muted dlnum">
          {gb(download.downloaded)} / {gb(download.total ?? 0)} GB
        </span>
      {:else}
        <span class="muted dlnum">{gb(download.downloaded)} GB…</span>
      {/if}
      <button
        class="quiet icon"
        onclick={() => cancelDownload(download.id)}
        title="cancel this download - partial files resume on retry"
        aria-label="cancel downloading {download.repo_id}"
      >
        <X size={14} />
      </button>
    {:else if download.status === 'failed'}
      <span class="warn">{download.error}</span>
    {:else}
      <span class="muted">{download.status}</span>
    {/if}
  </div>
{/each}
{#if cache?.warnings.length}
  <p class="muted skipped" title={cache.warnings.join('\n')}>
    {cache.warnings.length} cache
    {cache.warnings.length === 1 ? 'entry' : 'entries'} skipped - not model repos
    (hover for detail)
  </p>
{/if}

{#if cache && cache.repos.length === 0}
  <p class="muted">
    The hub cache is empty - models land here the first time a workflow
    downloads them.
  </p>
{/if}

{#if cache}
  <div class="tablewrap">
    <table>
      <thead>
        <tr>
          <th></th>
          <th>repo</th>
          <th class="num">size</th>
          <th class="num">files</th>
          <th>last used</th>
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
              <span class="sizenum">{size(repo.size_on_disk)}</span>
            </td>
            <td class="num muted">{repo.nb_files}</td>
            <td class="muted">{day(repo.last_accessed)}</td>
            <td>
              <button
                class="quiet icon danger"
                onclick={() => remove(repo)}
                disabled={deleting !== null}
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
                <td class="num muted">{size(revision.size_on_disk)}</td>
                <td></td>
                <td class="muted">{day(revision.last_modified)}</td>
                <td></td>
              </tr>
            {/each}
          {/if}
        {/each}
      </tbody>
    </table>
  </div>
{/if}

<style>
  .head {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.4rem 1rem;
    margin-bottom: 1rem;
  }
  /* Six columns of repo ids and sizes have a floor well above a phone -
     scroll the table itself rather than the whole document */
  .tablewrap {
    overflow-x: auto;
  }
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
  .disk {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    margin-left: auto;
    color: var(--muted);
    font-size: 0.85rem;
  }
  .diskbar {
    width: 90px;
    height: 6px;
    border-radius: 3px;
    background: var(--line);
    overflow: hidden;
    display: inline-block;
  }
  .diskbar > span {
    display: block;
    height: 100%;
    background: var(--accent);
    opacity: 0.7;
  }
  .warn {
    color: var(--danger, #c33);
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
  }
  .withicon {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  .dl {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem 0.7rem;
    font-size: 0.88rem;
    margin-bottom: 0.4rem;
  }
  .dl.failed .warn {
    font-size: 0.82rem;
  }
  .dlrepo {
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
  .dlbar > span {
    display: block;
    height: 100%;
    background: var(--accent);
  }
  .dlnum {
    font-variant-numeric: tabular-nums;
  }
  .skipped {
    font-size: 0.82rem;
    margin: 0 0 0.6rem;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }
  th {
    text-align: left;
    font-weight: 600;
    color: var(--muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 0.4rem 0.5rem;
    border-bottom: 1px solid var(--line);
  }
  td {
    padding: 0.35rem 0.5rem;
    border-bottom: 1px solid var(--line);
    vertical-align: middle;
  }
  .num {
    text-align: right;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .repo {
    font-weight: 500;
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
    border-bottom: none;
    padding-top: 0.1rem;
    padding-bottom: 0.1rem;
    font-size: 0.82rem;
  }
  .icon {
    display: inline-flex;
    padding: 0.25rem 0.4rem;
  }
</style>
