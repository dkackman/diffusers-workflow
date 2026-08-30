<script lang="ts">
  import {
    ChevronDown,
    ChevronRight,
    ExternalLink,
    HardDrive,
    Trash2,
  } from '@lucide/svelte'
  import { api } from '../api'
  import type { ModelCache, ModelRepo } from '../types'

  let cache = $state<ModelCache | null>(null)
  let error = $state('')
  let deleting = $state<string | null>(null)
  let expanded = $state<Record<string, boolean>>({})

  async function refresh() {
    try {
      cache = await api.listModels()
      error = ''
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    }
  }
  $effect(() => {
    refresh()
  })

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
      error = e instanceof Error ? e.message : String(e)
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

{#if error}<p class="warn">{error}</p>{/if}
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
              onclick={() => (expanded[repo.repo_id] = !expanded[repo.repo_id])}
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
{/if}

<style>
  .head {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 1rem;
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
