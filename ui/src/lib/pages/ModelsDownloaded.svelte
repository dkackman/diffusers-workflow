<script lang="ts">
  import { repoIdHint } from '../repoId'
  import { resource } from '../resource.svelte'
  import ActiveDownloads from '../ActiveDownloads.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import { formatBytes, gbFromBytes } from '../format'
  import Time from '../ui/Time.svelte'
  import {
    ChevronDown,
    ChevronRight,
    Download,
    ExternalLink,
    Trash2,
  } from '@lucide/svelte'
  import { api } from '../api'
  import { action } from '../action.svelte'
  import { modelDeleteConfirm } from '../destructive'
  import CopyButton from '../CopyButton.svelte'
  import LastUsedCell, { LAST_USED_TIP } from '../LastUsedCell.svelte'
  import type { ModelRepo } from '../types'

  let expanded = $state<Record<string, boolean>>({})
  let repoInput = $state('')
  const repoHint = $derived(repoIdHint(repoInput))

  // The hub cache follows the models feed (a delete, here or by an agent, or
  // a download finishing); ModelsPage reads it too, for its header
  const cacheRead = resource(() => api.listModels(), { watch: ['models'] })
  const cache = $derived(cacheRead.data ?? null)
  const startDownload = action(
    async (repo: string) => {
      await api.startDownload(repo)
      repoInput = ''
    },
    { doing: (repo) => `download ${repo}` },
  )

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
</div>

<ActiveDownloads onfinished={() => cacheRead.refresh()} />

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
                  class="quiet icon"
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
                    {#if revision.last_modified}<Time
                        at={revision.last_modified}
                        mode="day"
                      />{:else}—{/if}
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
    .dlform input {
      flex: 1;
      max-width: none;
    }
    /* repo and size are what a phone needs; the rest is on a wider screen */
    .datatable :global(.wideonly) {
      display: none;
    }
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
  /* A repo and its revisions are one group: no rule between them */
  tbody > tr:has(+ tr.revision) > :global(td) {
    border-bottom: none;
  }
  .revision td {
    padding-top: 0.1rem;
    padding-bottom: 0.1rem;
    font-size: 0.82rem;
  }
</style>
