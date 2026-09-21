<script lang="ts">
  import { Images, Inbox, Layers, Plus } from '@lucide/svelte'
  import { api } from '../api'
  import Empty from '../Empty.svelte'
  import { formatBytes } from '../format'
  import { latestProofs } from '../proofs'
  import { wsHref } from '../routes'
  import type { GalleryFile, JobSummary } from '../types'
  import {
    DEFAULT_WORKSPACE,
    loadWorkspaces,
    workspace,
  } from '../workspace.svelte'
  import { deleteWorkspaceWithConfirm } from '../workspaceActions'

  // Each panel loads on its own: the page is a glance at the workspace, and
  // a slow gallery must not hold the jobs list back
  let recent = $state<GalleryFile[] | null>(null)
  let jobs = $state<JobSummary[] | null>(null)
  let workflows = $state<string[] | null>(null)
  let proofs = $state<Record<string, GalleryFile>>({})
  let assetCount = $state<number | null>(null)
  // A failed fetch is not the same thing as a workspace that genuinely has
  // nothing - each panel gets its own error, shown in place of its content,
  // so a 500 reads as a 500 rather than as an empty workspace
  let recentError = $state<string | null>(null)
  let jobsError = $state<string | null>(null)
  let workflowsError = $state<string | null>(null)
  let assetsError = $state<string | null>(null)

  const name = $derived(workspace.current)
  const RECENT = 8
  const WORKFLOWS = 6
  const JOBS = 5

  const errorMessage = (e: unknown) =>
    e instanceof Error ? e.message : String(e)

  $effect(() => {
    void workspace.current
    recent = null
    jobs = null
    workflows = null
    assetCount = null
    recentError = null
    jobsError = null
    workflowsError = null
    assetsError = null
    loadWorkspaces()
    api
      .gallery()
      .then((r) => {
        recent = r.files.slice(0, RECENT)
        proofs = latestProofs(r.files)
      })
      .catch((e) => {
        recent = []
        recentError = errorMessage(e)
      })
    api
      .listJobs(workspace.current, JOBS + 1)
      .then((r) => (jobs = r.jobs.reverse()))
      .catch((e) => {
        jobs = []
        jobsError = errorMessage(e)
      })
    api
      .listWorkflows()
      .then((r) => (workflows = r.workflows))
      .catch((e) => {
        workflows = []
        workflowsError = errorMessage(e)
      })
    api
      .listAssets()
      .then(
        (r) =>
          (assetCount = r.assets.filter(
            (a) => a.origin === 'workspace',
          ).length),
      )
      .catch((e) => {
        assetCount = 0
        assetsError = errorMessage(e)
      })
  })

  // The ones that have produced something first - the same order the catalog
  // uses, so the overview and the catalog agree on what is "recent"
  const featured = $derived(
    [...(workflows ?? [])]
      .sort(
        (l, r) =>
          (proofs[l] ? 0 : 1) - (proofs[r] ? 0 : 1) || l.localeCompare(r),
      )
      .slice(0, WORKFLOWS),
  )
  const running = $derived(jobs?.find((j) => j.status === 'running') ?? null)
  const recentJobs = $derived(
    (jobs ?? []).filter((j) => j.status !== 'running').slice(0, JOBS),
  )
  const directory = $derived(
    workspace.root
      ? name === DEFAULT_WORKSPACE
        ? workspace.root
        : `${workspace.root}/${name}`
      : null,
  )
  const usage = $derived(workspace.usage[name])
  const leaf = (n: string) => n.split('/').pop() ?? n
</script>

<div class="head">
  <h1>{name}</h1>
  {#if directory}<code class="muted dir">{directory}</code>{/if}
</div>

<section class="panel">
  <h2>
    <a class="plain" href={wsHref(name, 'gallery')}
      ><Images size={15} /> Recent outputs</a
    >
  </h2>
  {#if recentError}
    <p class="muted">Could not load recent outputs: {recentError}</p>
  {:else if recent === null}
    <p class="muted">loading…</p>
  {:else if recent.length === 0}
    <Empty
      >{#snippet icon()}<Images size={36} strokeWidth={1.5} />{/snippet}Nothing
      generated yet — run a workflow.</Empty
    >
  {:else}
    <div class="strip">
      {#each recent as file (file.name)}
        <a
          class="plain frame tile"
          href={wsHref(name, 'gallery')}
          title={file.label}
        >
          {#if file.kind === 'image'}<img
              src={api.galleryThumbnailUrl(file.name)}
              alt={file.label}
              loading="lazy"
            />
          {:else if file.kind === 'video'}<video
              src={file.url}
              muted
              playsinline
              preload="metadata"
            ></video>
          {:else}<span class="muted">{file.label}</span>{/if}
        </a>
      {/each}
    </div>
  {/if}
</section>

<div class="two">
  <section class="panel">
    <h2><a class="plain" href={wsHref(name, 'jobs')}>Jobs</a></h2>
    {#if jobsError}
      <p class="muted">Could not load jobs: {jobsError}</p>
    {:else if jobs === null}
      <p class="muted">loading…</p>
    {:else if jobs.length === 0}
      <Empty
        >{#snippet icon()}<Inbox size={36} strokeWidth={1.5} />{/snippet}No jobs
        yet.</Empty
      >
    {:else}
      <ul class="jobs">
        {#if running}
          <li class="runningnow">
            <span class="chip running">running</span>
            <span class="wf">{running.workflow}</span>
            <a class="plain" href={wsHref(name, 'jobs', running.id)}
              >{running.id}</a
            >
          </li>
        {/if}
        {#each recentJobs as job (job.id)}
          <li>
            <span class="chip {job.status}">{job.status}</span>
            <span class="wf">{job.workflow}</span>
            <a class="plain" href={wsHref(name, 'jobs', job.id)}>{job.id}</a>
          </li>
        {/each}
      </ul>
    {/if}
  </section>

  <section class="panel">
    <h2>
      <a class="plain" href={wsHref(name, 'workflows')}
        ><Layers size={15} /> Workflows</a
      >
      {#if workflows}<span class="num muted">{workflows.length}</span>{/if}
      <span class="flex"></span>
      <a
        class="button withicon new"
        href={wsHref(name, 'edit')}
        title="new workflow"><Plus size={15} /> New workflow</a
      >
    </h2>
    {#if workflowsError}
      <p class="muted">Could not load workflows: {workflowsError}</p>
    {:else if workflows === null}
      <p class="muted">loading…</p>
    {:else if workflows.length === 0}
      <Empty
        >{#snippet icon()}<Layers size={36} strokeWidth={1.5} />{/snippet}No
        workflows yet — create one in the editor, or run an example.</Empty
      >
    {:else}
      <div class="cards">
        {#each featured as wf (wf)}
          <a
            class="plain card"
            href={wsHref(name, 'workflows', ...wf.split('/'))}
          >
            {#if proofs[wf]}
              <span class="cardframe">
                {#if proofs[wf].kind === 'image'}<img
                    src={api.galleryThumbnailUrl(proofs[wf].name)}
                    alt=""
                    loading="lazy"
                  />
                {:else}<video
                    src={proofs[wf].url}
                    muted
                    playsinline
                    preload="metadata"
                  ></video>{/if}
              </span>
            {/if}
            <span class="cardname">{leaf(wf)}</span>
          </a>
        {/each}
      </div>
    {/if}
  </section>
</div>

<div class="two">
  <section class="panel">
    <h2><a class="plain" href={wsHref(name, 'assets')}>Assets and disk</a></h2>
    {#if assetsError}
      <p class="muted">Could not load assets: {assetsError}</p>
    {:else}
      <dl>
        <dt>assets</dt>
        <dd>
          {#if assetCount === null}…{:else}{`${assetCount} ${assetCount === 1 ? 'asset' : 'assets'}`}{/if}
        </dd>
        {#if usage}
          <dt>on disk</dt>
          <dd class="num" title="{usage.files} files">
            {formatBytes(usage.bytes)}
          </dd>
        {/if}
      </dl>
    {/if}
  </section>
  <section class="panel">
    <h2>Manage</h2>
    <p class="muted">
      Deleting removes the workspace's workflows, assets and outputs on the
      server. The prompt library and the shared asset library are untouched.
    </p>
    <button
      class="quiet danger"
      onclick={() => deleteWorkspaceWithConfirm(name)}
      disabled={name === DEFAULT_WORKSPACE}
      title={name === DEFAULT_WORKSPACE
        ? 'the default workspace is the root itself and cannot be deleted'
        : 'delete this workspace and everything in it'}
    >
      Delete workspace
    </button>
  </section>
</div>

<style>
  .head {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    margin-bottom: var(--space-4);
  }
  .dir {
    font-size: var(--t-xs);
  }
  section {
    margin-bottom: var(--space-4);
  }
  h2 {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  h2 a {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  .new {
    font-family: var(--font-sans);
    font-size: var(--t-sm);
  }
  .two {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-4);
  }
  @media (max-width: 640px) {
    .two {
      grid-template-columns: 1fr;
    }
  }
  .strip {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: var(--space-2);
  }
  .tile {
    aspect-ratio: 1;
  }
  .jobs {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .jobs li {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    font-family: var(--font-mono);
    font-size: var(--t-sm);
  }
  .jobs .wf {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .jobs .runningnow {
    box-shadow: inset 2px 0 0 var(--live);
    padding-left: var(--space-2);
  }
  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: var(--space-2);
  }
  .card {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    font-family: var(--font-mono);
    font-size: var(--t-sm);
    border: 1px solid var(--line);
    border-radius: var(--radius-2);
    overflow: hidden;
    padding-bottom: var(--space-2);
  }
  .card:hover {
    border-color: var(--ink);
  }
  .cardname {
    padding: 0 var(--space-2);
  }
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
  dl {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: var(--space-1) var(--space-3);
    margin: 0;
  }
  dt {
    color: var(--muted);
  }
  dd {
    margin: 0;
    font-family: var(--font-mono);
  }
</style>
