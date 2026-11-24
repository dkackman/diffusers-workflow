<script lang="ts">
  import {
    ClipboardList,
    Eye,
    EyeOff,
    Images,
    Inbox,
    Layers,
    Plus,
    Trash2,
  } from '@lucide/svelte'
  import StatusChip from '../ui/StatusChip.svelte'
  import OutputTile from '../ui/OutputTile.svelte'
  import { outputMedia } from '../media'
  import { api } from '../api'
  import { attention, doneFigure } from '../request/attention'
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import NoteEditor from '../ui/NoteEditor.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import Skeleton from '../ui/Skeleton.svelte'
  import { WORKSPACE } from '../references'
  import { formatBytes } from '../format'
  import { workflowName } from '../terms'
  import { hiddenWorkspaces, setHidden } from '../hiddenWorkspaces.svelte'
  import { latestProofs, proofByRun } from '../proofs'
  import { fileHref, jobHref, wsHref } from '../routes'
  import type { JobSummary } from '../types'
  import {
    DEFAULT_WORKSPACE,
    invalidateWorkspaces,
    loadWorkspaces,
    relistWorkspaces,
    workspace,
  } from '../workspace.svelte'
  import { deleteWorkspaceWithConfirm } from '../workspaceActions'
  import { action } from '../action.svelte'
  import ActionButton from '../ui/ActionButton.svelte'

  const name = $derived(workspace.current)
  // It asks first itself: the question needs the server's count
  const deleting = action(deleteWorkspaceWithConfirm, {
    doing: (ws: string) => `delete workspace ${ws}`,
  })
  const RECENT = 8
  const WORKFLOWS = 6
  const JOBS = 5

  // Each panel is its own resource and its own LoadState: the page is a
  // glance at the workspace, a slow gallery must not hold the jobs list
  // back, and one failing panel says so in place without blanking the rest.
  // Each reloads quietly when its own kind moves
  const gallery = resource(
    async () => {
      const { files, workspace: listed } = await api.gallery()
      return {
        workspace: listed,
        recent: files.slice(0, RECENT),
        proofs: latestProofs(files),
        runs: proofByRun(files),
      }
    },
    { watch: ['gallery'] },
  )
  // The running job and the last finished ones are two requests, not one
  // cut of the queue: queued jobs are newer than the running one, so a deep
  // queue would push it out of a single newest-N answer and fill the panel
  // with jobs that have done nothing yet
  const jobs = resource(
    async () => {
      const ws = workspace.current
      const [live, finished] = await Promise.all([
        api.listJobs(ws, 1, 'running'),
        api.listJobs(ws, JOBS, 'succeeded,failed,cancelled'),
      ])
      return {
        running: live.jobs[0] ?? null,
        finished: finished.jobs.reverse(),
      }
    },
    { watch: ['jobs'] },
  )
  // The workspace's own, as the Workflows page lists them - the read-only
  // examples have their own page
  const workflows = resource(
    async () => {
      const r = await api.listWorkflows()
      return r.workflows.filter((wf) => r.details?.[wf]?.writable !== false)
    },
    { watch: ['workflows'] },
  )
  const assets = resource(
    async () =>
      (await api.listAssets()).assets.filter((a) => a.origin === 'workspace')
        .length,
    { watch: ['assets'] },
  )
  // Open requests only, and nothing at all when there are none or the call
  // fails: that panel is a pointer, not a place for an error
  const requests = resource(() => api.listRequests(workspace.current), {
    watch: ['requests'],
  })
  const openRequests = $derived(
    (requests.data?.requests ?? []).filter((q) => q.state === 'open'),
  )

  // The note box is fed by `workspace.notes` alone, which only the
  // workspaces watch touches, and NoteEditor keeps an open draft when a
  // newer note arrives. The listing is module state (workspace.svelte.ts):
  // this is how it stays fresh, not where it is held
  resource(() => relistWorkspaces(), { watch: ['workspaces'] })

  // The ones that have produced something first - the same order the catalog
  // uses, so the overview and the catalog agree on what is "recent"
  const proofs = $derived(gallery.data?.proofs ?? {})
  const runs = $derived(gallery.data?.runs ?? {})
  // The workspace the listing came from: every tile's thumbnail and link
  // names it, so a switch never asks the new workspace for the old files
  const listed = $derived(gallery.data?.workspace ?? name)
  const featured = $derived(
    [...(workflows.data ?? [])]
      .sort(
        (l, r) =>
          (proofs[l] ? 0 : 1) - (proofs[r] ? 0 : 1) || l.localeCompare(r),
      )
      .slice(0, WORKFLOWS),
  )
  const directory = $derived(
    workspace.root
      ? name === DEFAULT_WORKSPACE
        ? workspace.root
        : `${workspace.root}/${name}`
      : null,
  )
  const usage = $derived(workspace.usage[name])
  function reloadWorkspaces() {
    invalidateWorkspaces()
    void loadWorkspaces()
  }
  const leaf = (n: string) => n.split('/').pop() ?? n
</script>

<PageHeader title={name} secondary={directory ?? undefined} baseline />
<NoteEditor
  note={workspace.notes[name]}
  target={`${WORKSPACE}${name}`}
  label="Your note on this workspace"
  onsaved={reloadWorkspaces}
/>

<section class="panel">
  <h2>
    <a class="plain" href={wsHref(name, 'gallery')}>
      <Images size={15} /> Recent outputs
    </a>
  </h2>
  <LoadState
    of={gallery}
    label="recent outputs"
    empty={gallery.data?.recent.length === 0}
  >
    {#snippet skeleton()}<Skeleton
        tiles={RECENT}
        label="loading recent outputs"
      />{/snippet}
    {#snippet emptyIcon()}<Images size={36} strokeWidth={1.5} />{/snippet}
    {#snippet emptyText()}Nothing generated yet — run a workflow.{/snippet}
    {#snippet children({ recent })}
      <div class="strip">
        {#each recent as file (file.name)}
          <OutputTile
            file={outputMedia(file, listed)}
            size="strip"
            caption="none"
            href={fileHref(listed, 'gallery', file.name)}
            alt={file.label}
            title={file.label}
            aria-label={file.label}
          />
        {/each}
      </div>
    {/snippet}
  </LoadState>
</section>

<!-- A job as one link: its status, its workflow and its run's proof -->
{#snippet jobRow(job: JobSummary, live: boolean)}
  {@const proof = job.run_id ? runs[job.run_id] : undefined}
  <li class:row-live={live}>
    <a
      class="plain jobrow"
      href={jobHref(job.id, job.workspace)}
      title="job {job.id}"
    >
      <!-- A fixed slot, so the names line up with a proof or without -->
      <span class="jobproof">
        {#if proof}
          <OutputTile
            file={outputMedia(proof, listed)}
            size="mini"
            caption="none"
            alt=""
          />
        {/if}
      </span>
      <StatusChip status={job.status} />
      <span class="wf">{workflowName(job)}</span>
    </a>
  </li>
{/snippet}

{#if openRequests.length}
  <section class="panel">
    <h2>
      <a class="plain" href={wsHref(name, 'requests')}>
        <ClipboardList size={15} /> Requests
      </a>
      <span class="flex"></span>
      <a class="plain more" href={wsHref(name, 'requests')}>all requests</a>
    </h2>
    <ul class="jobs">
      {#each openRequests as req (req.id)}
        <li>
          <a class="plain wf" href={wsHref(name, 'requests', req.id)}>
            {req.title}
          </a>
          <span class="muted">{attention(req).text}</span>
          <span class="muted num" title="items done, of those in the plan">
            {doneFigure(req)}
          </span>
        </li>
      {/each}
    </ul>
  </section>
{/if}

<div class="two">
  <section class="panel">
    <h2><a class="plain" href={wsHref(name, 'jobs')}>Jobs</a></h2>
    <LoadState
      of={jobs}
      label="jobs"
      empty={!!jobs.data &&
        jobs.data.finished.length === 0 &&
        !jobs.data.running}
    >
      {#snippet emptyIcon()}<Inbox size={36} strokeWidth={1.5} />{/snippet}
      {#snippet emptyText()}No jobs yet.{/snippet}
      {#snippet children({ running, finished })}
        <ul class="jobs">
          {#if running}{@render jobRow(running, true)}{/if}
          {#each finished as job (job.id)}{@render jobRow(job, false)}{/each}
        </ul>
      {/snippet}
    </LoadState>
  </section>

  <section class="panel">
    <h2>
      <a class="plain" href={wsHref(name, 'workflows')}>
        <Layers size={15} /> Workflows
      </a>
      {#if workflows.data}<span class="num muted">
          {workflows.data.length}
        </span>{/if}
      <span class="flex"></span>
      <a
        class="quiet icon plain"
        href={wsHref(name, 'edit')}
        title="new workflow"
        aria-label="new workflow"
      >
        <Plus size={15} />
      </a>
    </h2>
    <LoadState
      of={workflows}
      label="workflows"
      empty={workflows.data?.length === 0}
    >
      {#snippet emptyIcon()}<Layers size={36} strokeWidth={1.5} />{/snippet}
      {#snippet emptyText()}No workflows yet — create one in the editor, or run
        an example.{/snippet}
      <div class="cards">
        {#each featured as wf (wf)}
          <a
            class="plain card"
            href={wsHref(name, 'workflows', ...wf.split('/'))}
          >
            {#if proofs[wf]}
              <OutputTile
                file={outputMedia(proofs[wf], listed)}
                size="card"
                caption="none"
                alt=""
              />
            {/if}
            <span class="cardname">{leaf(wf)}</span>
          </a>
        {/each}
      </div>
    </LoadState>
  </section>
</div>

<div class="two">
  <section class="panel">
    <h2><a class="plain" href={wsHref(name, 'assets')}>Assets and disk</a></h2>
    <LoadState of={assets} label="the asset count">
      {#snippet children(count)}
        <dl>
          <dt>assets</dt>
          <dd>{`${count} ${count === 1 ? 'asset' : 'assets'}`}</dd>
          {#if usage}
            <dt>on disk</dt>
            <dd class="num" title="{usage.files} files">
              {formatBytes(usage.bytes)}
            </dd>
          {/if}
        </dl>
      {/snippet}
    </LoadState>
  </section>
  <section class="panel">
    <h2>Manage</h2>
    {#if name === DEFAULT_WORKSPACE}
      <p class="muted">
        The default workspace is the server's own folder, so it can't be hidden
        or deleted.
      </p>
    {:else}
      <button
        class="quiet withicon"
        onclick={() => setHidden(name, !hiddenWorkspaces.names.includes(name))}
        title="a choice for this browser only - the workspace itself is unchanged"
      >
        {#if hiddenWorkspaces.names.includes(name)}<Eye size={14} />Show in
          sidebar{:else}<EyeOff size={14} />Hide from sidebar{/if}
      </button>
      <ActionButton
        act={deleting}
        args={[name]}
        danger
        title="delete this workspace and everything in it"
      >
        <Trash2 size={14} />Delete workspace
      </ActionButton>
    {/if}
  </section>
</div>

<style>
  section {
    margin-bottom: var(--space-4);
  }
  h2 a {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  .more {
    font-size: var(--t-sm);
    font-weight: normal;
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
    grid-template-columns: repeat(auto-fill, 96px);
    gap: var(--space-2);
  }
  .jobs {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .jobs li,
  .jobrow {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    font-family: var(--font-mono);
    font-size: var(--t-sm);
  }
  .jobrow {
    flex: 1;
    min-width: 0;
  }
  .jobproof {
    flex: none;
    width: 48px;
  }
  .jobs .wf {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .jobs .row-live {
    padding-left: var(--space-2);
  }
  .jobrow:hover .wf {
    text-decoration: underline;
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
  /* The proof is a banner across the card's top: the card's border is its
     frame, so the tile keeps only the rule above the name */
  .card :global(.filetile) {
    border-width: 0 0 1px;
    border-radius: 0;
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
