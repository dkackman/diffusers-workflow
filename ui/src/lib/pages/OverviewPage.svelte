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
  import ItemProof from '../request/ItemProof.svelte'
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
  import Time from '../ui/Time.svelte'
  import { workflowName } from '../terms'
  import { compareNames } from '../names'
  import { hiddenWorkspaces, setHidden } from '../hiddenWorkspaces.svelte'
  import { latestProofs, proofByRun } from '../proofs'
  import { fileHref, jobHref, sharedHref, wsHref } from '../routes'
  import type { JobSummary } from '../types'
  import {
    DEFAULT_WORKSPACE,
    invalidateWorkspaces,
    loadWorkspaces,
    workspace,
  } from '../workspace.svelte'
  import { deleteWorkspaceWithConfirm } from '../workspaceActions'
  import { action } from '../action.svelte'
  import ActionButton from '../ui/ActionButton.svelte'
  import FirstRunCard from '../FirstRunCard.svelte'
  import { firstRunLink } from '../firstRun'

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
      const listing = await api.listWorkflows()
      return {
        listing,
        names: listing.workflows.filter(
          (wf) => listing.details?.[wf]?.writable !== false,
        ),
      }
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
  // this is how it stays fresh, not where it is held. The Sidebar, always
  // mounted, relists on a `workspaces` move; this one only joins that read
  // (`loadWorkspaces` hands back the one on its way), so a move lists the
  // workspaces once, and it never reads at first: the shell already has
  resource(() => loadWorkspaces(), { watch: ['workspaces'], lazy: true })

  // The ones that have produced something first - the same order the catalog
  // uses, so the overview and the catalog agree on what is "recent"
  const proofs = $derived(gallery.data?.proofs ?? {})
  const runs = $derived(gallery.data?.runs ?? {})
  // The workspace the listing came from: every tile's thumbnail and link
  // names it, so a switch never asks the new workspace for the old files
  const listed = $derived(gallery.data?.workspace ?? name)
  const featured = $derived(
    [...(workflows.data?.names ?? [])]
      .sort(
        (l, r) =>
          (proofs[l] ? 0 : 1) - (proofs[r] ? 0 : 1) || compareNames(l, r),
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
{#if workflows.status !== 'loading'}
  <FirstRunCard link={firstRunLink(workflows.data?.listing ?? null, name)} />
{/if}
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
    {#snippet emptyText()}Nothing generated yet.
      <a href={wsHref(name, 'workflows')}>Run a workflow</a> and what it makes lands
      here.{/snippet}
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
  {@const when = job.finished_at ?? job.started_at ?? job.created_at ?? null}
  <li class:row-live={live}>
    <a
      class="plain jobrow"
      href={jobHref(job.id, job.workspace)}
      title="job {job.id}"
    >
      <!-- The run's proof, or the dashed slot that says why there is none:
           every row is the same height -->
      <span class="jobproof">
        <ItemProof
          output={proof ? { name: proof.name, kind: proof.kind } : null}
          state={job.status}
          workspace={listed}
          alt=""
          size="cover"
        />
      </span>
      <span class="jobtext">
        <span class="line1">
          <StatusChip status={job.status} />
          <span class="wf">{workflowName(job)}</span>
        </span>
        <span class="when">
          {#if when !== null}<span class="at">
              <Time at={when} />
            </span><span class="sep" aria-hidden="true">
              &nbsp;·&nbsp;
            </span>{/if}<span class="jid" title={job.id}>{job.id}</span>
        </span>
      </span>
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
        <li class="reqrow">
          <a class="plain wf" href={wsHref(name, 'requests', req.id)}>
            {req.title}
          </a>
          <span class="muted num" title="items done, of those in the plan">
            {doneFigure(req)}
          </span>
          {#if req.updated_at}<span class="muted reqwhen">
              <Time at={Date.parse(req.updated_at) / 1000} label="changed" />
            </span>{/if}
          <span class="muted reqline">{attention(req).text}</span>
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
      {#snippet emptyText()}No jobs yet.
        <a href={wsHref(name, 'workflows')}>Run a workflow</a> to start one.{/snippet}
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
          {workflows.data.names.length}
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
      empty={workflows.data?.names.length === 0}
    >
      {#snippet emptyIcon()}<Layers size={36} strokeWidth={1.5} />{/snippet}
      {#snippet emptyText()}No workflows yet.
        <a href={wsHref(name, 'edit')}>Create one in the editor</a>, or
        <a href={sharedHref('examples')}>start from an example</a>.{/snippet}
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
  .jobs .jobtext {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: var(--space-1);
  }
  .jobs .line1 {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
  }
  /* The title and the figure on a line, what it needs under them: the
     sentence is long, and must not push the title out at 390 px */
  .jobs .reqrow {
    flex-wrap: wrap;
    row-gap: 0;
  }
  .jobs .reqwhen {
    font-size: var(--t-xs);
  }
  .jobs .reqline {
    flex: 1 0 100%;
    font-family: inherit;
    font-size: var(--t-xs);
    overflow-wrap: anywhere;
  }
  .jobs .wf {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  /* "<time> · <id>" on its own line: the id tells apart jobs that finish in
     one minute, so the time gives way first (it ellipsizes, and a phone
     drops it); an id longer than the row is clipped, and its title holds
     the whole of it */
  .jobs .when {
    display: flex;
    min-width: 0;
    color: var(--muted);
    white-space: pre;
    font-size: var(--t-xs, var(--t-sm));
  }
  .jobs .when .sep {
    flex: none;
  }
  .jobs .when .jid {
    flex: 0 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .jobs .when .at {
    flex: 0 1000 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  @media (max-width: 600px) {
    .jobs .when .at,
    .jobs .when .sep {
      display: none;
    }
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
