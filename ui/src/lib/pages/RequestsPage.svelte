<script lang="ts">
  import { ClipboardList } from '@lucide/svelte'
  import { api } from '../api'
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import ItemProof from '../request/ItemProof.svelte'
  import RequestProgress from '../request/RequestProgress.svelte'
  import StateTag from '../request/StateTag.svelte'
  import {
    attention,
    doneFigure,
    notesWaiting,
    orderRequests,
    summaryBands,
  } from '../request/attention'
  import { formatMinutes, formatSpent } from '../format'
  import Time from '../ui/Time.svelte'
  import { hashState } from '../router.svelte'
  import { wsHref } from '../routes'
  import { workspace } from '../workspace.svelte'

  const name = $derived(workspace.current)

  // Open requests are what the page is for; `show=all` in the hash's query
  // brings the closed ones back, and survives opening one and coming back
  const show = hashState('show', 'open', {
    parse: (raw) => (raw === 'all' ? 'all' : 'open'),
    format: (value) => value,
  })

  // A request an agent writes shows within one feed interval, marked
  const list = resource(() => api.listRequests(workspace.current), {
    watch: ['requests'],
    rows: (data) => data.requests.map((req) => [req.id, req]),
  })
  // Why requests.json could not be read, when it is there and is no ledger
  const unreadable = $derived(list.data?.unreadable ?? null)

  const ordered = $derived(orderRequests(list.data?.requests ?? []))
  const open = $derived(ordered.filter((req) => req.state === 'open'))
  const visible = $derived(show.value === 'all' ? ordered : open)
  const closed = $derived(ordered.length - open.length)
</script>

<PageHeader title="Requests">
  {#snippet actions()}<select
      bind:value={show.value}
      title="which requests to list"
      aria-label="show"
    >
      <option value="open">open</option>
      <option value="all">all</option>
    </select>{/snippet}
</PageHeader>

<LoadState
  of={list}
  label="requests"
  empty={!unreadable && ordered.length === 0}
>
  {#snippet emptyIcon()}<ClipboardList size={36} strokeWidth={1.5} />{/snippet}
  {#snippet emptyText()}No requests in this workspace yet. One appears when you
    ask your agent for something and it writes its plan here.{/snippet}
  {#if unreadable}
    <p class="panel warn-edge" role="alert">
      This workspace's requests file could not be read: {unreadable}
    </p>
  {:else if visible.length === 0}
    <p class="muted">
      No open requests.
      <button type="button" class="quiet" onclick={() => (show.value = 'all')}>
        show closed ({closed})
      </button>
    </p>
  {:else}
    <div class="panel list">
      {#each visible as req (req.id)}
        {@const line = attention(req)}
        <a
          class="row plain"
          class:row-live={line.live}
          class:row-changed={list.changed.has(req.id)}
          href={wsHref(name, 'requests', req.id)}
        >
          <span class="cover">
            {#each req.cover as output, i (i)}
              <ItemProof
                {output}
                state="done"
                workspace={name}
                alt=""
                size="cover"
              />
            {/each}
          </span>
          <span class="namecol">
            <span class="title">{req.title}</span>
            <span class="idline">
              <code class="muted">{req.id}</code>
              <StateTag value={req.state} />
            </span>
          </span>
          <span class="attention {line.kind}">
            {#if line.kind === 'running'}
              <span class="pulse-dot" aria-hidden="true"></span>
            {/if}
            {line.text}
          </span>
          <span class="progress">
            <RequestProgress bands={summaryBands(req)} mini />
            <span class="num" title="items done, of those in the plan">
              {doneFigure(req)}
            </span>
          </span>
          <span class="figures num muted">
            {#if notesWaiting(req) > 0}
              <span>
                {notesWaiting(req)}
                {notesWaiting(req) === 1 ? 'note' : 'notes'} waiting for the agent
              </span>
            {/if}
            <span>{formatSpent(req.spent_minutes)} spent</span>
            {#if req.state === 'open' && req.remaining_minutes > 0}
              <span>{formatMinutes(req.remaining_minutes)} left</span>
            {/if}
          </span>
          <span class="updated num muted" title="last written">
            {#if req.updated_at}<Time
                at={Date.parse(req.updated_at) / 1000}
                mode="exact"
              />{/if}
          </span>
        </a>
      {/each}
    </div>
  {/if}
</LoadState>

<style>
  /* Fixed tracks past the title, so the columns line up down the list */
  .row {
    display: grid;
    grid-template-columns:
      152px minmax(0, 1.2fr) minmax(0, 1fr)
      108px 9.5rem 8.5rem;
    gap: 1rem;
    align-items: center;
  }
  .cover {
    display: flex;
    gap: 4px;
  }
  /* Written for a person, so not in the mono a workflow's name takes */
  .title {
    font-weight: 600;
    overflow-wrap: anywhere;
  }
  .idline {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0 var(--space-2);
    font-size: var(--t-xs);
  }
  .attention {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--t-sm);
    color: var(--muted);
    min-width: 0;
  }
  .attention.failed {
    color: var(--bad);
    font-weight: 600;
  }
  .attention.idle,
  .attention.needs,
  .attention.running {
    color: var(--ink);
  }
  .attention.idle,
  .attention.needs {
    font-weight: 600;
  }
  .progress {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--t-xs);
  }
  .figures {
    display: flex;
    flex-direction: column;
    font-size: var(--t-xs);
  }
  .updated {
    font-size: var(--t-xs);
    text-align: right;
  }
  /* When it was written is the first thing to go as the row narrows;
     below that everything drops under the title, beside the pictures */
  @container (max-width: 860px) {
    .row {
      grid-template-columns:
        152px minmax(0, 1.2fr) minmax(0, 1fr)
        108px 9.5rem;
    }
    .updated {
      display: none;
    }
  }
  @container (max-width: 640px) {
    .row {
      grid-template-columns: 48px minmax(0, 1fr);
      gap: 0.2rem 0.7rem;
      align-items: start;
    }
    .row > * {
      grid-column: 2;
    }
    .row > .cover {
      grid-column: 1;
      grid-row: 1 / span 4;
      flex-direction: column;
    }
    .figures {
      flex-direction: row;
      gap: var(--space-3);
    }
  }
</style>
