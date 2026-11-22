<script lang="ts">
  import { ClipboardList } from '@lucide/svelte'
  import { api } from '../api'
  import { changes } from '../changes.svelte'
  import Empty from '../Empty.svelte'
  import Skeleton from '../ui/Skeleton.svelte'
  import ItemProof from '../request/ItemProof.svelte'
  import RequestProgress from '../request/RequestProgress.svelte'
  import StateTag from '../request/StateTag.svelte'
  import {
    attention,
    doneFigure,
    orderRequests,
    summaryBands,
  } from '../request/attention'
  import { formatMinutes, formatMtime, formatSpent } from '../format'
  import { latestOnly } from '../latest'
  import { hashQuery, setHashQuery } from '../router.svelte'
  import { wsHref } from '../routes'
  import type { LedgerSummary } from '../types'
  import { workspace } from '../workspace.svelte'

  let requests = $state<LedgerSummary[] | null>(null)
  let error = $state<string | null>(null)
  // Why requests.json could not be read, when it is there and is no ledger
  let unreadable = $state<string | null>(null)
  const name = $derived(workspace.current)

  // Open requests are what the page is for; `show=all` in the hash's query
  // brings the closed ones back, and survives opening one and coming back
  let show = $state(hashQuery().show === 'all' ? 'all' : 'open')
  $effect(() => setHashQuery({ show: show === 'all' ? 'all' : '' }))
  $effect(() => {
    const sync = () => (show = hashQuery().show === 'all' ? 'all' : 'open')
    window.addEventListener('hashchange', sync)
    return () => window.removeEventListener('hashchange', sync)
  })

  // A reply overtaken by a later load (a switch, a newer reload) is dropped
  const beginList = latestOnly()
  /** Read the ledger. A quiet load is the change feed's: it keeps the list
   * on screen while it asks, and a failed one keeps the list it had */
  function load(ws: string, { quiet = false } = {}) {
    const current = beginList()
    if (!quiet) {
      requests = null
      error = null
    }
    api
      .listRequests(ws)
      .then((r) => {
        if (!current()) return
        requests = r.requests
        unreadable = r.unreadable ?? null
        error = null
      })
      .catch((e) => {
        if (!current() || (quiet && requests !== null)) return
        requests = []
        error = e instanceof Error ? e.message : String(e)
      })
  }
  $effect(() => load(workspace.current))
  // A request an agent writes shows within one feed interval
  $effect(() => {
    const ws = workspace.current
    return changes(ws).watch(['requests'], () => load(ws, { quiet: true }))
  })

  const ordered = $derived(orderRequests(requests ?? []))
  const open = $derived(ordered.filter((req) => req.state === 'open'))
  const visible = $derived(show === 'all' ? ordered : open)
  const closed = $derived(ordered.length - open.length)
  const written = (at: string) => formatMtime(Date.parse(at) / 1000)
</script>

<div class="pagehead">
  <h1>Requests</h1>
  <select bind:value={show} title="which requests to list" aria-label="show">
    <option value="open">open</option>
    <option value="all">all</option>
  </select>
</div>

{#if error}
  <p role="alert">
    Could not load requests: {error}
    <button type="button" class="quiet" onclick={() => load(name)}>
      Retry
    </button>
  </p>
{:else if requests === null}
  <Skeleton lines={4} label="loading requests" />
{:else if unreadable}
  <p class="panel warn-edge" role="alert">
    This workspace's requests file could not be read: {unreadable}
  </p>
{:else if ordered.length === 0}
  <Empty>
    {#snippet icon()}<ClipboardList size={36} strokeWidth={1.5} />{/snippet}No
    requests in this workspace yet. One appears when you ask your agent for
    something and it writes its plan here.
  </Empty>
{:else if visible.length === 0}
  <p class="muted">
    No open requests.
    <button type="button" class="quiet" onclick={() => (show = 'all')}>
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
          <span>{formatSpent(req.spent_minutes)} spent</span>
          {#if req.state === 'open' && req.remaining_minutes > 0}
            <span>{formatMinutes(req.remaining_minutes)} left</span>
          {/if}
        </span>
        <span class="updated num muted" title="last written">
          {req.updated_at ? written(req.updated_at) : ''}
        </span>
      </a>
    {/each}
  </div>
{/if}

<style>
  .pagehead h1 {
    margin: 0;
    flex: 1;
  }
  .pagehead select {
    max-width: 110px;
  }
  .list {
    padding: 0.3rem;
  }
  /* Fixed tracks past the title, so the columns line up down the list */
  .row {
    display: grid;
    grid-template-columns:
      152px minmax(0, 1.2fr) minmax(0, 1fr)
      108px 9.5rem 8.5rem;
    gap: 1rem;
    align-items: center;
    padding: 0.55rem 0.8rem;
    border-radius: 6px;
    color: var(--ink);
  }
  .row:hover {
    background: var(--panel-2);
  }
  .cover {
    display: flex;
    gap: 4px;
  }
  .namecol {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    min-width: 0;
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
  .attention.running {
    color: var(--ink);
  }
  .attention.idle {
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
