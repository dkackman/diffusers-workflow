<script lang="ts">
  import { ClipboardList } from '@lucide/svelte'
  import { api } from '../api'
  import Empty from '../Empty.svelte'
  import Skeleton from '../ui/Skeleton.svelte'
  import { formatMinutes } from '../format'
  import { statusCounts } from '../requestResult'
  import { wsHref } from '../routes'
  import type { LedgerSummary } from '../types'
  import { workspace } from '../workspaceState.svelte'

  let requests = $state<LedgerSummary[] | null>(null)
  let error = $state<string | null>(null)
  const name = $derived(workspace.current)

  $effect(() => {
    const ws = workspace.current
    requests = null
    error = null
    api
      .listRequests(ws)
      .then((r) => (requests = r.requests))
      .catch((e) => {
        requests = []
        error = e instanceof Error ? e.message : String(e)
      })
  })
</script>

<h1>Requests</h1>
<p class="muted">
  What was asked for in this workspace, as a plan of items and the jobs made for
  each.
</p>

{#if error}
  <p class="muted">Could not load requests: {error}</p>
{:else if requests === null}
  <Skeleton lines={4} label="loading requests" />
{:else if requests.length === 0}
  <Empty
    >{#snippet icon()}<ClipboardList size={36} strokeWidth={1.5} />{/snippet}No
    requests yet — an agent opens one with set_request.</Empty
  >
{:else}
  <ul class="rows">
    {#each requests as req (req.id)}
      <li class="panel">
        <div class="top">
          <a class="plain title" href={wsHref(name, 'requests', req.id)}
            >{req.title}</a
          >
          <span class="state {req.state}">{req.state}</span>
        </div>
        <div class="meta muted">
          <code>{req.id}</code>
          <span>{req.items} {req.items === 1 ? 'item' : 'items'}</span>
          {#if statusCounts(req.status_counts)}<span
              >{statusCounts(req.status_counts)}</span
            >{/if}
          <span class="num"
            >{formatMinutes(req.spent_minutes)} spent · {formatMinutes(
              req.remaining_minutes,
            )} remaining</span
          >
        </div>
      </li>
    {/each}
  </ul>
{/if}

<style>
  .rows {
    list-style: none;
    margin: var(--space-4) 0 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .top {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
  }
  .title {
    font-weight: 600;
    flex: 1;
  }
  .state {
    font-size: var(--t-xs);
    font-family: var(--font-mono);
    color: var(--muted);
  }
  .state.open {
    color: var(--ink);
  }
  .meta {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-3);
    font-size: var(--t-sm);
    margin-top: var(--space-1);
  }
</style>
