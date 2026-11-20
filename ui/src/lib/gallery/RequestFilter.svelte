<script lang="ts">
  import { X } from '@lucide/svelte'
  import { api } from '../api'
  import { parseRequestFilter } from '../requestFilter'
  import { hashQuery, setHashQuery } from '../router.svelte'
  import { wsHref } from '../routes'
  import type { LedgerSummary } from '../types'
  import { workspace } from '../workspace.svelte'

  // The gallery's request filter: `req=<id>` in the hash narrows it to that
  // requests-ledger request's runs (the page asks the server, reading
  // `request`). An `id/item` value from the Jobs page keeps its id. With no
  // request chosen it offers a picker of the workspace's requests, hidden
  // when there are none or the call fails.
  let { request = $bindable('') }: { request: string } = $props()

  request = parseRequestFilter(hashQuery().req).request
  $effect(() => setHashQuery({ req: request }))
  $effect(() => {
    const sync = () => (request = parseRequestFilter(hashQuery().req).request)
    window.addEventListener('hashchange', sync)
    return () => window.removeEventListener('hashchange', sync)
  })

  let requests = $state<LedgerSummary[]>([])
  $effect(() => {
    const ws = workspace.current
    let live = true
    void (async () => {
      try {
        const list = await api.listRequests(ws)
        if (live) requests = list.requests
      } catch {
        if (live) requests = []
      }
    })()
    return () => (live = false)
  })
  const title = $derived(requests.find((r) => r.id === request)?.title)
</script>

{#if request}
  <span class="reqchip" data-testid="req-filter">
    <a
      class="plain"
      href={wsHref(workspace.current, 'requests', request)}
      title={title ? `${title} - open this request` : 'open this request'}
      >request <code>{request}</code></a
    >
    <button
      class="quiet icon"
      onclick={() => (request = '')}
      title="remove the request filter"
      aria-label="remove the request filter"><X size={12} /></button
    >
  </span>
{:else if requests.length}
  <select
    bind:value={request}
    title="show only the runs made for one request"
    aria-label="filter by request"
  >
    <option value="">all requests</option>
    {#each requests as r (r.id)}
      <option value={r.id}>{r.title || r.id}</option>
    {/each}
  </select>
{/if}

<style>
  .reqchip {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    font-size: var(--t-xs);
    padding: 0.05rem 0.5rem;
  }
</style>
