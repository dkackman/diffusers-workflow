<script lang="ts">
  import { X } from '@lucide/svelte'
  import { api } from '../api'
  import { formatRequestFilter, parseRequestFilter } from '../requestFilter'
  import { hashState } from '../router.svelte'
  import { wsHref } from '../routes'
  import { resource } from '../resource.svelte'
  import { untrack } from 'svelte'
  import { workspace as current } from '../workspace.svelte'

  // The request filter of the Gallery and the Jobs page: `req=<id>` in the
  // hash narrows the page to that requests-ledger request's runs,
  // `req=<id>/<item>` to one item's (the page asks the server, reading
  // `request` and `item`). With no request chosen it offers a picker of
  // `workspace`'s requests (the current one unless the page pins another),
  // hidden when there are none, the call fails or it is not `pickable`;
  // picking one clears any item.
  let {
    request = $bindable(''),
    item = $bindable(''),
    workspace = '',
    pickable = true,
  }: {
    request: string
    item: string
    workspace?: string
    pickable?: boolean
  } = $props()
  const ws = $derived(workspace || current.current)

  const req = hashState('req', '')
  const fromHash = () => {
    const parsed = parseRequestFilter(req.value)
    request = parsed.request
    item = parsed.request ? parsed.item : ''
  }
  // Read at init, so the page's first listing already asks for it
  fromHash()
  $effect(() => {
    void req.value
    untrack(fromHash)
  })
  // The page clearing its filters clears this one too
  $effect(() => {
    const value = formatRequestFilter({ request, item })
    untrack(() => {
      if (value !== req.value) req.value = value
    })
  })

  // The picker is a convenience, so a failed read hides it rather than
  // showing an error; a request an agent writes shows within one feed interval
  const list = resource(() => api.listRequests(ws), {
    watch: ['requests'],
    workspace: () => ws,
  })
  const requests = $derived(list.data?.requests ?? [])
  const title = $derived(requests.find((r) => r.id === request)?.title)
</script>

{#if request}
  <span class="reqchip" data-testid="req-filter">
    <a
      class="plain"
      href={wsHref(ws, 'requests', request)}
      title={`${request} - open this request`}
    >
      {#if title}request: {title}{:else}request: <code>{request}</code>{/if}
      {#if item}
        › <code>{item}</code>{/if}
    </a>
    <button
      class="quiet icon"
      onclick={() => {
        request = ''
        item = ''
      }}
      title="remove the request filter"
      aria-label="remove the request filter"
    >
      <X size={12} />
    </button>
  </span>
{:else if pickable && requests.length}
  <select
    bind:value={request}
    title="show only the jobs made for one request"
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
