<script lang="ts">
  import { X } from '@lucide/svelte'
  import { api } from '../api'
  import { formatRequestFilter, parseRequestFilter } from '../requestFilter'
  import { hashQuery, setHashQuery } from '../router.svelte'
  import { wsHref } from '../routes'
  import { resource } from '../resource.svelte'
  import { workspace } from '../workspace.svelte'

  // The gallery's request filter: `req=<id>` in the hash narrows it to that
  // requests-ledger request's runs, `req=<id>/<item>` to one item's (the
  // page asks the server, reading `request` and `item`). With no request
  // chosen it offers a picker of the workspace's requests, hidden when
  // there are none or the call fails; picking one clears any item.
  let {
    request = $bindable(''),
    item = $bindable(''),
  }: { request: string; item: string } = $props()

  const fromHash = () => {
    const parsed = parseRequestFilter(hashQuery().req)
    request = parsed.request
    item = parsed.request ? parsed.item : ''
  }
  fromHash()
  $effect(() => setHashQuery({ req: formatRequestFilter({ request, item }) }))
  $effect(() => {
    window.addEventListener('hashchange', fromHash)
    return () => window.removeEventListener('hashchange', fromHash)
  })

  // The picker is a convenience, so a failed read hides it rather than
  // showing an error; a request an agent writes shows within one feed interval
  const list = resource(() => api.listRequests(workspace.current), {
    watch: ['requests'],
  })
  const requests = $derived(list.data?.requests ?? [])
  const title = $derived(requests.find((r) => r.id === request)?.title)
</script>

{#if request}
  <span class="reqchip" data-testid="req-filter">
    <a
      class="plain"
      href={wsHref(workspace.current, 'requests', request)}
      title={title ? `${title} - open this request` : 'open this request'}
    >
      {#if item}<code>{request} › {item}</code>{:else}request <code>
          {request}
        </code>{/if}
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
{:else if requests.length}
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
