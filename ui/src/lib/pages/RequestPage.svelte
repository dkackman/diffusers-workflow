<script lang="ts">
  import { api, ApiError } from '../api'
  import Skeleton from '../ui/Skeleton.svelte'
  import RequestItem from '../request/RequestItem.svelte'
  import { poll } from '../poll'
  import { formatExact, formatMinutes } from '../format'
  import { latestOnly } from '../latest'
  import { wsHref } from '../routes'
  import type { LedgerRequest } from '../types'
  import { workspace } from '../workspace.svelte'

  let { requestId }: { requestId: string } = $props()

  let request = $state<LedgerRequest | null>(null)
  let missing = $state(false)
  let error = $state<string | null>(null)
  const ws = $derived(workspace.current)

  // A reply for a request the page has since left is dropped
  const beginLoad = latestOnly()
  async function load() {
    const current = beginLoad()
    try {
      const found = await api.getRequest(ws, requestId)
      if (!current()) return
      request = found
      missing = false
      error = null
    } catch (e) {
      if (!current()) return
      request = null
      if (e instanceof ApiError && e.status === 404) missing = true
      else error = e instanceof Error ? e.message : String(e)
    }
  }

  $effect(() => {
    void ws
    void requestId
    request = null
    missing = false
    error = null
    void load()
  })

  // Machine state moves on its own: follow it while a job is in flight. A
  // comment draft lives in its item component, so a refresh never touches it
  const inFlight = $derived(
    request?.items.some((i) => i.status === 'queued' || i.status === 'running'),
  )
  $effect(() => {
    if (!inFlight) return
    return poll(
      () =>
        api
          .getRequest(ws, requestId)
          .then((r) => (request = r))
          .catch(() => {}),
      5000,
      { immediate: false },
    )
  })

  /** The one edit: the plan goes back as read, one comment replaced. A stale
   * revision rejects with the server's 409 and the item shows it. */
  async function saveComment(key: string, text: string | null) {
    const read = request
    if (!read) return
    request = await api.putRequest(ws, requestId, {
      title: read.title,
      goal: read.goal,
      state: read.state,
      revision: read.revision,
      why: `comment on ${key}`,
      items: read.items.map((item) =>
        item.key === key
          ? { ...item, comment: text === null ? null : { text } }
          : item,
      ) as LedgerRequest['items'],
    })
  }

  const history = $derived(
    request ? [...request.history].sort((a, b) => b.revision - a.revision) : [],
  )
  const when = (at: string) => formatExact(Date.parse(at) / 1000)
</script>

<p class="muted back"><a href={wsHref(ws, 'requests')}>all requests</a></p>

{#if missing}
  <p>This request no longer exists.</p>
{:else if error}
  <p class="muted">Could not load the request: {error}</p>
{:else if request === null}
  <Skeleton lines={4} label="loading request" />
{:else}
  <div class="head">
    <h1>{request.title}</h1>
    <span class="state">{request.state}</span>
    <span class="muted num">revision {request.revision}</span>
  </div>
  <p class="muted">
    <code>{request.id}</code> ·
    <a href={`${wsHref(ws, 'jobs')}?req=${encodeURIComponent(request.id)}`}
      >its jobs</a
    >
    ·
    <a href={`${wsHref(ws, 'gallery')}?req=${encodeURIComponent(request.id)}`}
      >its outputs</a
    >
  </p>

  {#if request.prompt}
    <section class="panel">
      <h2>Asked</h2>
      <p class="verbatim">{request.prompt}</p>
    </section>
  {/if}
  <section class="panel">
    <h2>Goal</h2>
    <p class="verbatim">{request.goal}</p>
    <p class="muted num">
      {formatMinutes(request.spent_minutes)} spent · {formatMinutes(
        request.remaining_minutes,
      )} remaining
    </p>
  </section>

  <h2>Items</h2>
  <ol class="items">
    {#each request.items as item (item.key)}
      <RequestItem
        {item}
        workspace={ws}
        onsave={(text) => saveComment(item.key, text)}
        onreload={load}
      />
    {/each}
  </ol>

  <h2>History</h2>
  <ul class="history">
    {#each history as entry (entry.revision)}
      <li>
        <span class="num">r{entry.revision}</span>
        <span class="muted">{when(entry.at)}</span>
        <span class="by">{entry.by}</span>
        <span>{entry.why}</span>
        {#if entry.goal}<p class="muted verbatim goal">
            goal: {entry.goal}
          </p>{/if}
      </li>
    {/each}
  </ul>
{/if}

<style>
  .back {
    font-size: var(--t-sm);
  }
  .head {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
  }
  .state {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
  }
  .verbatim {
    white-space: pre-wrap;
    margin: 0 0 var(--space-2);
  }
  section {
    margin: var(--space-3) 0;
  }
  .items {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .history {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    font-size: var(--t-sm);
  }
  .history li {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-3);
  }
  .by {
    font-family: var(--font-mono);
  }
  .goal {
    flex-basis: 100%;
    margin: 0;
  }
</style>
