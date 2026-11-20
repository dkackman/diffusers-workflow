<script lang="ts">
  import { api, ApiError } from '../api'
  import Skeleton from '../ui/Skeleton.svelte'
  import ClampedText from '../ui/ClampedText.svelte'
  import PlanChanges from '../request/PlanChanges.svelte'
  import RequestHeader from '../request/RequestHeader.svelte'
  import RequestItem from '../request/RequestItem.svelte'
  import { announce } from '../request/announce'
  import { planState, titleId } from '../request/itemState'
  import { poll } from '../poll'
  import { newerOf, refreshInterval } from '../request/refresh'
  import { formatMtime } from '../format'
  import { latestOnly } from '../latest'
  import { wsHref } from '../routes'
  import type { LedgerRequest } from '../types'
  import { workspace } from '../workspace.svelte'

  let { requestId }: { requestId: string } = $props()

  let request = $state<LedgerRequest | null>(null)
  let missing = $state(false)
  let error = $state<string | null>(null)
  // When the request held was read, and why the read after it failed: a
  // failed refresh keeps what is on the page and says how old it is
  let readAt = $state(0)
  let stale = $state<string | null>(null)
  // What the one live region says: a refresh's changes in a sentence
  let said = $state('')
  const ws = $derived(workspace.current)
  const plan = $derived(request ? planState(request) : null)

  function hold(next: LedgerRequest) {
    const news = announce(request, next)
    if (news) said = news
    request = next
    readAt = Date.now() / 1000
    missing = false
    error = null
    stale = null
  }
  // A reply older than what is held (a read sent before a save and
  // answered after it) does not replace it
  const take = (found: LedgerRequest) => hold(newerOf(request, found))
  // A read that fails over a request the page holds keeps it - and with it
  // the note drafts in its rows - unless the request is gone
  function failed(e: unknown) {
    const message = e instanceof Error ? e.message : String(e)
    if (e instanceof ApiError && e.status === 404) {
      request = null
      missing = true
    } else if (request) stale = message
    else error = message
  }

  // A reply for a request the page has since left is dropped
  const beginLoad = latestOnly()
  async function load() {
    const current = beginLoad()
    try {
      const found = await api.getRequest(ws, requestId)
      if (current()) take(found)
    } catch (e) {
      if (current()) failed(e)
    }
  }

  $effect(() => {
    void ws
    void requestId
    request = null
    missing = false
    error = null
    stale = null
    said = ''
    void load()
  })

  // The plan and its machine state move without this page: an agent rewrites
  // the plan, a job is queued, runs, ends. Follow it for as long as it can
  // change (refreshInterval), keeping the newer of what is held and what
  // came back (newerOf). A note's draft lives in its row, which the keyed
  // list keeps, so a refresh never touches it - and never moves focus
  const interval = $derived(refreshInterval(request))
  $effect(() => {
    if (interval === null) return
    const [from, id] = [ws, requestId]
    let live = true
    const stop = poll(
      async () => {
        try {
          const found = await api.getRequest(from, id)
          if (live) take(found)
        } catch (e) {
          if (live) failed(e)
        }
      },
      interval,
      { immediate: false },
    )
    return () => {
      live = false
      stop()
    }
  })

  /** The one edit: the plan goes back as read, one note replaced - an
   * item's by its key, the request's own with null. A stale revision
   * rejects with the server's 409 and the note shows it. */
  async function saveComment(key: string | null, text: string | null) {
    const read = request
    if (!read) return
    const [to, id] = [ws, requestId]
    const comment = text === null ? null : { text }
    const saved = await api.putRequest(to, id, {
      title: read.title,
      goal: read.goal,
      state: read.state,
      revision: read.revision,
      why: key === null ? 'comment on the request' : `comment on ${key}`,
      ...(key === null ? { comment } : {}),
      items: read.items.map((item) =>
        item.key === key ? { ...item, comment } : item,
      ) as LedgerRequest['items'],
    })
    // Left for another request while the save was out: it is not this one's
    if (to === ws && id === requestId) hold(saved)
  }

  /** Go to an item's row: a progress segment or a dependency's name */
  function goTo(key: string) {
    const title = document.getElementById(titleId(key))
    if (!title) return
    const still = window.matchMedia?.('(prefers-reduced-motion: reduce)')
    title.scrollIntoView?.({
      behavior: still?.matches ? 'auto' : 'smooth',
      block: 'center',
    })
    title.focus({ preventScroll: true })
  }

  // Under 640px the rail's sections fold: what the page is for comes first.
  // The page's own width, not the viewport's - the sidebar takes its share
  let page = $state<HTMLElement>()
  let width = $state(0)
  $effect(() => {
    // jsdom has no ResizeObserver; the page then stays unfolded
    if (!page || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(() => (width = page?.clientWidth ?? 0))
    observer.observe(page)
    return () => observer.disconnect()
  })
  const narrow = $derived(width > 0 && width < 640)
</script>

{#snippet ask(req: LedgerRequest)}
  {#if req.prompt}
    <section>
      <h2>What you asked</h2>
      <ClampedText text={req.prompt} lines={6} />
    </section>
  {/if}
  <section>
    <h2>Goal</h2>
    <ClampedText text={req.goal} lines={6} />
  </section>
{/snippet}

<p class="sr-only" aria-live="polite" data-testid="request-live">{said}</p>

<div class="reqpage" bind:this={page}>
  <p class="muted back"><a href={wsHref(ws, 'requests')}>all requests</a></p>

  {#if missing}
    <p>
      This request no longer exists. The agent may have removed it.
      <a href={wsHref(ws, 'requests')}>See all requests</a>
    </p>
  {:else if error}
    <p role="alert">
      Could not load this request: {error}
      <button type="button" class="quiet" onclick={load}>Retry</button>
    </p>
  {:else if request === null || plan === null}
    <Skeleton lines={4} label="loading request" />
  {:else}
    {#if stale}
      <div class="panel warn-edge stale" role="status">
        <span
          >Showing the plan as of {formatMtime(readAt)}. Could not refresh: {stale}</span
        >
        <button type="button" class="quiet" onclick={load}>Retry</button>
      </div>
    {/if}
    <div class="layout">
      <div class="maincol">
        <RequestHeader
          {request}
          {plan}
          workspace={ws}
          onselect={goTo}
          onsavenote={(text) => saveComment(null, text)}
          onreload={load}
        />
        {#if plan.items.length === 0}
          <p class="muted">
            The agent has not planned any items for this request yet.
          </p>
        {:else}
          <div class="panel list">
            <ol class="items" aria-label="The plan">
              {#each plan.live as row (row.item.key)}
                <RequestItem
                  {row}
                  {request}
                  workspace={ws}
                  onsave={(text) => saveComment(row.item.key, text)}
                  onreload={load}
                  ongo={goTo}
                />
              {/each}
            </ol>
            {#if plan.dropped.length}
              <details class="droppedgroup">
                <summary>Dropped ({plan.dropped.length})</summary>
                <ul class="items">
                  {#each plan.dropped as row (row.item.key)}
                    <RequestItem
                      {row}
                      {request}
                      workspace={ws}
                      onsave={(text) => saveComment(row.item.key, text)}
                      onreload={load}
                      ongo={goTo}
                    />
                  {/each}
                </ul>
              </details>
            {/if}
          </div>
        {/if}
      </div>

      <aside class="rail" aria-label="The ask, the goal and the plan's changes">
        {#if narrow}
          <details>
            <summary>What you asked, and the goal</summary>
            {@render ask(request)}
          </details>
          <details>
            <summary>Plan changes ({request.history.length})</summary>
            <PlanChanges
              history={request.history}
              revision={request.revision}
              heading={false}
            />
          </details>
        {:else}
          {@render ask(request)}
          <PlanChanges history={request.history} revision={request.revision} />
        {/if}
      </aside>
    </div>
  {/if}
</div>

<style>
  .reqpage {
    container-type: inline-size;
  }
  .back {
    font-size: var(--t-sm);
    margin: 0 0 var(--space-3);
  }
  .stale {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2) var(--space-4);
    padding: 0.55rem 0.8rem;
    margin-bottom: var(--space-4);
    font-size: var(--t-sm);
  }
  .layout {
    display: flex;
    flex-direction: column;
    gap: var(--space-5);
  }
  .maincol {
    min-width: 0;
  }
  .list {
    padding: 0.3rem;
  }
  .items {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .items > :global(li + li) {
    border-top: 1px solid var(--line);
  }
  .droppedgroup {
    border-top: 1px solid var(--line);
  }
  summary {
    cursor: pointer;
    padding: 0.55rem 0.8rem;
    font-size: var(--t-sm);
    font-weight: 600;
  }
  .droppedgroup summary {
    color: var(--muted);
  }
  .rail {
    display: flex;
    flex-direction: column;
    gap: var(--space-5);
    min-width: 0;
  }
  .rail details {
    border-top: 1px solid var(--line);
  }
  .rail details summary {
    padding-left: 0;
  }
  .rail details section + section {
    margin-top: var(--space-4);
  }
  /* What was asked and the goal are written for a person: sans, at the
     rail's size, not the mono of a heading that names an engine thing */
  .rail h2 {
    font-family: var(--font-sans);
    font-size: var(--t-base);
    margin: 0 0 var(--space-1);
  }
  .rail section {
    font-size: var(--t-sm);
  }
  @container (min-width: 880px) {
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
      align-items: start;
    }
    /* Under the app's own sticky header, as the editor's bar is */
    .rail {
      position: sticky;
      top: 76px;
      max-height: calc(100vh - 76px - var(--space-4));
      overflow-y: auto;
    }
  }
</style>
