<script lang="ts">
  import { untrack } from 'svelte'
  import { api, ApiError } from '../api'
  import Skeleton from '../ui/Skeleton.svelte'
  import ClampedText from '../ui/ClampedText.svelte'
  import PlanChanges from '../request/PlanChanges.svelte'
  import PlanItems from '../request/PlanItems.svelte'
  import RequestHeader from '../request/RequestHeader.svelte'
  import { announce } from '../request/announce'
  import { planState, titleId } from '../request/itemState'
  import { PlanStore } from '../request/planStore.svelte'
  import { poll } from '../poll'
  import { refreshInterval } from '../request/refresh'
  import { formatMtime } from '../format'
  import { latestOnly } from '../latest'
  import { wsHref } from '../routes'
  import type { LedgerRequest } from '../types'
  import { workspace } from '../workspace.svelte'

  let { requestId }: { requestId: string } = $props()

  let missing = $state(false)
  let error = $state<string | null>(null)
  // When the request held was read, and why the read after it failed: a
  // failed refresh keeps what is on the page and says how old it is
  let readAt = $state(0)
  let stale = $state<string | null>(null)
  // What the one live region says: a refresh's changes in a sentence, or
  // what an edit of the user's own did
  let said = $state('')
  const ws = $derived(workspace.current)

  // What the server last said, and the edits not yet written over it
  // (planStore.svelte.ts). The page draws the store's view; every edit is
  // an intent handed to it, applied to the read only when it is sent
  const store = new PlanStore({
    get: () => api.getRequest(ws, requestId),
    put: (body) => api.putRequest(ws, requestId, body),
    onread(previous, next) {
      const news = announce(previous, next)
      if (news) said = news
      readAt = Date.now() / 1000
    },
  })
  const request = $derived(store.view)
  const plan = $derived(request ? planState(request) : null)

  // A reply older than what is held (a read sent before a save and
  // answered after it) does not replace it: the store keeps the newer
  function take(found: LedgerRequest) {
    store.take(found)
    missing = false
    error = null
    stale = null
  }
  // A read that fails over a request the page holds keeps it - and with it
  // the note drafts in its rows - unless the request is gone
  function failed(e: unknown) {
    const message = e instanceof Error ? e.message : String(e)
    if (e instanceof ApiError && e.status === 404) {
      store.reset()
      missing = true
    } else if (store.read) stale = message
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
    untrack(() => store.reset())
    missing = false
    error = null
    stale = null
    said = ''
    void load()
  })

  // The plan and its machine state move without this page: an agent rewrites
  // the plan, a job is queued, runs, ends. Follow it for as long as it can
  // change (refreshInterval). A refresh replaces the store's read and
  // leaves its queue, so it never undoes an edit; a note's draft lives in
  // its row, which the keyed list keeps - and a refresh never moves focus
  const interval = $derived(refreshInterval(store.read))
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

  /** The note on the whole request: an intent like any other edit */
  const saveNote = (text: string | null) =>
    store.push({ kind: 'note', key: null, text }).then(() => {})

  /** Show one write among the plan's changes: the notice's "See changes" */
  function seeChanges(revision: number) {
    const entry = document.getElementById(`plan-change-r${revision}`)
    const target = entry ?? rail
    if (!target) return
    const fold = target.closest('details')
    if (fold) fold.open = true
    target.tabIndex = -1
    target.scrollIntoView?.({ block: 'center' })
    target.focus({ preventScroll: true })
  }
  let rail = $state<HTMLElement>()

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
          onsavenote={saveNote}
          onreload={load}
        />
        <PlanItems
          {store}
          {request}
          {plan}
          workspace={ws}
          ongo={goTo}
          onreload={load}
          onsay={(text) => (said = text)}
          onsee={seeChanges}
        />
      </div>

      <!-- Not an <aside>: the app's drawer styles reach every one below 900px -->
      <div
        class="rail"
        bind:this={rail}
        role="complementary"
        aria-label="The ask, the goal and the plan's changes"
      >
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
      </div>
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
  summary {
    cursor: pointer;
    padding: 0.55rem 0.8rem;
    font-size: var(--t-sm);
    font-weight: 600;
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
