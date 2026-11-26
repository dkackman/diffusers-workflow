<script lang="ts">
  import { onDestroy, tick, untrack } from 'svelte'
  import { ApiError } from '../api'
  import { describeError } from '../apiErrors'
  import Details from '../ui/Details.svelte'
  import ErrorLine from '../ui/ErrorLine.svelte'
  import Skeleton from '../ui/Skeleton.svelte'
  import ClampedText from '../ui/ClampedText.svelte'
  import CloseProposal from '../request/CloseProposal.svelte'
  import PlanChanges from '../request/PlanChanges.svelte'
  import PlanItems from '../request/PlanItems.svelte'
  import RequestHeader from '../request/RequestHeader.svelte'
  import RequestMenu from '../request/RequestMenu.svelte'
  import RunDialog from '../request/RunDialog.svelte'
  import {
    heldPlan,
    itemPlan,
    readyPlan,
    type RunPlan,
  } from '../request/runPlan'
  import { ReviewCountdown } from '../request/review.svelte'
  import { goWs } from '../router.svelte'
  import { announce } from '../request/announce'
  import * as seen from '../request/planDiff'
  import { answeredNow, answeredText, noteItem } from '../request/answered'
  import { noteAnchor } from '../request/noteState'
  import { planState, titleId } from '../request/itemState'
  import { changedText, goneText } from '../request/conflict'
  import {
    PlanStore,
    requestRoutes,
    type PlanIo,
  } from '../request/planStore.svelte'
  import { itemRows, newerOf, refreshInterval } from '../request/refresh'
  import { resource } from '../resource.svelte'
  import Time from '../ui/Time.svelte'
  import { SECTION_LABELS, wsHref } from '../routes'
  import PageHeader from '../ui/PageHeader.svelte'
  import { notify } from '../toast'
  import type { LedgerRequest } from '../types'
  import { workspace } from '../workspace.svelte'

  let { requestId }: { requestId: string } = $props()

  // When the request held was read: a failed refresh keeps what is on the
  // page and says how old it is
  let readAt = $state(0)
  // What the one live region says: a refresh's changes in a sentence, or
  // what an edit of the user's own did
  let said = $state('')
  // One page, one store, one request (RequestRoute keys it): fixed here, so
  // a store still writing after the page has gone writes to its own request
  const ws = untrack(() => workspace.current)
  const id = untrack(() => requestId)

  // Held items count down from each read the store keeps; one reaching
  // zero reads the request again (review.svelte.ts)
  const countdown = new ReviewCountdown(() => void reads.reload())

  // What the server last said, and the edits not yet written over it
  // (planStore.svelte.ts). The page draws the store's view; every edit is
  // an intent handed to it, applied to the read only when it is sent
  const routes = requestRoutes(ws, id)
  const io: PlanIo = {
    ...routes,
    onread(previous, next) {
      const news = announce(previous, next)
      if (news) said = news
      readAt = Date.now() / 1000
      countdown.read(next)
      // What the agent changed, and since this browser last looked (planDiff)
      looked = seen.afterRead(ws, previous, next, looked)
      if (previous)
        for (const note of answeredNow(previous, next)) told(next, note.id)
    },
    // The user's own edit is never marked as an agent's: the next quiet
    // read diffs against what the save read back
    onsaved: (held) => reads.adopt(held),
    // A stale retry's fresh read is another writer's change: marked
    onquiet: (read) => reads.adopt(read, { mark: true }),
    onunshown: (conflict) =>
      notify.error(
        conflict.kind === 'changed'
          ? `Not saved. ${conflict.title}: ${changedText(conflict)}`
          : goneText(conflict),
      ),
  }
  const store = new PlanStore(io, `${ws}/${id}`)
  onDestroy(() => store.leave())

  // The agent wrote after one of the user's waiting notes: said once per
  // note while the page is open, never for one answered before it opened.
  let toldOf: Record<string, true> = {}
  let looked = $state.raw(seen.UNSEEN)
  function told(next: LedgerRequest, id: string) {
    if (toldOf[id]) return
    toldOf[id] = true
    const key = noteItem(next, id)
    notify.news(answeredText(next), {
      label: 'Show the note',
      onClick: () => void showNote(key, id),
    })
  }
  const request = $derived(store.view)
  const plan = $derived(request ? planState(request) : null)

  // The plan and its machine state move without this page: an agent rewrites
  // the plan, a job is queued, runs, ends. The feed's `requests` kind says
  // so within one CHANGES_POLL_MS, and while the request can change on its
  // own (refreshInterval) a timer reads it too. A read replaces the store's
  // read through `take` - so through `newerOf`, and a reply sent before a
  // save never puts the old revision back - and leaves its queue, so it
  // never undoes an edit; a note's draft lives in its row, which the keyed
  // list keeps, and a refresh never moves focus. An item a quiet read
  // changed is marked (`changed`)
  // The interval is derived, so a read that keeps it does not re-arm the
  // timer: the resource's effect reads `interval`, not `store.read`
  const interval = $derived(refreshInterval(store.read))
  const reads = resource((current) => store.refresh(current), {
    watch: ['requests'],
    alsoLive: () => interval,
    merge: newerOf,
    rows: itemRows,
  })
  // A request that is gone is said so, and nothing of it is kept; a read
  // that fails over one the page holds keeps it - and with it the note
  // drafts in its rows - and says how old it is
  // A loud read (a switch, Retry) shows the skeleton, never the failure of
  // the request it is leaving
  const settled = $derived(reads.status !== 'loading')
  const missing = $derived(
    settled && reads.error instanceof ApiError && reads.error.status === 404,
  )
  $effect(() => {
    if (missing) untrack(() => store.reset())
  })
  const failed = $derived(settled && reads.error !== undefined && !missing)
  const error = $derived(!store.read && failed)
  const stale = $derived(
    store.read && failed ? describeError(reads.error) : null,
  )
  const load = () => reads.reload()

  // The priced run dialog, for what is ready or for one item. The page
  // never starts anything itself: a newly ready item waits to be asked for
  let running = $state<RunPlan | null>(null)
  // What the dialog would offer, ready and runnable failed items alike: the
  // header's button and its count come from the same plan the dialog opens on
  const offered = $derived(
    request?.state === 'open' && plan ? readyPlan(plan).candidates.length : 0,
  )
  /** Jobs were queued: show them at once, not at the next refresh - and
   * again for one still being sent as the dialog closed, which would
   * otherwise read as ready and invite a second run */
  function ran(queued: number, landing?: Promise<number>) {
    running = null
    if (queued) void load()
    void landing?.then((landed) => {
      if (landed > queued) void load()
    })
  }

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

  /** Go to a note: its item's row opened (the title toggles it, as a
   * press would), out of the dropped fold, and the note in view */
  async function showNote(key: string | null, id: string) {
    const title = key ? document.getElementById(titleId(key)) : null
    const fold = title?.closest('details')
    if (fold) fold.open = true
    if (title?.getAttribute('aria-expanded') === 'false') title.click()
    await tick()
    const note = document.getElementById(noteAnchor(id))
    if (!note) return
    note.scrollIntoView?.({ block: 'center' })
    note.focus({ preventScroll: true })
  }

  // Under 640px the rail's sections fold: what the page is for comes first.
  // The page's own width, not the viewport's - the sidebar takes its share
  let page = $state<HTMLElement>()
  let header = $state<ReturnType<typeof RequestHeader>>()
  let requestMenu = $state<ReturnType<typeof RequestMenu>>()
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

{#snippet changes(req: LedgerRequest, heading: boolean)}
  <PlanChanges
    history={req.history}
    revision={req.revision}
    {heading}
    diffs={looked.diffs}
    since={looked.since}
    ondismiss={() => (looked = seen.dismissed(ws, store.read, looked))}
  />
{/snippet}

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

{#if running && plan && request}
  <RunDialog
    run={running}
    {plan}
    workspace={ws}
    requestId={request.id}
    onclose={ran}
  />
{/if}

<div class="reqpage" bind:this={page}>
  {#if missing || error || request === null || plan === null}
    <!-- the loaded page's RequestHeader owns the header; until then, and when
         the load fails, this one names the page and goes back -->
    <PageHeader
      title="Request"
      secondary={requestId}
      back={{
        href: wsHref(ws, 'requests'),
        label: SECTION_LABELS.ws.requests,
      }}
    />
  {/if}
  {#if missing}
    <p>This request no longer exists. The agent may have removed it.</p>
  {:else if error}
    <ErrorLine error={reads.error} doing="load this request" />
    <button type="button" class="quiet" onclick={load}>Retry</button>
  {:else if request === null || plan === null}
    <Skeleton lines={4} label="loading request" />
  {:else}
    {#if stale}
      <div class="panel warn-edge stale" role="status">
        <span>
          Showing the plan as of <Time
            at={readAt}
            mode="exact"
          />{`. Couldn't refresh: ${stale.title}`}
        </span>
        <Details summary="Engine detail" detail={stale.detail ?? ''} />
        <button type="button" class="quiet" onclick={load}>Retry</button>
      </div>
    {/if}
    <div class="layout">
      <div class="maincol">
        <RequestHeader
          bind:this={header}
          {request}
          {plan}
          workspace={ws}
          onselect={goTo}
          onreload={load}
          {countdown}
          onrunheld={(keys) => (running = heldPlan(plan, keys))}
        >
          {#snippet actions()}
            {#if offered}
              <button
                type="button"
                title="Price and queue what can run now"
                onclick={() => (running = readyPlan(plan))}
              >
                Run {offered}…
              </button>
            {/if}
          {/snippet}
          {#snippet menu()}
            <RequestMenu
              bind:this={requestMenu}
              {request}
              {store}
              workspace={ws}
              onreload={load}
              ondeleted={() => goWs('requests')}
            />
          {/snippet}
        </RequestHeader>
        <CloseProposal
          {request}
          {store}
          onclose={(state) => requestMenu?.setState(state)}
          closing={requestMenu?.busy() ?? false}
        />
        <PlanItems
          {store}
          {request}
          {plan}
          workspace={ws}
          ongo={goTo}
          onreload={load}
          changed={reads.changed}
          onsay={(text) => (said = text)}
          onsee={seeChanges}
          onrun={(key) => (running = itemPlan(plan, key))}
          {countdown}
          onmove={(text, refs) => header?.takeNote(text, refs)}
          onreopen={() => requestMenu?.setState('open')}
          reopening={requestMenu?.busy() ?? false}
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
            {@render changes(request, false)}
          </details>
        {:else}
          {@render ask(request)}
          {@render changes(request, true)}
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .reqpage {
    container-type: inline-size;
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
