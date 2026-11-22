<script lang="ts">
  import type { Snippet } from 'svelte'
  import NoteList from './NoteList.svelte'
  import RequestProgress from './RequestProgress.svelte'
  import StateTag from './StateTag.svelte'
  import { CHIP, type PlanState } from './itemState'
  import { listening } from './listening.svelte'
  import { notesWaiting } from './noteState'
  import type { ReviewCountdown } from './review.svelte'
  import { bannerText, holdGroup, runNowKeys } from './reviewText'
  import { agentSaid, standing } from './standing'
  import { formatMinutes, formatMtime, formatSpent } from '../format'
  import { requestQuery } from '../requestFilter'
  import { wsHref } from '../routes'
  import type { LedgerNoteRef, LedgerRequest } from '../types'

  // Where the ask stands, before anything else on the page: what is
  // happening, what wants the user, how far along it is and what it cost.
  // `actions` is the place of the page's one filled button
  let {
    request,
    plan,
    workspace,
    onselect,
    onreload,
    actions,
    menu,
    countdown,
    onrunheld,
  }: {
    request: LedgerRequest
    plan: PlanState
    workspace: string
    /** Go to an item's row: a progress segment was pressed */
    onselect: (key: string) => void
    /** Read the request again: a note was written */
    onreload: () => void | Promise<void>
    actions?: Snippet
    /** Beside the state: the request's own menu */
    menu?: Snippet
    /** The page's review countdowns, anchored to its reads */
    countdown?: ReviewCountdown
    /** Open the run dialog for these held items: "Run now…" */
    onrunheld?: (keys: string[]) => void
  } = $props()

  let list = $state<ReturnType<typeof NoteList>>()
  let box = $state<HTMLElement>()
  /** Take an item's draft into the request's note box, about the item,
   * and bring the box into view */
  export async function takeNote(text: string, refs: LedgerNoteRef[]) {
    await list?.add({ text, refs })
    box?.scrollIntoView?.({ block: 'nearest' })
  }

  const stand = $derived(standing(request, plan))
  // The agent's latest word: where an answer to a note, or "ready, not
  // run", is said, so the page does not hide it in Plan changes
  const said = $derived(agentSaid(request))
  const waiting = $derived(notesWaiting(request))
  const segments = $derived(
    plan.live.map((state) => ({
      key: state.item.key,
      title: state.item.title,
      display: state.display,
      word: CHIP[state.display].word,
    })),
  )
  const open = $derived(request.state === 'open')

  // The review banner: one for the request, worded by its most urgent
  // hold and by whether a session listens. Its digits tick, so it is not a
  // live region: the page announces a change of its state once
  const channel = $derived(listening(workspace))
  const hold = $derived(holdGroup(plan.held))
  const counting = $derived(
    plan.held.some((r) => r.review?.state === 'holding'),
  )
  $effect(() => {
    if (counting && countdown) return countdown.hold()
  })
  const left = $derived(
    hold
      ? (countdown?.left(hold.first.item.key) ??
          hold.first.review?.seconds_left ??
          0)
      : 0,
  )
  const banner = $derived(
    hold ? bannerText(hold, channel.listening, left) : null,
  )
  const runNow = $derived(hold ? runNowKeys(hold, channel.listening) : [])
</script>

<header class="reqhead">
  <div class="titleline">
    <h1>{request.title}</h1>
    <StateTag value={request.state} />
    {#if menu}<span class="menu">{@render menu()}</span>{/if}
  </div>
  <p class="id"><code class="muted">{request.id}</code></p>

  <p class="standing">{stand.sentence}</p>
  {#if stand.needs}<p class="needs">{stand.needs}</p>{/if}
  {#if said}
    <p class="said" title={said.why}>
      <span class="who">
        The agent, r{said.revision} · {formatMtime(Date.parse(said.at) / 1000)}:
      </span>
      {said.why}
    </p>
  {/if}
  {#if segments.length}
    <RequestProgress {segments} {onselect} />
  {/if}
  <p class="figures num">
    <span>{formatSpent(request.spent_minutes)} spent</span>
    {#if open && request.remaining_minutes > 0}
      <span>{formatMinutes(request.remaining_minutes)} left</span>
    {/if}
    {#if waiting}
      <span>{waiting} {waiting === 1 ? 'note' : 'notes'} waiting</span>
    {/if}
  </p>

  <div class="actions">
    {#if actions}<div class="primary">{@render actions()}</div>{/if}
    <a
      class="plain quiet"
      href={wsHref(workspace, 'jobs') + requestQuery(request.id)}
    >
      its jobs
    </a>
    <a
      class="plain quiet"
      href={wsHref(workspace, 'gallery') + requestQuery(request.id)}
    >
      its outputs
    </a>
  </div>

  {#if banner}
    <div class="panel hold" data-state={hold?.state}>
      <p>{banner}</p>
      {#if runNow.length && onrunheld}
        <button type="button" class="quiet" onclick={() => onrunheld(runNow)}>
          Run now…
        </button>
      {/if}
    </div>
  {/if}

  <div class="notebox" bind:this={box}>
    <NoteList
      bind:this={list}
      notes={request.notes}
      {request}
      item={null}
      {workspace}
      about="this request"
      pickItems
      ongo={onselect}
      onchanged={onreload}
    />
  </div>
</header>

<style>
  /* scrollIntoView lands below the sticky app header */
  .notebox {
    scroll-margin-top: 70px;
  }
  .reqhead {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    margin-bottom: var(--space-4);
  }
  .titleline {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
  }
  /* A request's title is written for a person, so it is not in the mono
     face the engine's names take */
  h1 {
    font-family: var(--font-sans);
    font-weight: 600;
    letter-spacing: 0;
    line-height: 1.2;
    min-width: 0;
    overflow-wrap: anywhere;
  }
  .menu {
    margin-left: auto;
    align-self: center;
  }
  p {
    margin: 0;
  }
  .id {
    font-size: var(--t-xs);
    margin-top: calc(-1 * var(--space-1));
  }
  .standing {
    font-size: var(--t-md);
    line-height: 1.35;
    margin-top: var(--space-2);
  }
  .needs {
    font-weight: 600;
  }
  /* The agent's latest word, three lines at most; Plan changes has it whole */
  .said {
    font-size: var(--t-sm);
    max-width: 72ch;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .said .who {
    color: var(--muted);
  }
  .figures {
    display: flex;
    flex-wrap: wrap;
    gap: 0 var(--space-4);
    font-size: var(--t-sm);
    color: var(--muted);
  }
  .hold {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2) var(--space-4);
    padding: 0.55rem 0.8rem;
    font-size: var(--t-sm);
  }
  .hold p {
    flex: 1 1 20rem;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: var(--space-2);
  }
  .actions a {
    padding: 0.25rem 0.6rem;
    font-size: var(--t-sm);
  }
  .primary {
    order: 1;
    margin-left: auto;
  }
  @container (max-width: 640px) {
    .primary {
      order: 0;
      flex-basis: 100%;
      margin-left: 0;
    }
    .primary :global(button) {
      width: 100%;
    }
  }
</style>
