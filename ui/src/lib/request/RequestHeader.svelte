<script lang="ts">
  import type { Snippet } from 'svelte'
  import NoteList from './NoteList.svelte'
  import RequestProgress from './RequestProgress.svelte'
  import StateTag from './StateTag.svelte'
  import { CHIP, type PlanState } from './itemState'
  import { notesWaiting } from './noteState'
  import { standing } from './standing'
  import { formatMinutes, formatSpent } from '../format'
  import { requestQuery } from '../requestFilter'
  import { wsHref } from '../routes'
  import type { LedgerRequest } from '../types'

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
  } = $props()

  const stand = $derived(standing(request, plan))
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
      href={wsHref(workspace, 'jobs') + requestQuery(request.id)}>its jobs</a
    >
    <a
      class="plain quiet"
      href={wsHref(workspace, 'gallery') + requestQuery(request.id)}
      >its outputs</a
    >
  </div>

  <NoteList
    notes={request.notes}
    {request}
    item={null}
    {workspace}
    about="this request"
    pickItems
    ongo={onselect}
    onchanged={onreload}
  />
</header>

<style>
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
  .figures {
    display: flex;
    flex-wrap: wrap;
    gap: 0 var(--space-4);
    font-size: var(--t-sm);
    color: var(--muted);
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
