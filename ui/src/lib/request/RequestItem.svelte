<script lang="ts">
  import { tick, untrack, type Snippet } from 'svelte'
  import {
    Check,
    ChevronDown,
    ChevronRight,
    MessageSquare,
  } from '@lucide/svelte'
  import ItemMore, { deps } from './ItemMore.svelte'
  import ItemProof from './ItemProof.svelte'
  import {
    activeJob,
    blockedLine,
    failedLine,
    CHIP,
    figureOf,
    moreId,
    opensByDefault,
    resultMark,
    revisedLine,
    shownTake,
    takesLine,
    titleId,
    type ItemState,
    type Take,
  } from './itemState'
  import { noteWaiting } from './noteState'
  import type { ReviewCountdown } from './review.svelte'
  import { chipText } from './reviewText'
  import { openInHash } from '../mediaViewer'
  import { wsHref } from '../routes'
  import type { LedgerNoteRef, LedgerRequest } from '../types'

  // One item of the plan as a row: what stands for it, what it is called,
  // its state as a word, one figure; opened, ItemMore under it. The
  // snippets are where a row's controls go; none is drawn here
  let {
    row,
    request,
    workspace,
    onreload,
    ongo,
    handle,
    action,
    menu,
    notice,
    onaccept,
    onkeep,
    countdown,
    onmove,
    changed = false,
  }: {
    row: ItemState
    request: LedgerRequest
    workspace: string
    /** Read the request again: a note was written, or a save was stale */
    onreload: () => void | Promise<void>
    /** Go to another item's row: a dependency's name was pressed */
    ongo: (key: string) => void
    /** The first cell: the reorder grip */
    handle?: Snippet<[ItemState]>
    /** Beside "Open job": the row's own action */
    action?: Snippet<[ItemState]>
    /** The last cell: the row's menu, with what it can ask of the row */
    menu?: Snippet<[ItemState, { note: () => void }]>
    /** Under the row: what the page has to say about an edit to it */
    notice?: Snippet<[ItemState]>
    /** Accept a take as the result (null clears the choice) */
    onaccept?: (take: Take, job: string | null) => void
    /** Keep a revised item's shown take as its result: accept it again */
    onkeep?: (take: Take) => void
    /** The page's review countdowns: what a held row's chip counts down */
    countdown?: ReviewCountdown
    /** "For the whole request instead" on this item's new note */
    onmove?: (text: string, refs: LedgerNoteRef[]) => void
    /** A quiet read changed this item: marked until CHANGED_MARK_MS */
    changed?: boolean
  } = $props()

  const item = $derived(row.item)
  const chip = $derived(CHIP[row.display])
  const waitingNote = $derived(
    item.notes.some((note) => noteWaiting(note, request)),
  )
  const latestNote = $derived(item.notes.at(-1) ?? null)
  // The user's own toggle wins for as long as the page holds this row
  // A deep link to one of its takes (`open=<key>/…`) opens the row
  let chosen = $state<boolean | null>(
    untrack(() => openInHash()?.startsWith(`${row.item.key}/`)) ? true : null,
  )
  const open = $derived(chosen ?? opensByDefault(row, waitingNote))

  const held = $derived(
    row.review ? chipText(row.review, countdown?.left(item.key) ?? null) : null,
  )
  const figure = $derived(figureOf(row))
  const takes = $derived(takesLine(row))
  const blocked = $derived(blockedLine(row, request.state === 'open'))
  const why = $derived(
    row.display === 'failed' || row.display === 'cancelled'
      ? failedLine(item)
      : null,
  )
  const mark = $derived(
    row.display === 'done' || row.display === 'revised'
      ? resultMark(item)
      : null,
  )
  const shown = $derived(shownTake(item))
  const revised = $derived(
    row.display === 'revised' ? revisedLine(item, shown) : null,
  )
  const job = $derived(activeJob(item))
  const under = $derived(open && item.attempts.length > 0)

  /** Open under the button just pressed: chosen, it would not by default */
  function accept(take: Take, job: string | null) {
    chosen = true
    onaccept?.(take, job)
  }

  let more = $state<ReturnType<typeof ItemMore>>()
  /** Open the row and its new note's box: "Add a note" in the menu */
  async function openNote() {
    chosen = true
    await tick()
    void more?.add()
  }
</script>

<li
  class="item"
  class:dropped={item.dropped}
  class:row-changed={changed}
  data-key={item.key}
>
  <!-- The row alone is the live one: opened, its region is not machine state -->
  <div
    data-testid="item-row"
    class="row"
    class:under
    class:row-live={row.display === 'running'}
  >
    <span class="handle">{@render handle?.(row)}</span>
    <div class="main">
      <div class="titleline">
        <button
          type="button"
          class="bare title"
          data-testid="item-title"
          id={titleId(item.key)}
          aria-expanded={open}
          aria-controls={moreId(item.key)}
          onclick={() => (chosen = !open)}
        >
          {#if open}<ChevronDown size={14} aria-hidden="true" />{:else}
            <ChevronRight size={14} aria-hidden="true" />{/if}
          <span class="name" data-testid="item-name">{item.title}</span>
        </button>
        <code class="muted key">{item.key}</code>
        {#if held}<span class="chip idle held">{held}</span>{/if}
      </div>
      {#if row.display === 'waiting'}
        <p class="sub">waits on {@render deps(row.blockers, ongo)}</p>
      {:else if blocked}
        <p class="sub">{blocked}</p>
      {:else if row.noWorkflow && !item.dropped}
        <p class="sub muted">
          No stored workflow. Only the agent can run this.
        </p>
      {/if}
      {#if why}<p class="sub bad why" title={why}>{why}</p>{/if}
      {#if revised}
        <p class="sub revised">
          <span>{revised}</span>
          {#if shown && onkeep && !item.dropped}
            <button
              type="button"
              class="quiet keep"
              onclick={() => onkeep(shown)}
            >
              Keep this take
            </button>
          {/if}
        </p>
      {/if}
      {#if takes}
        <p class="sub">
          <span class="muted">{takes}</span>
          {#if item.attempts.length > 1 && mark === 'accepted'}
            <span class="mark">
              <Check size={12} aria-hidden="true" /> accepted
            </span>
          {:else if item.attempts.length > 1 && mark === 'latest'}
            <span class="mark">latest, none accepted</span>
          {/if}
        </p>
      {/if}
      {#if latestNote && !open}
        <p class="sub peek" title="your latest note for the agent">
          <MessageSquare size={12} aria-hidden="true" />
          <span>{latestNote.text.trim().split('\n')[0]}</span>
          {#if item.notes.length > 1}
            <span class="more-notes">+{item.notes.length - 1}</span>
          {/if}
        </p>
      {/if}
    </div>
    <span
      data-testid="item-proof"
      class="proofcell"
      class:bare={!item.result?.output}
      class:under
    >
      <ItemProof
        output={item.result?.output ?? null}
        state={row.display}
        {workspace}
        href={item.result ? wsHref(workspace, 'jobs', item.result.job) : null}
        alt="result of {item.title}"
      />
    </span>
    <span class="chipcell">
      <span class="chip {chip.tone}" data-testid="item-chip">{chip.word}</span>
    </span>
    <span class="figure num muted">{figure}</span>
    <span class="tail">
      {#if job || action}
        <span class="action">
          {#if job}
            <a
              class="plain quiet openjob"
              href={wsHref(workspace, 'jobs', job)}
            >
              Open job
            </a>
          {/if}
          {@render action?.(row)}
        </span>
      {/if}
      {#if menu}<span class="menu">
          {@render menu(row, { note: openNote })}
        </span>{/if}
    </span>
  </div>
  {#if notice}{@render notice(row)}{/if}

  <ItemMore
    bind:this={more}
    {row}
    {request}
    {workspace}
    {open}
    {ongo}
    {onreload}
    onaccept={onaccept ? accept : undefined}
    {onmove}
    onnote={() => (chosen = true)}
  />
</li>

<style>
  .item {
    list-style: none;
    --handle: 24px;
  }
  /* A finger needs 44px of handle */
  @media (pointer: coarse) {
    .item {
      --handle: 44px;
    }
  }
  /* A fixed chip column: a change of status never moves its neighbours */
  .row {
    display: grid;
    grid-template-columns:
      var(--handle) var(--proof, 96px) minmax(0, 1fr)
      90px 7.5rem minmax(5.5rem, auto);
    grid-template-areas: 'handle proof main chip figure tail';
    gap: var(--space-3);
    align-items: center;
    padding: 0.55rem 0.8rem 0.55rem 0.2rem;
  }
  .handle {
    grid-area: handle;
  }
  /* Open with takes: no track, and the title starts at the grip */
  .row.under {
    --proof: 0px;
  }
  .proofcell.under {
    display: none;
  }
  .row.under .main {
    grid-column-start: proof;
  }
  .proofcell {
    grid-area: proof;
    display: flex;
  }
  .main {
    grid-area: main;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }
  .chipcell {
    grid-area: chip;
  }
  .figure {
    grid-area: figure;
    font-size: var(--t-xs);
    text-align: right;
  }
  .tail {
    grid-area: tail;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: var(--space-2);
  }
  .action,
  .menu {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .openjob {
    padding: 0.25rem 0.6rem;
    font-size: var(--t-sm);
    white-space: nowrap;
  }
  .titleline {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0 var(--space-2);
  }
  .title {
    display: inline-flex;
    align-items: baseline;
    gap: var(--space-1);
    padding: 0;
    color: var(--ink);
    font-size: var(--t-base);
    font-weight: 600;
    text-align: left;
    min-width: 0;
  }
  .title:hover {
    background: transparent;
    text-decoration: underline;
    text-underline-offset: 2px;
  }
  .title :global(svg) {
    flex: none;
    align-self: center;
    color: var(--muted);
  }
  .name {
    overflow-wrap: anywhere;
  }
  .key {
    font-size: var(--t-xs);
  }
  .held {
    white-space: nowrap;
  }
  .dropped {
    opacity: 0.72;
  }
  .dropped .name {
    text-decoration: line-through;
  }
  .sub {
    margin: 0;
    font-size: var(--t-sm);
  }
  .revised {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: var(--space-2);
  }
  .keep {
    padding: 0.1rem 0.5rem;
    font-size: var(--t-sm);
    white-space: nowrap;
  }
  .mark {
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
    font-weight: 600;
    margin-left: var(--space-2);
  }
  .peek {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    color: var(--muted);
    min-width: 0;
  }
  /* One line on any row; the whole reason is the title */
  .why {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .peek span:not(.more-notes) {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  /* Under 900px of panel: the chip and the figure under the title */
  @container (max-width: 900px) {
    .row {
      grid-template-columns:
        var(--handle) var(--proof, 96px) auto minmax(0, 1fr)
        minmax(5.5rem, auto);
      grid-template-areas:
        'handle proof main main tail'
        'handle proof chip figure tail';
      row-gap: var(--space-1);
    }
    .figure {
      text-align: left;
    }
  }

  /* Narrow: chip and title, proof and figure, then the action's width */
  @container (max-width: 640px) {
    .row {
      grid-template-columns: var(--handle) 96px minmax(0, 1fr) auto;
      grid-template-areas:
        'handle chip main menu'
        '. proof figure figure'
        '. action action action';
      gap: var(--space-2);
      align-items: start;
    }
    .tail {
      display: contents;
    }
    /* With no picture to show, the chip has said it: no empty slot */
    .proofcell.bare {
      display: none;
    }
    .action {
      grid-area: action;
    }
    .menu {
      grid-area: menu;
    }
    .openjob {
      flex: 1;
      text-align: center;
    }
    .figure {
      align-self: center;
    }
    .row.under .main {
      grid-column: main;
    }
  }
</style>
