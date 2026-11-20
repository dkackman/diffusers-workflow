<script lang="ts">
  import { tick, type Snippet } from 'svelte'
  import {
    Check,
    ChevronDown,
    ChevronRight,
    MessageSquare,
  } from '@lucide/svelte'
  import AttemptStrip from './AttemptStrip.svelte'
  import ItemNote from './ItemNote.svelte'
  import ItemProof from './ItemProof.svelte'
  import {
    activeJob,
    CHIP,
    figureOf,
    opensByDefault,
    resultMark,
    takesLine,
    titleId,
    type ItemRef,
    type ItemState,
    type Take,
  } from './itemState'
  import { draftKey } from './draft'
  import { noteLine, noteWaiting } from './noteState'
  import { displayValue } from '../editor'
  import { wsHref } from '../routes'
  import type { LedgerRequest } from '../types'

  // One item of the plan as a row: what stands for it, what it is called,
  // its state as a word, one figure - and, opened, its takes, what it waits
  // on and what waits on it, what it would run, and the note for the agent.
  // The snippets are where a row's controls go; none is drawn here
  let {
    row,
    request,
    workspace,
    onsave,
    onreload,
    ongo,
    handle,
    action,
    menu,
    notice,
    onaccept,
  }: {
    row: ItemState
    request: LedgerRequest
    workspace: string
    /** Store the item's note (null clears it); rejects with the server's error */
    onsave: (text: string | null) => Promise<void>
    /** Read the request again, after a save found it had moved on */
    onreload: () => void
    /** Go to another item's row: a dependency's name was pressed */
    ongo: (key: string) => void
    /** The first cell: the reorder grip */
    handle?: Snippet<[ItemState]>
    /** Beside "Open job": the row's own action */
    action?: Snippet<[ItemState]>
    /** The last cell: the row's menu, with what it can ask of the row */
    menu?: Snippet<[ItemState, RowApi]>
    /** Under the row: what the page has to say about an edit to it */
    notice?: Snippet<[ItemState]>
    /** Accept a take as the result (null clears the choice) */
    onaccept?: (take: Take, job: string | null) => void
  } = $props()

  /** What a row's menu can ask of it */
  type RowApi = { note: () => void }

  const item = $derived(row.item)
  const chip = $derived(CHIP[row.display])
  const waitingNote = $derived(noteWaiting(item.comment, request))
  // The user's own toggle wins for as long as the page holds this row
  let chosen = $state<boolean | null>(null)
  const open = $derived(chosen ?? opensByDefault(row, waitingNote))

  const figure = $derived(figureOf(row))
  const takes = $derived(takesLine(row))
  const mark = $derived(row.display === 'done' ? resultMark(item) : null)
  const job = $derived(activeJob(item))
  const args = $derived(Object.entries(item.arguments ?? {}))
  const moreId = $derived(`req-item-${item.key}-more`)
  const firstLine = (text: string) => text.trim().split('\n')[0]

  /** The row stays open under the button just pressed: with a take
   * chosen it would no longer open by default */
  function accept(take: Take, job: string | null) {
    chosen = true
    onaccept?.(take, job)
  }

  let note = $state<ReturnType<typeof ItemNote>>()
  /** Open the row and its note box: "Leave a note" in the menu */
  async function openNote() {
    chosen = true
    await tick()
    void note?.edit()
  }
</script>

{#snippet refs(list: ItemRef[])}
  {#each list as dep, i (i)}
    {i ? ', ' : ''}<button
      type="button"
      class="bare ref"
      onclick={() => ongo(dep.key)}>{dep.title}</button
    ><span class:bad={dep.display === 'failed'}
      >{` (${CHIP[dep.display].word})`}</span
    >
  {/each}
{/snippet}

<li class="item" class:dropped={item.dropped} data-key={item.key}>
  <!-- The row alone is the live one: opened, its region is not machine state -->
  <div class="row" class:row-live={row.display === 'running'}>
    <span class="handle">{@render handle?.(row)}</span>
    <div class="main">
      <div class="titleline">
        <button
          type="button"
          class="bare title"
          id={titleId(item.key)}
          aria-expanded={open}
          aria-controls={moreId}
          onclick={() => (chosen = !open)}
        >
          {#if open}<ChevronDown size={14} aria-hidden="true" />{:else}
            <ChevronRight size={14} aria-hidden="true" />{/if}
          <span class="name">{item.title}</span>
        </button>
        <code class="muted key">{item.key}</code>
      </div>
      {#if row.display === 'waiting'}
        <p class="sub">waits on {@render refs(row.blockers)}</p>
      {:else if row.noWorkflow && !item.dropped}
        <p class="sub muted">
          No stored workflow. Only the agent can run this.
        </p>
      {/if}
      {#if takes}
        <p class="sub">
          <span class="muted">{takes}</span>
          {#if item.attempts.length > 1 && mark === 'accepted'}
            <span class="mark"
              ><Check size={12} aria-hidden="true" /> accepted</span
            >
          {:else if item.attempts.length > 1 && mark === 'latest'}
            <span class="mark">latest, none accepted</span>
          {/if}
        </p>
      {/if}
      {#if item.comment && !open}
        <p class="sub peek" title="your note for the agent">
          <MessageSquare size={12} aria-hidden="true" />
          <span>{firstLine(item.comment.text)}</span>
        </p>
      {/if}
    </div>
    <span class="proofcell" class:bare={!item.result?.output}>
      <ItemProof
        output={item.result?.output ?? null}
        state={row.display}
        {workspace}
        href={item.result ? wsHref(workspace, 'jobs', item.result.job) : null}
        alt="result of {item.title}"
      />
    </span>
    <span class="chipcell"
      ><span class="chip {chip.tone}">{chip.word}</span></span
    >
    <span class="figure num muted">{figure}</span>
    <span class="tail">
      {#if job || action}
        <span class="action">
          {#if job}
            <a class="plain quiet openjob" href={wsHref(workspace, 'jobs', job)}
              >Open job</a
            >
          {/if}
          {@render action?.(row)}
        </span>
      {/if}
      {#if menu}<span class="menu">{@render menu(row, { note: openNote })}</span
        >{/if}
    </span>
  </div>
  {#if notice}{@render notice(row)}{/if}

  <!-- Hidden, not removed: the note's draft is in here, and closing the
       row must not lose it -->
  <div class="more" id={moreId} hidden={!open}>
    {#if open && item.attempts.length}
      <AttemptStrip
        {item}
        {workspace}
        requestId={request.id}
        onaccept={item.dropped || !onaccept ? undefined : accept}
      />
    {/if}
    {#if row.after.length}
      <p class="sub">after: {@render refs(row.after)}</p>
    {/if}
    {#if row.neededBy.length}
      <p class="sub">needed by: {@render refs(row.neededBy)}</p>
    {/if}
    {#if item.workflow && args.length}
      <details class="runs">
        <summary
          >Runs <code>{item.workflow}</code> with {args.length}
          {args.length === 1 ? 'argument' : 'arguments'}</summary
        >
        <ul class="args">
          {#each args as [name, value] (name)}
            <li title={displayValue(value)}>{name} = {displayValue(value)}</li>
          {/each}
        </ul>
      </details>
    {:else if item.workflow}
      <p class="sub">Runs <code>{item.workflow}</code> with no arguments</p>
    {/if}
    <ItemNote
      bind:this={note}
      draftKey={draftKey(workspace, request.id, item.key)}
      comment={item.comment}
      about={item.title}
      line={item.comment ? noteLine(item.comment, request) : null}
      editable={!item.dropped}
      {onsave}
      {onreload}
    />
  </div>
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
  /* Where a dragged row would land (reorder.svelte.ts): ink, the
     pointer's place, as a drop target is */
  .item:global(.insert-before) {
    box-shadow: inset 0 2px 0 var(--ink);
  }
  .item:global(.insert-after) {
    box-shadow: inset 0 -2px 0 var(--ink);
  }
  .item:global(.dragging) {
    position: relative;
    z-index: 1;
    opacity: 0.85;
    background: var(--panel);
  }
  /* handle, proof, main, chip, figure, actions. The chip's column is
     fixed, so a change of status never moves what is beside it */
  .row {
    display: grid;
    grid-template-columns: var(--handle) 96px minmax(0, 1fr) 90px 7.5rem minmax(
        5.5rem,
        auto
      );
    grid-template-areas: 'handle proof main chip figure tail';
    gap: var(--space-3);
    align-items: center;
    padding: 0.55rem 0.8rem 0.55rem 0.2rem;
  }
  .handle {
    grid-area: handle;
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
  .peek span {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .ref {
    /* in the line, as a link in a sentence is: a long title wraps with it */
    display: inline;
    text-align: left;
    padding: 0;
    color: var(--ink);
    font-size: inherit;
    text-decoration: underline;
    text-underline-offset: 2px;
  }
  .bad {
    color: var(--bad);
  }
  .more {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    /* under the main column: past the handle and the proof */
    padding: 0 0.8rem var(--space-4)
      calc(0.2rem + var(--handle) + 96px + 2 * var(--space-3));
  }
  .more[hidden] {
    display: none;
  }
  .runs summary {
    font-size: var(--t-sm);
    cursor: pointer;
  }
  .args {
    list-style: none;
    margin: var(--space-1) 0 0;
    padding: 0;
    font-family: var(--font-mono);
    font-size: var(--t-xs);
  }
  .args li {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* Narrow: the chip and the title, then the proof and the figure, then
     the row's action across the width */
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
      text-align: left;
      align-self: center;
    }
    .more {
      padding-left: calc(0.2rem + var(--handle) + var(--space-2));
    }
  }
</style>
