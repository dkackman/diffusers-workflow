<script lang="ts">
  import { tick, type Snippet } from 'svelte'
  import {
    Check,
    ChevronDown,
    ChevronRight,
    MessageSquare,
  } from '@lucide/svelte'
  import AttemptStrip from './AttemptStrip.svelte'
  import ItemProof from './ItemProof.svelte'
  import NoteList from './NoteList.svelte'
  import {
    activeJob,
    blockedLine,
    CHIP,
    figureOf,
    opensByDefault,
    resultMark,
    revisedLine,
    shownTake,
    takesLine,
    titleId,
    type ItemRef,
    type ItemState,
    type Take,
  } from './itemState'
  import { noteWaiting } from './noteState'
  import type { ReviewCountdown } from './review.svelte'
  import { chipText } from './reviewText'
  import { addRef } from './noteRefs'
  import { displayValue } from '../editor'
  import { wsHref } from '../routes'
  import type { LedgerNoteRef, LedgerRequest } from '../types'

  // One item of the plan as a row: what stands for it, what it is called,
  // its state as a word, one figure - and, opened, its takes, what it waits
  // on and what waits on it, what it would run, and the notes for the agent.
  // The snippets are where a row's controls go; none is drawn here
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
    /** Read the request again: a note was written, or a save found the
     * request had moved on */
    onreload: () => void | Promise<void>
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
    /** Keep a revised item's shown take as its result: accept it again */
    onkeep?: (take: Take) => void
    /** The page's review countdowns: what a held row's chip counts down */
    countdown?: ReviewCountdown
    /** "For the whole request instead" on this item's new note */
    onmove?: (text: string, refs: LedgerNoteRef[]) => void
    /** A quiet read changed this item: marked until CHANGED_MARK_MS */
    changed?: boolean
  } = $props()

  /** What a row's menu can ask of it */
  type RowApi = { note: () => void }

  const item = $derived(row.item)
  const chip = $derived(CHIP[row.display])
  const waitingNote = $derived(
    item.notes.some((note) => noteWaiting(note, request)),
  )
  const latestNote = $derived(item.notes.at(-1) ?? null)
  // The user's own toggle wins for as long as the page holds this row
  let chosen = $state<boolean | null>(null)
  const open = $derived(chosen ?? opensByDefault(row, waitingNote))

  const held = $derived(
    row.review ? chipText(row.review, countdown?.left(item.key) ?? null) : null,
  )
  const figure = $derived(figureOf(row))
  const takes = $derived(takesLine(row))
  const blocked = $derived(blockedLine(row))
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
  const args = $derived(Object.entries(item.arguments ?? {}))
  const moreId = $derived(`req-item-${item.key}-more`)
  const firstLine = (text: string) => text.trim().split('\n')[0]

  /** The row stays open under the button just pressed: with a take
   * chosen it would no longer open by default */
  function accept(take: Take, job: string | null) {
    chosen = true
    onaccept?.(take, job)
  }

  let notes = $state<ReturnType<typeof NoteList>>()
  // What the new note is about so far: the takes "Note this take" added
  let pending = $state<LedgerNoteRef[]>([])
  /** Open the row and its new note's box: "Add a note" in the menu */
  async function openNote() {
    chosen = true
    await tick()
    void notes?.add()
  }
  /** "Note this take": the take, and its proof file, join the new note */
  function noteTake(take: Take) {
    const { job, output } = take.attempt
    pending = addRef(pending, {
      item: item.key,
      job,
      ...(output ? { output: output.name } : {}),
    })
    void openNote()
  }
</script>

{#snippet refs(list: ItemRef[])}
  {#each list as dep, i (i)}
    {i ? ', ' : ''}<button
      type="button"
      class="bare ref"
      onclick={() => ongo(dep.key)}
    >
      {dep.title}
    </button><span class:bad={dep.display === 'failed'}>
      {` (${CHIP[dep.display].word})`}
    </span>
  {/each}
{/snippet}

<li
  class="item"
  class:dropped={item.dropped}
  class:row-changed={changed}
  data-key={item.key}
>
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
        {#if held}<span class="chip idle held">{held}</span>{/if}
      </div>
      {#if row.display === 'waiting'}
        <p class="sub">waits on {@render refs(row.blockers)}</p>
      {:else if blocked}
        <p class="sub">{blocked}</p>
      {:else if row.noWorkflow && !item.dropped}
        <p class="sub muted">
          No stored workflow. Only the agent can run this.
        </p>
      {/if}
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
          <span>{firstLine(latestNote.text)}</span>
          {#if item.notes.length > 1}
            <span class="more-notes">+{item.notes.length - 1}</span>
          {/if}
        </p>
      {/if}
    </div>
    <!-- Open, the takes below show the same picture larger: the cell keeps
      its width so the row does not move -->
    <span
      class="proofcell"
      class:bare={!item.result?.output}
      class:under={open && item.attempts.length > 0}
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
      <span class="chip {chip.tone}">{chip.word}</span>
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

  <!-- Hidden, not removed: the notes' drafts are in here, and closing the
       row must not lose them -->
  <div class="more" id={moreId} hidden={!open}>
    {#if open && item.attempts.length}
      <AttemptStrip
        {item}
        {workspace}
        onaccept={item.dropped || !onaccept ? undefined : accept}
        onnote={item.dropped ? undefined : noteTake}
      />
    {/if}
    {#if row.after.length}
      <p class="sub">after: {@render refs(row.after)}</p>
    {/if}
    {#if row.neededBy.length}
      <p class="sub">needed by: {@render refs(row.neededBy)}</p>
    {/if}
    <NoteList
      bind:this={notes}
      bind:pendingRefs={pending}
      notes={item.notes}
      {request}
      item={item.key}
      {workspace}
      about={item.title}
      {ongo}
      {onmove}
      onchanged={onreload}
    />
    <!-- What it runs is detail: after the notes, not between the takes and
      them -->
    {#if item.workflow && args.length}
      <details class="runs">
        <summary>
          Runs <code>{item.workflow}</code> with {args.length}
          {args.length === 1 ? 'argument' : 'arguments'}
        </summary>
        <ul class="args">
          {#each args as [name, value] (name)}
            <li title={displayValue(value)}>{name} = {displayValue(value)}</li>
          {/each}
        </ul>
      </details>
    {:else if item.workflow}
      <p class="sub">Runs <code>{item.workflow}</code> with no arguments</p>
    {/if}
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
  .proofcell.under {
    visibility: hidden;
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
  .peek .more-notes {
    flex: none;
    overflow: visible;
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
