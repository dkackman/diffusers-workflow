<script lang="ts" module>
  import { CHIP, type ItemRef } from './itemState'
  // Other items named in a sentence, each a way to its row with its state
  // in words: the row's "waits on" and the lines here
  export { deps }
</script>

<script lang="ts">
  import AttemptStrip from './AttemptStrip.svelte'
  import NoteList from './NoteList.svelte'
  import { moreId, type ItemState, type Take } from './itemState'
  import { addRef } from './noteRefs'
  import { displayValue } from '../editor'
  import type { LedgerNoteRef, LedgerRequest } from '../types'

  // A row's open half: its takes, what it follows and what needs it, the
  // notes for the agent, then what it runs. Hidden, never removed, while
  // the row is closed: the notes' drafts are in here
  let {
    row,
    request,
    workspace,
    open,
    ongo,
    onreload,
    onaccept,
    onmove,
    pending = $bindable([]),
  }: {
    row: ItemState
    request: LedgerRequest
    workspace: string
    open: boolean
    ongo: (key: string) => void
    onreload: () => void | Promise<void>
    /** Accept a take; absent where no take can be accepted */
    onaccept?: (take: Take, job: string | null) => void
    /** "For the whole request instead" on this item's new note */
    onmove?: (text: string, refs: LedgerNoteRef[]) => void
    /** What the new note is about so far: the takes "Note this take" added */
    pending?: LedgerNoteRef[]
  } = $props()

  const item = $derived(row.item)
  const args = $derived(Object.entries(item.arguments ?? {}))
  // Open with takes, the row's picture steps aside: this starts at the grip
  const under = $derived(open && item.attempts.length > 0)

  let notes = $state<ReturnType<typeof NoteList>>()
  /** Open the new note's box and focus it, seeded with a draft and refs */
  export function add(seed?: { text: string; refs: LedgerNoteRef[] }) {
    return notes?.add(seed)
  }
  /** "Note this take": the take, and its proof file, join the new note */
  function noteTake(take: Take) {
    const { job, output } = take.attempt
    pending = addRef(pending, {
      item: item.key,
      job,
      ...(output ? { output: output.name } : {}),
    })
    void add()
  }
</script>

{#snippet deps(list: ItemRef[], go: (key: string) => void)}
  {#each list as dep, i (i)}
    {i ? ', ' : ''}<button
      type="button"
      class="bare ref"
      onclick={() => go(dep.key)}
    >
      {dep.title}
    </button><span class:bad={dep.display === 'failed'}>
      {` (${CHIP[dep.display].word})`}
    </span>
  {/each}
{/snippet}

<div class="more" class:under id={moreId(item.key)} hidden={!open}>
  {#if open && item.attempts.length}
    <AttemptStrip
      {item}
      {workspace}
      onaccept={item.dropped ? undefined : onaccept}
      onnote={item.dropped || request.state !== 'open' ? undefined : noteTake}
    />
  {/if}
  {#if row.after.length}
    <p class="sub">after: {@render deps(row.after, ongo)}</p>
  {/if}
  {#if row.neededBy.length}
    <p class="sub">needed by: {@render deps(row.neededBy, ongo)}</p>
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
  <!-- What it runs is detail: after the notes, folded -->
  {#if item.workflow}
    <details class="runs">
      <summary>
        Runs <code>{item.workflow}</code>{args.length
          ? ` with ${args.length} ${args.length === 1 ? 'argument' : 'arguments'}`
          : ''}
      </summary>
      {#if args.length}
        <ul class="args">
          {#each args as [name, value] (name)}
            <li title={displayValue(value)}>{name} = {displayValue(value)}</li>
          {/each}
        </ul>
      {:else}
        <p class="sub muted">No arguments.</p>
      {/if}
    </details>
  {/if}
</div>

<style>
  .more {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    /* under the main column: past the handle and the proof (--handle is
       the row's) */
    padding: 0 0.8rem var(--space-4)
      calc(0.2rem + var(--handle) + 96px + 2 * var(--space-3));
  }
  .more.under {
    padding-left: calc(0.2rem + var(--handle) + var(--space-3));
  }
  .more[hidden] {
    display: none;
  }
  .sub {
    margin: 0;
    font-size: var(--t-sm);
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
  @container (max-width: 640px) {
    .more,
    .more.under {
      padding-left: calc(0.2rem + var(--handle) + var(--space-2));
    }
  }
</style>
