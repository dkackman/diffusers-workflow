<script lang="ts">
  import { tick } from 'svelte'
  import { GripVertical } from '@lucide/svelte'
  import { action } from '../action.svelte'
  import { api } from '../api'
  import { ApiError } from '../apiErrors'
  import { confirmDialog } from '../confirm.svelte'
  import { undoToast } from '../destructive'
  import { requestQuery } from '../requestFilter'
  import { wsHref } from '../routes'
  import { notify } from '../toast'
  import type { LedgerNoteRef, LedgerRequest } from '../types'
  import ConflictNotice from './ConflictNotice.svelte'
  import { dropWarnings, restoredText } from './dropConfirm'
  import ItemMenu from './ItemMenu.svelte'
  import { itemMenu } from './menuEntries'
  import {
    nextTake,
    titleId,
    type ItemState,
    type PlanState,
    type Take,
  } from './itemState'
  import {
    onThisRequest,
    type Outcome,
    type PlanStore,
  } from './planStore.svelte'
  import { handleId, Reorder } from './reorder.svelte'
  import {
    laterIn,
    moveTarget,
    placeBefore,
    runsAfter,
    titlesOf,
    type Direction,
    type Move,
  } from './reorderMath'
  import RequestItem from './RequestItem.svelte'
  import type { ReviewCountdown } from './review.svelte'

  // The plan as a list the user can edit: each row with its handle, its
  // menu and its one action, and - where an edit met the agent's - what
  // became of it. Every edit goes to the store as an intent; nothing here
  // changes the plan it is shown
  let {
    store,
    request,
    plan,
    workspace,
    ongo,
    onreload,
    onsay,
    onsee,
    onrun,
    countdown,
    onmove,
    changed,
    onreopen,
    reopening = false,
  }: {
    store: PlanStore
    /** The request as the page shows it: the store's view */
    request: LedgerRequest
    plan: PlanState
    workspace: string
    /** Go to an item's row */
    ongo: (key: string) => void
    /** Read the request again: a note was written, or a save found it
     * had moved on */
    onreload: () => void | Promise<void>
    /** Tell a screen reader what an edit did */
    onsay: (text: string) => void
    /** Show a write among the plan's changes */
    onsee: (revision: number) => void
    /** Open the priced run dialog for one item; none, and no row offers it */
    onrun?: (key: string) => void
    /** The page's review countdowns, for a held row's chip */
    countdown?: ReviewCountdown
    /** An item's note moved to the request's own box */
    onmove?: (text: string, refs: LedgerNoteRef[]) => void
    /** The items a quiet read changed, by key: marked (`.row-changed`) */
    changed?: ReadonlySet<string>
    /** Reopen a closed request: the header menu's own Reopen */
    onreopen?: () => void
    /** A Reopen is in flight: the button waits */
    reopening?: boolean
  } = $props()

  // A closed request is read-only: no grip, Accept, row action or note box
  const open = $derived(request.state === 'open')

  let list = $state<HTMLElement>()
  const keys = $derived(plan.live.map((row) => row.item.key))
  const total = $derived(plan.live.length)

  /** A write that failed for a reason that is not the plan moving */
  function failed(error: unknown) {
    const stale = error instanceof ApiError && error.code === 'stale_revision'
    if (stale) {
      notify.error('The plan kept changing, so that was not saved. Try again.')
    } else {
      notify.failure('save that', error)
    }
  }
  const quietly = (write: Promise<Outcome>) => void write.catch(failed)

  let toasts = 0
  /** An edit shown at once, with the toast that takes it back - once, and
   * only on this request. The toast goes if the edit was not written after
   * all: held back to ask, no longer applying, or refused */
  function undoable(
    write: Promise<Outcome>,
    message: string,
    back: () => void,
  ) {
    const id = `request-undo-${++toasts}`
    undoToast(message, back, onThisRequest(store), id)
    write
      .then((outcome) => {
        if (outcome.status !== 'saved') notify.dismiss(id)
      })
      .catch((error) => {
        notify.dismiss(id)
        failed(error)
      })
  }

  // Rows placed above something they depend on: the dependency wins
  const later = $derived(open ? runsAfter(plan) : {})

  function moved(row: ItemState, move: Move) {
    quietly(
      store.stage({ kind: 'move', key: row.item.key, before: move.before }),
    )
    const next = placeBefore(keys, row.item.key, move.before)
    const wins = laterIn(next, row.item.key, row.after)
    onsay(
      `${row.item.title} moved to position ${move.position} of ${move.total}.` +
        (wins.length ? ` It still runs after ${titlesOf(wins)}.` : ''),
    )
  }
  const byKey = (key: string) => plan.live.find((row) => row.item.key === key)
  const reorder = new Reorder({
    keys: () => keys,
    list: () => list,
    onmove: (key, move) => {
      const row = byKey(key)
      if (row) moved(row, move)
    },
    oncommit: () => store.commit(),
  })
  $effect(() => () => reorder.destroy())

  // The star is the accept's own: Undo takes back the acceptance only, and
  // a star that fails leaves the acceptance (notify.failure says so). One
  // action per accept: a shared one would drop a second accept's star while
  // the first is on its way, under a toast that says it was starred
  const star = (name: string) =>
    action(() => api.setStarred(name, true), {
      doing: 'star the accepted take',
    }).run()

  function accept(row: ItemState, take: Take, job: string | null) {
    const { key, title } = row.item
    const previous = row.item.accepted?.job ?? null
    const file = take.attempt.output?.name
    const write = store.push({ kind: 'accept', key, job })
    const starred = job !== null && file ? ' and starred it' : ''
    undoable(
      write,
      job === null
        ? `Cleared the accepted take of ${title}`
        : `Accepted take ${take.n} for ${title}${starred}`,
      () => quietly(store.push({ kind: 'accept', key, job: previous })),
    )
    if (starred && file)
      void write
        .then((outcome) => {
          if (outcome.status === 'saved') void star(file)
        })
        .catch(() => undefined)
  }

  /** "Keep this take": the revised item's shown take accepted again, which
   * the server records as kept against the plan as it now stands */
  function keep(row: ItemState, take: Take) {
    const { key, title } = row.item
    quietly(store.push({ kind: 'accept', key, job: take.attempt.job }))
    onsay(`Kept take ${take.n} for ${title}.`)
  }

  async function drop(row: ItemState) {
    const { key, title } = row.item
    const warnings = dropWarnings(row)
    if (
      warnings.length &&
      !(await confirmDialog(warnings.join('\n\n'), { confirmLabel: 'Drop' }))
    )
      return
    const at = keys.indexOf(key)
    const next = keys[at + 1] ?? keys[at - 1]
    undoable(store.drop(key), `Dropped ${title}`, () =>
      quietly(store.restore(key)),
    )
    await tick()
    if (next) document.getElementById(titleId(next))?.focus()
  }

  async function restore(row: ItemState) {
    const { key, title } = row.item
    const remembered = key in store.memory
    try {
      const outcome = await store.restore(key)
      if (outcome.status === 'saved' && !remembered)
        notify.success(restoredText(title, outcome.stripped))
    } catch (error) {
      failed(error)
    }
  }

  function select(row: ItemState, id: string, note: () => void) {
    if (id === 'note') return note()
    if (id === 'drop') return void drop(row)
    if (id === 'retake') return onrun?.(row.item.key)
    const move = moveTarget(keys, row.item.key, id as Direction)
    if (!move) return
    moved(row, move)
    store.commit()
  }

  const entries = (row: ItemState) =>
    itemMenu(row, {
      total,
      jobsHref:
        wsHref(workspace, 'jobs') + requestQuery(request.id, row.item.key),
      outputsHref:
        wsHref(workspace, 'gallery') + requestQuery(request.id, row.item.key),
      canRun: !!onrun,
      open,
    })

  const shown = $derived(new Set(plan.items.map((row) => row.item.key)))
  /** Cards for edits whose item is no longer on the page */
  const orphans = $derived(
    store.conflicts.filter((c) => c.key === null || !shown.has(c.key)),
  )
  const answer = (id: number) => quietly(store.again(id))
</script>

{#snippet handle(row: ItemState)}
  <button
    type="button"
    class="bare grip"
    id={handleId(row.item.key)}
    aria-label="Reorder {row.item.title}, position {row.position} of {total}"
    title="Drag to reorder. Ready items run in this order."
    onkeydown={(event) => reorder.keydown(event, row.item.key)}
    onblur={reorder.blur}
    onpointerdown={(event) => reorder.pointerdown(event, row.item.key)}
    onpointermove={reorder.pointermove}
    onpointerup={reorder.pointerup}
    onpointercancel={reorder.cancel}
  >
    <GripVertical size={16} aria-hidden="true" />
  </button>
{/snippet}

{#snippet rowAction(row: ItemState)}
  {#if row.item.dropped}
    <button type="button" class="quiet act" onclick={() => restore(row)}>
      Restore
    </button>
  {:else if onrun && row.runnable && row.display === 'revised'}
    <button type="button" class="quiet act" onclick={() => onrun(row.item.key)}>
      Run take {nextTake(row.item)}…
    </button>
  {:else if onrun && row.runnable}
    <button type="button" class="quiet act" onclick={() => onrun(row.item.key)}>
      {row.display === 'ready' ? 'Run…' : 'Run again…'}
    </button>
  {:else if onrun && row.canRetake}
    <!-- In plain view, not only in the menu: after a note the next take is
      the usual next step -->
    <button
      type="button"
      class="quiet act"
      aria-label="Run another take of {row.item.title}"
      onclick={() => onrun(row.item.key)}
    >
      Another take…
    </button>
  {/if}
{/snippet}

{#snippet menu(row: ItemState, api: { note: () => void })}
  {@const list = entries(row)}
  {#if list.length}
    <ItemMenu
      label="More actions for {row.item.title}"
      entries={list}
      onselect={(id) => select(row, id, api.note)}
    />
  {/if}
{/snippet}

{#snippet notice(row: ItemState)}
  {#if later[row.item.key]}
    <p class="onrow hint muted">
      Runs after {titlesOf(later[row.item.key])} whatever its place.
    </p>
  {/if}
  {#each store.conflicts.filter((c) => c.key === row.item.key) as conflict (conflict.id)}
    <div class="onrow">
      <ConflictNotice
        {conflict}
        onagain={() => answer(conflict.id)}
        ondismiss={() => store.dismiss(conflict.id)}
      />
    </div>
  {/each}
{/snippet}

{#snippet item(row: ItemState)}
  <RequestItem
    {row}
    {request}
    {workspace}
    {onreload}
    {ongo}
    handle={row.item.dropped || !open ? undefined : handle}
    action={open ? rowAction : undefined}
    {menu}
    {notice}
    onaccept={open ? (take, job) => accept(row, take, job) : undefined}
    onkeep={open ? (take) => keep(row, take) : undefined}
    {countdown}
    {onmove}
    changed={changed?.has(row.item.key) ?? false}
  />
{/snippet}

{#if store.notice}
  <div class="above">
    <ConflictNotice
      applied={store.notice.entry}
      ondismiss={() => (store.notice = null)}
      {onsee}
    />
  </div>
{/if}
{#each orphans as conflict (conflict.id)}
  <div class="above">
    <ConflictNotice {conflict} ondismiss={() => store.dismiss(conflict.id)} />
  </div>
{/each}

{#if !open}
  <div class="panel closed">
    <p role="note">
      This request is {request.state}. Reopen it to change the plan or leave a
      note.
    </p>
    {#if onreopen}
      <button type="button" onclick={onreopen} disabled={reopening}>
        {reopening ? 'Reopening…' : 'Reopen'}
      </button>
    {/if}
  </div>
{/if}

{#if plan.items.length === 0}
  <p class="muted">The agent has not planned any items for this request yet.</p>
{:else}
  <div class="panel list">
    <ol class="items" aria-label="The plan" bind:this={list}>
      {#each plan.live as row (row.item.key)}
        {@render item(row)}
      {/each}
    </ol>
    {#if plan.dropped.length}
      <details class="droppedgroup">
        <summary>Dropped ({plan.dropped.length})</summary>
        <ul class="items">
          {#each plan.dropped as row (row.item.key)}
            {@render item(row)}
          {/each}
        </ul>
      </details>
    {/if}
  </div>
{/if}

<style>
  .above {
    margin-bottom: var(--space-3);
  }
  .closed {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-3);
    margin: 0 0 var(--space-3);
  }
  .closed p {
    margin: 0;
  }
  .hint {
    margin: 0;
    font-size: var(--t-sm);
  }
  .onrow {
    padding: 0 0.8rem var(--space-3) calc(0.2rem + 24px + var(--space-3));
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
  /* A row being dragged, and where it would land: the reorder sets these
     on the row's <li>. Ink, the pointer's place, as a drop target is */
  .items > :global(li.insert-before) {
    box-shadow: inset 0 2px 0 var(--ink);
  }
  .items > :global(li.insert-after) {
    box-shadow: inset 0 -2px 0 var(--ink);
  }
  .items > :global(li.dragging) {
    position: relative;
    z-index: 1;
    opacity: 0.85;
    background: var(--panel);
  }
  .droppedgroup {
    border-top: 1px solid var(--line);
  }
  summary {
    cursor: pointer;
    padding: 0.55rem 0.8rem;
    font-size: var(--t-sm);
    font-weight: 600;
    color: var(--muted);
  }
  .grip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    padding: 0.3rem 0;
    color: var(--muted);
    cursor: grab;
    /* the drag is the page's, not a scroll */
    touch-action: none;
  }
  .grip:active {
    cursor: grabbing;
  }
  .act {
    padding: 0.25rem 0.6rem;
    font-size: var(--t-sm);
    white-space: nowrap;
  }
  @media (pointer: coarse) {
    .grip {
      min-height: 44px;
    }
    .act {
      min-height: 44px;
    }
  }
  @container (max-width: 640px) {
    .act {
      flex: 1;
    }
  }
</style>
