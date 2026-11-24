<script lang="ts">
  import { MessageSquare } from '@lucide/svelte'
  import { api } from '../api'
  import type { LedgerNote, LedgerNoteRef, LedgerRequest } from '../types'
  import ItemNote from './ItemNote.svelte'
  import NoteRef from './NoteRef.svelte'
  import { draftKey, refKey } from './draft'
  import { listening } from './listening.svelte'
  import { answerOf, noteAnchor, noteLine } from './noteState'
  import { holdsStopped } from './reviewText'
  import { addRef, collapseRefs, writtenRefs } from './noteRefs'
  import { action } from '../action.svelte'

  // The user's notes for the agent on a request or one of its items: each
  // with where it stands and what it is about, then the box a new one is
  // written in. Every write goes to the request-notes routes from here,
  // then `onchanged` (the page reads the plan again). The box's references
  // so far are `pendingRefs`: a page adds a take to them, the user takes
  // one off
  let {
    notes,
    request,
    item,
    workspace,
    about,
    filter,
    pendingRefs = $bindable([]),
    pickItems = false,
    ongo,
    onadded,
    onmove,
    onopen,
    onchanged,
  }: {
    notes: LedgerNote[]
    request: LedgerRequest
    /** The item the notes are on; null for the request's own */
    item: string | null
    workspace: string
    /** What the notes are on, in words: an item's title, "this request" */
    about: string
    /** Show only the notes this keeps */
    filter?: (note: LedgerNote) => boolean
    /** What the new note is about so far */
    pendingRefs?: LedgerNoteRef[]
    /** The request's own box: pick any of the plan's live items */
    pickItems?: boolean
    /** Go to an item's row, from an item's chip */
    ongo?: (key: string) => void
    /** A new note was stored and its box is empty again; the box's
     * references have been emptied just before */
    onadded?: (note: LedgerNote) => void
    /** An item's box offers "For the whole request instead": the draft and
     * what it is about, for the page to take to the request's own box */
    onmove?: (text: string, refs: LedgerNoteRef[]) => void
    /** A box here was opened to write in */
    onopen?: () => void
    onchanged: () => void | Promise<void>
  } = $props()

  const channel = $derived(listening(workspace))
  $effect(() => channel.hold())

  const shown = $derived(filter ? notes.filter(filter) : notes)
  // A dropped item, or a closed request, keeps its notes to read; the
  // ledger takes no more until it is reopened
  const editable = $derived(
    request.state === 'open' &&
      (item === null ||
        request.items.some((i) => i.key === item && !i.dropped)),
  )
  const live = $derived(request.items.filter((i) => !i.dropped))
  // What saving a note here would stop: the request's note every hold, an
  // item's note that item's own
  const stops = $derived(holdsStopped(request, item))
  // The picker offers the items the new note is not yet about
  const unpicked = $derived(
    live.filter(
      (i) => !pendingRefs.some((r) => r.item === i.key && !r.job && !r.output),
    ),
  )

  let fresh = $state<ReturnType<typeof ItemNote>>()
  /** Open the new note's box */
  export function add(seed?: { text: string; refs: LedgerNoteRef[] }) {
    if (seed)
      for (const ref of seed.refs) pendingRefs = addRef(pendingRefs, ref)
    return fresh?.edit(seed?.text)
  }
  function move(text: string) {
    const refs = writtenRefs(pendingRefs)
    onmove?.(text, refs.length ? refs : [{ item: item as string }])
    pendingRefs = []
  }

  // The stored note, held until the box has emptied its draft: the
  // references go then, not before, so a sent draft is never kept under
  // the key of a box with none
  let stored: LedgerNote | null = null
  // Each box's writes are an action, so a second Save while one is on its
  // way sends nothing. A refusal is not the action's to toast: it goes back
  // to the box, which says it beside the draft (ItemNote's store)
  type Write =
    | { add: string }
    | { edit: LedgerNote; text: string }
    | { remove: LedgerNote }
  // Keyed by box (a note's id, '' for the new one); nothing renders it
  const boxes: Record<string, ReturnType<typeof boxWriter>> = {}
  function boxWriter() {
    return action(
      async (w: Write): Promise<{ refused: unknown }> => {
        try {
          if ('add' in w) await added(w.add)
          else if ('edit' in w) await edited(w.edit, w.text)
          else await deleted(w.remove)
          return { refused: null }
        } catch (refused) {
          return { refused }
        }
      },
      { doing: 'save the note' },
    )
  }
  async function write(box: string, w: Write) {
    boxes[box] ??= boxWriter()
    const outcome = await boxes[box].run(w)
    // Undefined while another write from this box is out: the action's
    // guard, the only one here
    if (outcome?.refused) throw outcome.refused
  }
  async function added(text: string) {
    await channel.refresh()
    const refs = writtenRefs(pendingRefs)
    const written = await api.addNote(workspace, {
      request: request.id,
      ...(item === null ? {} : { item }),
      text,
      ...(refs.length ? { refs } : {}),
    })
    stored = written.note
    await onchanged()
  }
  function sent() {
    pendingRefs = []
    if (stored) onadded?.(stored)
    stored = null
  }
  async function edited(note: LedgerNote, text: string) {
    await channel.refresh()
    await api.editNote(workspace, { request: request.id, note: note.id, text })
    await onchanged()
  }
  async function deleted(note: LedgerNote) {
    await api.deleteNote(workspace, request.id, note.id)
    await onchanged()
  }
  const unref = (ref: LedgerNoteRef) =>
    (pendingRefs = pendingRefs.filter((r) => r !== ref))
  function pick(event: Event & { currentTarget: HTMLSelectElement }) {
    const key = event.currentTarget.value
    if (key) pendingRefs = addRef(pendingRefs, { item: key })
    event.currentTarget.value = ''
  }
</script>

{#snippet chip(ref: LedgerNoteRef)}
  <NoteRef {ref} {request} {workspace} {ongo} />
{/snippet}

{#snippet picker()}
  {#if unpicked.length}
    <select
      class="picker"
      aria-label="Add an item this note is about"
      onchange={pick}
    >
      <option value="">+ Add an item</option>
      {#each unpicked as each (each.key)}
        <option value={each.key}>{each.key} · {each.title}</option>
      {/each}
    </select>
  {/if}
{/snippet}

{#if shown.length || editable}
  <section class="notes" aria-label="Notes for the agent on {about}">
    <!-- With none saved, the box's button says it: no heading over it -->
    {#if shown.length}
      <p class="label">
        <MessageSquare size={13} aria-hidden="true" /> Note for the agent
      </p>
      <ul class="list">
        {#each shown as note (note.id)}
          <li id={noteAnchor(note.id)} tabindex="-1">
            <ItemNote
              {note}
              {about}
              line={noteLine(note, request, new Date(), channel.listening)}
              answer={answerOf(note, request)}
              {editable}
              draftKey={draftKey(workspace, request.id, item, `@${note.id}`)}
              listening={channel.listening}
              {stops}
              refs={collapseRefs(note.refs ?? [])}
              {chip}
              onsave={(text) => write(note.id, { edit: note, text })}
              ondelete={() => write(note.id, { remove: note })}
              onreload={onchanged}
              {onopen}
            />
          </li>
        {/each}
      </ul>
    {/if}
    {#if editable}
      <ItemNote
        bind:this={fresh}
        note={null}
        {about}
        line={null}
        draftKey={draftKey(workspace, request.id, item, refKey(pendingRefs))}
        listening={channel.listening}
        {stops}
        refs={pendingRefs}
        {chip}
        onremoveref={unref}
        extra={pickItems && live.length ? picker : undefined}
        onsave={(text) => write('', { add: text })}
        onsaved={sent}
        onmove={item !== null && onmove ? move : undefined}
        onreload={onchanged}
        {onopen}
      />
    {/if}
  </section>
{/if}

<style>
  .notes {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    max-width: 68ch;
  }
  .label {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    margin: 0;
    font-size: var(--t-sm);
    font-weight: 600;
  }
  .list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .list li:focus {
    outline: 2px solid var(--ink);
    outline-offset: 2px;
  }
  .picker {
    width: auto;
    max-width: 32ch;
    margin: 0;
    padding: 0.05rem 0.4rem;
    font-size: var(--t-xs);
  }
</style>
