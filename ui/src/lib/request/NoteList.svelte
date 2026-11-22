<script lang="ts">
  import { MessageSquare } from '@lucide/svelte'
  import { api } from '../api'
  import type { LedgerNote, LedgerNoteRef, LedgerRequest } from '../types'
  import ItemNote from './ItemNote.svelte'
  import NoteRef from './NoteRef.svelte'
  import { draftKey, refKey } from './draft'
  import { listening } from './listening.svelte'
  import { noteAnchor, noteLine } from './noteState'
  import { addRef, collapseRefs, writtenRefs } from './noteRefs'

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
    onchanged: () => void | Promise<void>
  } = $props()

  const channel = $derived(listening(workspace))
  $effect(() => channel.hold())

  const shown = $derived(filter ? notes.filter(filter) : notes)
  // A dropped item keeps its notes to read; the ledger takes no more
  const editable = $derived(
    item === null || request.items.some((i) => i.key === item && !i.dropped),
  )
  const live = $derived(request.items.filter((i) => !i.dropped))
  // The picker offers the items the new note is not yet about
  const unpicked = $derived(
    live.filter(
      (i) => !pendingRefs.some((r) => r.item === i.key && !r.job && !r.output),
    ),
  )

  let fresh = $state<ReturnType<typeof ItemNote>>()
  /** Open the new note's box */
  export function add() {
    return fresh?.edit()
  }

  // The stored note, held until the box has emptied its draft: the
  // references go then, not before, so a sent draft is never kept under
  // the key of a box with none
  let stored: LedgerNote | null = null
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
    <p class="label">
      <MessageSquare size={13} aria-hidden="true" /> Note for the agent
    </p>
    {#if shown.length}
      <ul class="list">
        {#each shown as note (note.id)}
          <li id={noteAnchor(note.id)} tabindex="-1">
            <ItemNote
              {note}
              {about}
              line={noteLine(note, request, new Date(), channel.listening)}
              {editable}
              draftKey={draftKey(workspace, request.id, item, `@${note.id}`)}
              listening={channel.listening}
              refs={collapseRefs(note.refs ?? [])}
              {chip}
              onsave={(text) => edited(note, text)}
              ondelete={() => deleted(note)}
              onreload={onchanged}
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
        refs={pendingRefs}
        {chip}
        onremoveref={unref}
        extra={pickItems && live.length ? picker : undefined}
        onsave={added}
        onsaved={sent}
        onreload={onchanged}
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
