<script lang="ts">
  import { untrack } from 'svelte'
  import { poll } from '../poll'
  import type { LedgerNote, LedgerNoteRef } from '../types'
  import NoteList from './NoteList.svelte'
  import type { LinkedRequest } from './linkedRequest.svelte'
  import { refKey } from './draft'
  import { noteWaiting } from './noteState'
  import { IDLE_MS } from './refresh'

  // The notes for the agent on one item, away from the request page: the
  // job page's and the gallery file's box. A new note is about `refs`
  // unless the page says otherwise (`noteAbout`), and is again once one is
  // added. A note added here is read back through `linked`; while any note
  // on the request waits, the request is read again now and then, so the
  // page sees the agent answer it
  let {
    linked,
    item,
    filter,
    refs,
  }: {
    linked: LinkedRequest
    item: string
    /** The notes this page is about */
    filter: (note: LedgerNote) => boolean
    /** What a new note is about, unless the page says otherwise */
    refs: LedgerNoteRef[]
  } = $props()

  let pendingRefs = $state<LedgerNoteRef[]>([])
  // Only a different page's references start the box over: an equal list
  // handed down again keeps what the user has added or taken off
  const page = $derived(refKey(refs))
  $effect(() => {
    void page
    pendingRefs = untrack(() => [...refs])
  })

  const request = $derived(linked.request)
  const home = $derived(request?.items.find((i) => i.key === item) ?? null)
  const waiting = $derived(
    !!request &&
      [...request.notes, ...request.items.flatMap((i) => i.notes)].some(
        (note) => noteWaiting(note, request),
      ),
  )
  $effect(() => {
    if (!waiting) return
    const from = linked
    return poll(() => from.read(), IDLE_MS, { immediate: false })
  })

  // NoteList empties the references once a note is sent; the next one is
  // about this page again, whether or not the read after it came back
  const added = () => (pendingRefs = [...refs])
  const changed = () => linked.read()

  let list = $state<ReturnType<typeof NoteList>>()
  let box = $state<HTMLElement>()
  /** Open the new note's box about `about`, and bring it into view */
  export async function noteAbout(about: LedgerNoteRef[]) {
    pendingRefs = about
    await list?.add()
    box?.scrollIntoView?.({ block: 'nearest' })
  }
</script>

{#if request && home}
  <div class="agentnotes panel" bind:this={box}>
    <NoteList
      bind:this={list}
      notes={home.notes}
      {request}
      {item}
      workspace={linked.workspace}
      about={home.title}
      {filter}
      bind:pendingRefs
      onadded={added}
      onchanged={changed}
    />
  </div>
{/if}

<style>
  /* Set apart from the run's and the file's own note: this one goes to the
     agent working the request */
  .agentnotes {
    border-left: 3px solid var(--ink);
    scroll-margin-top: 70px;
    margin-bottom: 1rem;
  }
</style>
