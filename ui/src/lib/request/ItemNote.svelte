<script lang="ts">
  import { tick, untrack, type Snippet } from 'svelte'
  import { X } from '@lucide/svelte'
  import { confirmDialog } from '../confirm.svelte'
  import { notify } from '../toast'
  import {
    draftAtRisk,
    keepDraft,
    readDraft,
    refusalText,
    saveFailure,
    type SaveFailure,
  } from './draft'
  import {
    LISTENING_LINE,
    NOT_LISTENING_LINE,
    type NoteLine,
  } from './noteState'
  import type { LedgerNote, LedgerNoteRef } from '../types'

  // One of the user's notes for the agent, on a request or one of its
  // items - or, with `note` null, the box a new one is written in. The
  // line under a saved note (noteState.ts) says only what the ledger can
  // show. The draft lives here, so a refresh that hands down the note
  // again never touches it - and, with a `draftKey`, in the tab's
  // sessionStorage too, so a reload does not lose it
  let {
    note,
    about,
    line,
    editable = true,
    draftKey = null,
    listening = false,
    refs = [],
    chip,
    onremoveref,
    extra,
    onsave,
    onsaved,
    ondelete,
    onreload,
  }: {
    /** The saved note; null for the box a new one is written in */
    note: LedgerNote | null
    /** What the note is on, for the box's label: an item's title, or
     * "this request" */
    about: string
    /** The line under the saved note; null when there is none */
    line: NoteLine | null
    /** False where the ledger would not keep an edit: a dropped item */
    editable?: boolean
    /** Where the unsaved draft is kept across a reload (draft.ts) */
    draftKey?: string | null
    /** Whether an agent session will hear the note now: the button says
     * "Send to agent", else "Save note". Saving works either way */
    listening?: boolean
    /** What the note is about: a saved note's, or a new one's so far */
    refs?: LedgerNoteRef[]
    /** How a reference is drawn */
    chip?: Snippet<[LedgerNoteRef]>
    /** A new note's reference taken off before it is saved */
    onremoveref?: (ref: LedgerNoteRef) => void
    /** More in the open box: the request's item picker */
    extra?: Snippet
    /** Store the text; rejects with the server's error */
    onsave: (text: string) => Promise<void>
    /** The save went through and the draft is emptied */
    onsaved?: () => void
    /** Remove the saved note; rejects with the server's error */
    ondelete?: () => Promise<void>
    /** Read the request again, after a save found it had moved on */
    onreload: () => void
  } = $props()

  const MAX = 4000
  const COUNT_FROM = 3600

  // A draft a reload left behind comes back as one set aside
  const left = untrack(() => {
    const text = readDraft(draftKey)
    return text !== null && draftAtRisk(text, note?.text ?? null) ? text : null
  })
  let editing = $state(false)
  let draft = $state(left ?? '')
  // Esc closed the box over words that are not saved: they are still here
  let kept = $state(left !== null)
  let saving = $state(false)
  // The server's answer to the last save, shown until the next one
  let refused = $state<SaveFailure | null>(null)
  let box = $state<HTMLTextAreaElement>()
  let opener = $state<HTMLButtonElement>()

  const saved = $derived(note?.text ?? null)

  // Words not yet saved are kept for the tab; saved or discarded,
  // forgotten - and a key that moved (a reference added or taken off)
  // leaves nothing behind under the old one
  let lastKey: string | null = null
  $effect(() => {
    const atRisk = (editing || kept) && draftAtRisk(draft, saved)
    if (lastKey && lastKey !== draftKey) keepDraft(lastKey, null)
    lastKey = draftKey
    keepDraft(draftKey, atRisk ? draft : null)
  })

  /** Open the box: the row's menu and "Note this take" ask for it too. An
   * open box keeps what is in it */
  export async function edit() {
    if (!editing) {
      if (!kept) draft = saved ?? ''
      kept = false
      refused = null
      editing = true
    }
    await tick()
    box?.focus()
  }
  async function shut() {
    editing = false
    await tick()
    opener?.focus()
  }
  /** Esc: out of the box, the draft kept for the next time it is opened */
  function setAside() {
    kept = draftAtRisk(draft, saved)
    void shut()
  }
  async function cancel() {
    if (
      draftAtRisk(draft, saved) &&
      !(await confirmDialog('Discard this draft of your note?', {
        confirmLabel: 'Discard',
        cancelLabel: 'Keep editing',
      }))
    )
      return
    kept = false
    void shut()
  }
  // The draft stays: it is saved again, against the request as it now is
  function reload() {
    refused = null
    onreload()
  }
  async function store(write: () => Promise<void>) {
    saving = true
    try {
      await write()
      kept = false
      refused = null
      draft = ''
      if (editing) void shut()
      if (note === null) onsaved?.()
    } catch (e) {
      refused = saveFailure(e)
      if (!refused) notify.error(e instanceof Error ? e.message : String(e))
    } finally {
      saving = false
    }
  }
  /** A saved note emptied is deleted; an empty new one is not sent */
  function save() {
    if (draft.trim()) return store(() => onsave(draft))
    if (note && ondelete) return store(ondelete)
    kept = false
    return shut()
  }
  async function remove() {
    if (
      !line?.answered &&
      !(await confirmDialog(
        'Delete this note? The agent has not had the chance to read it.',
        { confirmLabel: 'Delete' },
      ))
    )
      return
    if (ondelete) await store(ondelete)
  }
  function keys(event: KeyboardEvent) {
    if (event.key === 'Escape') {
      event.stopPropagation()
      setAside()
    } else if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault()
      void save()
    }
  }
</script>

{#snippet chips()}
  {#if refs.length}
    <ul class="refs" aria-label="What the note is about">
      {#each refs as ref (`${ref.item}/${ref.job ?? ''}/${ref.output ?? ''}`)}
        <li>
          {@render chip?.(ref)}
          {#if onremoveref}
            <button
              type="button"
              class="bare icon"
              aria-label="Remove {ref.output ?? ref.job ?? ref.item}"
              title="Remove from the note"
              onclick={() => onremoveref(ref)}
              ><X size={12} aria-hidden="true" /></button
            >
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
{/snippet}

{#if note || editable}
  <div class="note">
    {#if editing}
      {@render chips()}
      {@render extra?.()}
      <textarea
        bind:this={box}
        bind:value={draft}
        rows="3"
        maxlength={MAX}
        aria-label={note
          ? `Edit your note for the agent on ${about}`
          : `Note for the agent on ${about}`}
        placeholder="What should change?"
        onkeydown={keys}></textarea>
      <p class="hint muted">
        {listening ? LISTENING_LINE : NOT_LISTENING_LINE}
        {#if draft.length > COUNT_FROM}
          <span class="num">{draft.length} / {MAX}</span>
        {/if}
      </p>
      <div class="actions">
        <button
          type="button"
          class="quiet"
          onclick={save}
          disabled={saving}
          title="Ctrl+Enter"
          >{saving
            ? 'Saving…'
            : listening
              ? 'Send to agent'
              : 'Save note'}</button
        >
        <button type="button" class="bare" onclick={cancel} disabled={saving}
          >Cancel</button
        >
      </div>
    {:else}
      {#if note}
        <p class="text">{note.text}</p>
        {@render chips()}
        {#if line}<p class="hint muted">{line.text}</p>{/if}
      {:else}
        {@render chips()}
      {/if}
      {#if kept}<p class="hint">You have an unsaved draft.</p>{/if}
      {#if editable}
        <div class="actions">
          <button
            type="button"
            class="quiet"
            bind:this={opener}
            onclick={edit}
            disabled={saving}
            >{kept
              ? 'Continue the draft'
              : note
                ? 'Edit note'
                : 'Add a note for the agent'}</button
          >
          {#if note && ondelete}
            <button
              type="button"
              class="bare"
              onclick={remove}
              disabled={saving}>Delete note</button
            >
          {/if}
        </div>
      {/if}
    {/if}
    {#if refused}
      <p class="refused" role="alert">
        {refusalText(refused)}
        {#if refused.reload}
          <button type="button" class="quiet" onclick={reload}
            >Reload the plan</button
          >
        {/if}
      </p>
    {/if}
  </div>
{/if}

<style>
  .note {
    max-width: 68ch;
  }
  .text {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    margin: 0;
  }
  .hint {
    font-size: var(--t-sm);
    margin: var(--space-1) 0 0;
  }
  .refs {
    list-style: none;
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-2);
    margin: var(--space-1) 0;
    padding: 0;
  }
  .refs li {
    display: inline-flex;
    align-items: center;
    gap: 0.1rem;
  }
  .icon {
    padding: 0.1rem;
    color: var(--muted);
  }
  .actions {
    display: flex;
    gap: var(--space-2);
    margin-top: var(--space-2);
  }
  .refused {
    color: var(--bad);
    font-size: var(--t-sm);
    margin: var(--space-2) 0 0;
  }
  .refused button {
    margin-left: var(--space-2);
  }
</style>
