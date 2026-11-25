<script lang="ts">
  import { onMount, tick, type Snippet } from 'svelte'
  import { X } from '@lucide/svelte'
  import { confirmDialog } from '../confirm.svelte'
  import { destroyConfirm } from '../destructive'
  import { notify } from '../toast'
  import { refusalText, saveFailure, type SaveFailure } from './draft'
  import { NOTE_MAX, NoteDraft } from '../noteDraft.svelte'
  import {
    LISTENING_LINE,
    NEEDS_WORDS,
    NOT_LISTENING_LINE,
    type NoteLine,
  } from './noteState'
  import type { LedgerNote, LedgerNoteRef } from '../types'
  import Time from '../ui/Time.svelte'
  // Between the note's two times; spelled here, as a block trims a space
  const BETWEEN = ' · '

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
    answer = null,
    editable = true,
    draftKey = null,
    listening = false,
    stops = 0,
    refs = [],
    chip,
    onremoveref,
    extra,
    onsave,
    onsaved,
    ondelete,
    onreload,
    onmove,
    onopen,
  }: {
    /** The saved note; null for the box a new one is written in */
    note: LedgerNote | null
    /** What the note is on, for the box's label: an item's title, or
     * "this request" */
    about: string
    /** The line under the saved note; null when there is none */
    line: NoteLine | null
    /** What the agent said in its first write after the saved note
     * (answerOf); null while it has said nothing */
    answer?: { revision: number; why: string } | null
    /** False where the ledger would not keep an edit: a dropped item */
    editable?: boolean
    /** Where the unsaved draft is kept across a reload (draft.ts) */
    draftKey?: string | null
    /** Whether an agent session will hear the note now: the button says
     * "Send to agent", else "Save note". Saving works either way */
    listening?: boolean
    /** How many held items saving a note would stop from starting on
     * their own (holdsStopped); the open box warns while it is above 0 */
    stops?: number
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
    /** An item's new-note box only: hand the draft to the request's own
     * note box ("For the whole request instead") */
    onmove?: (text: string) => void
    /** The box was opened: a row keeps itself open under it */
    onopen?: () => void
  } = $props()

  const unreadNote = destroyConfirm({
    name: 'this note',
    noun: 'note',
    consequence: 'The agent has not had the chance to read it.',
    place: 'ledger',
  })

  const saved = $derived(note?.text ?? null)
  // The draft, kept for the tab under `draftKey`; a key that moved (a
  // reference added or taken off) carries it and leaves nothing behind
  const draft: NoteDraft = new NoteDraft({
    key: () => draftKey,
    saved: () => saved ?? '',
    save: async () => {
      await save()
    },
    onclose: () => void focusOpener(),
    rekey: 'move',
  })
  let saving = $state(false)
  // The server's answer to the last save, shown until the next one
  let refused = $state<SaveFailure | null>(null)
  let box = $state<HTMLTextAreaElement>()
  let opener = $state<HTMLButtonElement>()

  /** Open the box: the row's menu and "Note this take" ask for it too. An
   * open box keeps what is in it */
  export async function edit(text?: string) {
    onopen?.()
    if (!draft.editing) {
      draft.open()
      refused = null
    }
    // Moved-in words follow a draft already there, after a blank line
    if (text)
      draft.text = draft.text.trim()
        ? `${draft.text.trimEnd()}\n\n${text}`
        : text
    await tick()
    box?.focus()
  }
  // A draft restored from the tab's storage: its row opens with it, so a
  // read cannot close the row over it
  onMount(() => {
    if (draft.kept) onopen?.()
  })
  async function focusOpener() {
    await tick()
    opener?.focus()
  }
  async function cancel() {
    if (await draft.cancel()) void focusOpener()
  }
  // The draft stays: it is saved again, against the request as it now is
  function reload() {
    refused = null
    onreload()
  }
  /** Typing went on while a new note was out: the box keeps only the new
   * words, and closes when only blank ones came after */
  function keepTyped(sent: string) {
    if (!draft.text.startsWith(sent)) return
    draft.text = draft.text.slice(sent.length).trim()
    if (!draft.text) draft.done('')
  }
  /** Send `sent` (the box as it was at the press) through `write` */
  async function store(write: () => Promise<void>, sent: string) {
    // Ctrl+Enter is not a disabled button: a second press waits for none
    if (saving) return
    saving = true
    try {
      await write()
      const open = draft.editing
      refused = null
      // Words typed while it was out stay in the box
      if (draft.done(sent)) {
        if (open) void focusOpener()
      } else if (note === null) keepTyped(sent)
      if (note === null) onsaved?.()
    } catch (e) {
      refused = saveFailure(e)
      if (!refused) notify.failure('save the note', e)
    } finally {
      saving = false
    }
  }
  // A new note goes out only with words: references alone are not a note
  const wordless = $derived(note === null && !draft.text.trim())
  /** A saved note emptied is deleted, asking first as "Delete note" does;
   * a new one with no words is not sent, and the box stays open */
  function save(): Promise<void> | undefined {
    const sent = draft.text
    if (sent.trim()) return store(() => onsave(sent), sent)
    if (note && ondelete) return remove(sent)
  }
  async function remove(sent: string) {
    // The agent's answer line is what says it read the note
    if (
      !answer &&
      !line?.answered &&
      !(await confirmDialog(unreadNote.message, unreadNote))
    )
      return
    if (ondelete) await store(ondelete, sent)
  }
  function move() {
    if (!onmove || wordless) return
    onmove(draft.text)
    draft.done(draft.text)
    void focusOpener()
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
            >
              <X size={12} aria-hidden="true" />
            </button>
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
{/snippet}

{#if note || editable}
  <div class="note">
    {#if draft.editing}
      <textarea
        bind:this={box}
        bind:value={draft.text}
        rows="3"
        maxlength={NOTE_MAX}
        aria-label={note
          ? `Edit your note for the agent on ${about}`
          : `Note for the agent on ${about}`}
        placeholder="What should the agent change?"
        onkeydown={(e) => draft.keydown(e)}></textarea>
      {#if refs.length || extra}
        <div class="about">
          <span class="aboutlabel">About</span>
          {#if refs.length}
            {@render chips()}
          {:else}
            <span class="muted">{about}</span>
          {/if}
          {@render extra?.()}
        </div>
      {/if}
      {#if wordless && refs.length}
        <p class="hint">{NEEDS_WORDS}</p>
      {/if}
      {#if stops > 0}
        <p class="hint" role="alert">
          Saving this note stops {stops === 1 ? '1 item' : `${stops} items`}
          waiting for your OK from starting on their own.
        </p>
      {/if}
      <p class="hint muted">
        {listening ? LISTENING_LINE : NOT_LISTENING_LINE}
        {#if draft.counter}
          <span class="num">{draft.counter}</span>
        {/if}
      </p>
      <div class="actions">
        <button
          type="button"
          class="quiet send"
          onclick={save}
          disabled={saving || wordless}
          title="Ctrl+Enter"
        >
          {saving ? 'Saving…' : listening ? 'Send to agent' : 'Save note'}
        </button>
        <button type="button" class="bare" onclick={cancel} disabled={saving}>
          Cancel
        </button>
        {#if onmove && note === null}
          <button
            type="button"
            class="bare"
            onclick={move}
            disabled={saving || wordless}
          >
            For the whole request instead
          </button>
        {/if}
      </div>
    {:else}
      <!-- A new note's references show only in its open box: closed, a
        chip would read as a note already sent -->
      {#if note}
        <p class="text">{note.text}</p>
        {#if refs.length}
          <div class="about">
            <span class="aboutlabel">About</span>
            {@render chips()}
          </div>
        {/if}
        {#if answer}
          <p class="hint">
            The agent answered in r{answer.revision}: {answer.why}
          </p>
        {/if}
        {#if line}
          <!-- One sentence, kept on one line: a line break here is a space -->
          <!-- prettier-ignore -->
          <p class="hint muted noteline"><Time at={line.written} label="written" />{#if line.changed !== null}{BETWEEN}<Time at={line.changed} label="changed" />{/if}{`. ${line.standing}`}{#if line.agent}{` (${line.agent.lead}`}<Time at={line.agent.at} label="changed" by="agent" />).{/if}</p>
        {/if}
      {/if}
      {#if draft.kept}<p class="hint">You have an unsaved draft.</p>{/if}
      {#if editable}
        <div class="actions">
          <button
            type="button"
            class="quiet"
            bind:this={opener}
            onclick={() => edit()}
            disabled={saving}
          >
            {draft.kept
              ? 'Continue the draft'
              : note
                ? 'Edit note'
                : 'Add a note for the agent'}
          </button>
          {#if note && ondelete}
            <button
              type="button"
              class="bare"
              onclick={() => remove(draft.text)}
              disabled={saving}
              title={line?.answered
                ? 'The agent has answered; remove it from the list'
                : undefined}
            >
              {line?.answered ? 'Clear note' : 'Delete note'}
            </button>
          {/if}
        </div>
      {/if}
    {/if}
    {#if refused}
      <p class="refused" role="alert">
        {refusalText(refused)}
        {#if refused.reload}
          <button type="button" class="quiet" onclick={reload}>
            Reload the plan
          </button>
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
    margin: var(--space-1) 0 0;
  }
  .about {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1) var(--space-2);
    margin: var(--space-1) 0;
    font-size: var(--t-sm);
  }
  .aboutlabel {
    color: var(--muted);
    font-size: var(--t-xs);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .refs {
    list-style: none;
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-2);
    margin: 0;
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
  /* Outlined like every button but the page's one filled one, and at full
     strength once there is something to send, so ready and not-yet differ
     by more than the disabled fade */
  .send:not(:disabled) {
    color: var(--ink);
    border-color: var(--muted);
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
