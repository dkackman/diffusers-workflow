<script lang="ts">
  import { tick } from 'svelte'
  import { MessageSquare } from '@lucide/svelte'
  import { confirmDialog } from '../confirm.svelte'
  import { notify } from '../toast'
  import { draftAtRisk, refusalText, saveFailure } from './draft'
  import type { NoteLine } from './noteState'
  import type { LedgerComment } from '../types'

  // The user's one note for the agent, on a request or on one of its
  // items. Nothing is sent anywhere: the agent finds it the next time it
  // writes the plan, and the line under a saved note (noteState.ts) says
  // only what the ledger can show. The draft lives here, so a refresh that
  // hands down a new `comment` never touches it
  let {
    comment,
    about,
    line,
    editable = true,
    onsave,
    onreload,
  }: {
    comment: LedgerComment | null
    /** What the note is on, for the box's label: an item's title, or
     * "this request" */
    about: string
    /** The line under the saved note; null when there is no note */
    line: NoteLine | null
    /** False where the ledger would not keep an edit: a dropped item */
    editable?: boolean
    /** Store the note (null clears it); rejects with the server's error */
    onsave: (text: string | null) => Promise<void>
    /** Read the request again, after a save found it had moved on */
    onreload: () => void
  } = $props()

  const MAX = 4000
  const COUNT_FROM = 3600

  let editing = $state(false)
  let draft = $state('')
  // Esc closed the box over words that are not saved: they are still here
  let kept = $state(false)
  let saving = $state(false)
  // The server's answer to the last save, shown until the next one
  let refused = $state<{ message: string; reload: boolean } | null>(null)
  let box = $state<HTMLTextAreaElement>()
  let opener = $state<HTMLButtonElement>()

  const saved = $derived(comment?.text ?? null)

  async function edit() {
    if (!kept) draft = saved ?? ''
    kept = false
    refused = null
    editing = true
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
  async function store(text: string | null) {
    saving = true
    try {
      await onsave(text)
      kept = false
      refused = null
      if (editing) void shut()
    } catch (e) {
      refused = saveFailure(e)
      if (!refused) notify.error(e instanceof Error ? e.message : String(e))
    } finally {
      saving = false
    }
  }
  const save = () => store(draft.trim() ? draft : null)
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

{#if comment || editable}
  <div class="note">
    <p class="label">
      <MessageSquare size={13} aria-hidden="true" /> Note for the agent
    </p>
    {#if editing}
      <textarea
        bind:this={box}
        bind:value={draft}
        rows="3"
        maxlength={MAX}
        aria-label="Note for the agent on {about}"
        placeholder="What should change?"
        onkeydown={keys}></textarea>
      <p class="hint muted">
        Nothing is sent to the agent. It reads this the next time it updates the
        plan.
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
          title="Ctrl+Enter">{saving ? 'Saving…' : 'Save note'}</button
        >
        <button type="button" class="bare" onclick={cancel} disabled={saving}
          >Cancel</button
        >
      </div>
    {:else}
      {#if comment}
        <p class="text">{comment.text}</p>
        {#if line}<p class="hint muted">{line.text}</p>{/if}
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
              : comment
                ? 'Edit note'
                : 'Leave a note for the agent'}</button
          >
          {#if comment && line?.answered}
            <button
              type="button"
              class="bare"
              onclick={() => store(null)}
              disabled={saving}>Clear note</button
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
  .label {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    margin: 0 0 var(--space-1);
    font-size: var(--t-sm);
    font-weight: 600;
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
