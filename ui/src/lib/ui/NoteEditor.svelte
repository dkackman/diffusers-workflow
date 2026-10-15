<script lang="ts">
  import { Pencil } from '@lucide/svelte'
  import { setNote } from '../api'
  import ClampedText from './ClampedText.svelte'

  // A note on a workspace, asset, run or file: shown as a safe Markdown
  // subset (never HTML), edited in place. The server owns the 4000-character
  // limit and says so in its error, which stays on screen beside the box
  let {
    note,
    target,
    label = 'Note',
    placeholder = 'Add a note',
    workspace,
    lines = 4,
    onsaved,
  }: {
    note: string | null | undefined
    target: string
    /** The workspace a run or file target is read in; the picker's when
     * omitted */
    workspace?: string
    label?: string
    placeholder?: string
    /** Lines shown before the note folds behind "Show more" */
    lines?: number
    onsaved?: (note: string | null) => void
  } = $props()

  let editing = $state(false)
  let draft = $state('')
  let saving = $state(false)
  let error = $state<string | null>(null)
  // What the box last stored, until the parent hands down a newer note
  let shown = $derived(note ?? null)
  // Another target is another note, even when the text is the same. Only a
  // changed target ends an edit: a refetch landing mid-edit keeps the draft
  // svelte-ignore state_referenced_locally
  let editedTarget = target
  $effect(() => {
    if (target === editedTarget) return
    editedTarget = target
    editing = false
  })

  function edit() {
    draft = shown ?? ''
    error = null
    editing = true
  }

  async function save() {
    saving = true
    error = null
    try {
      const stored = await setNote(
        target,
        draft.trim() ? draft : null,
        workspace,
      )
      shown = stored.note ?? null
      editing = false
      onsaved?.(shown)
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    } finally {
      saving = false
    }
  }
</script>

<div class="note">
  {#if editing}
    <textarea
      bind:value={draft}
      maxlength="4000"
      rows="4"
      aria-label={label}
      {placeholder}></textarea>
    <p class="hint muted">
      Markdown: # heading · - list · **bold** · *italic* · `code` ·
      [link](https://…)
    </p>
    <div class="actions">
      <button type="button" class="quiet" onclick={save} disabled={saving}
        >Save</button
      >
      <button
        type="button"
        class="bare"
        onclick={() => (editing = false)}
        disabled={saving}>Cancel</button
      >
    </div>
    {#if error}<p class="error" role="alert">{error}</p>{/if}
  {:else}
    <div class="shown">
      <span class="label muted">{label}</span>
      {#if shown}
        <div class="text">
          <ClampedText
            text={shown}
            {lines}
            markdown
            rememberKey={`${workspace ?? ''}|${target}`}
          />
        </div>
      {:else}
        <p class="text muted">{placeholder}</p>
      {/if}
      <button
        type="button"
        class="quiet icon"
        aria-label="Edit note"
        title="edit this note"
        onclick={edit}><Pencil size={13} /></button
      >
    </div>
  {/if}
</div>

<style>
  .shown {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.5rem;
  }
  .label {
    font-size: var(--t-sm);
  }
  .text {
    min-width: 0;
    margin: 0;
  }
  .muted {
    color: var(--muted);
  }
  textarea {
    width: 100%;
    font: inherit;
    background: var(--panel-2);
    color: var(--ink);
  }
  .hint {
    font-size: var(--t-sm);
    margin: 0.25rem 0 0;
  }
  .actions {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.25rem;
  }
  .error {
    color: var(--bad);
    margin: 0.25rem 0 0;
  }
</style>
