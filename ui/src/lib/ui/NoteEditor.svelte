<script lang="ts">
  import { Pencil } from '@lucide/svelte'
  import { setNote } from '../api'

  // A note on a workspace, asset, run or file: shown as plain text, edited
  // in place. The server owns the 4000-character limit and says so in its
  // error, which stays on screen beside the box
  let {
    note,
    target,
    label = 'Note',
    placeholder = 'Add a note',
    onsaved,
  }: {
    note: string | null | undefined
    target: string
    label?: string
    placeholder?: string
    onsaved?: (note: string | null) => void
  } = $props()

  let editing = $state(false)
  let draft = $state('')
  let saving = $state(false)
  let error = $state<string | null>(null)
  // What the box last stored, until the parent hands down a newer note
  let shown = $state<string | null>(null)
  $effect(() => {
    // Another target is another note, even when the text is the same
    void target
    shown = note ?? null
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
      const stored = await setNote(target, draft.trim() ? draft : null)
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
        <p class="text">{shown}</p>
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
    white-space: pre-wrap;
    overflow-wrap: anywhere;
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
