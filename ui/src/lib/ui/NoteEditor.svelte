<script lang="ts">
  import ErrorLine from './ErrorLine.svelte'
  import { Pencil } from '@lucide/svelte'
  import { action } from '../action.svelte'
  import { api, setNote } from '../api'
  import { ApiError } from '../apiErrors'
  import { confirmDialog } from '../confirm.svelte'
  import { NOTE_MAX, NoteDraft, noteKey } from '../noteDraft.svelte'
  import { draftAtRisk } from '../request/draft'
  import { workspace as picker } from '../workspace.svelte'
  import ClampedText from './ClampedText.svelte'

  // A note on a workspace, asset, run or file: shown as a safe Markdown
  // subset (never HTML), edited in place. The server owns the 4000-character
  // limit and says so in its error, which stays on screen beside the box.
  // The draft is a NoteDraft (src/lib/noteDraft.svelte.ts), kept for the tab
  // so a remount or reload does not lose it. A newer note handed down
  // mid-edit (an agent, another tab), or found when a kept draft is
  // resumed over it, never replaces the draft: a notice shows it until the
  // user picks Use theirs or Keep mine. The write sends
  // `If-Match` with the etag of the note the user last saw (GET
  // /api/notes), so one that changed since is refused, read and shown in
  // that notice - never overwritten unseen. A read that fails writes
  // nothing: without the etag a save could not tell a newer note was there
  let {
    note,
    target,
    label = 'Note',
    placeholder = 'Add a note',
    workspace,
    lines = 4,
    editing = $bindable(false),
    onsaved,
    onresolved,
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
    /** Whether the box is open: read it, for a page that says the note
     * changed only when no box shows it already */
    editing?: boolean
    onsaved?: (note: string | null) => void
    /** The user answered a newer note (Use theirs or Keep mine) */
    onresolved?: () => void
  } = $props()

  /** Where a save lands: kept from before any wait, so a box switched to
   * another note meanwhile never writes onto it */
  type At = { key: string; target: string; workspace?: string }
  const here = (): At => ({
    key: noteKey(workspace ?? picker.current, target),
    target,
    workspace,
  })

  let saving = $state(false)
  // A save is under way, its question or its read included: Save waits
  let pending = $state(false)
  let error = $state<unknown>(null)
  // What the box last stored, until the parent hands down a newer note
  let shown = $derived(note ?? null)
  // The note the draft answers: as it was when the edit began, or the
  // newer one the user took in; undefined while the box is shut
  let base = $state<string | null | undefined>(undefined)
  // The server's own note on the target as last read, and its etag
  let known = $state<{ key: string; note: string | null; etag: string }>()
  const draft: NoteDraft = new NoteDraft({
    key: () => here().key,
    saved: () => (base === undefined ? shown : base) ?? '',
    save: () => save(),
  })
  const newer = $derived(draft.editing && !saving && shown !== base)
  $effect(() => {
    editing = draft.editing
    if (!editing) base = undefined
  })

  const etagOf = (text: string | null, at: At) =>
    known?.key === at.key && known.note === text ? known.etag : undefined

  /** The box still edits the note `at` names */
  const still = (at: At) => draft.editing && at.key === here().key

  /** Read the note's etag; a note other than the one shown is shown, so
   * the notice offers it. A note not written yet has an etag too (that of
   * ''). The read's error when it failed, else null */
  async function learn(at: At): Promise<{ error: unknown } | null> {
    try {
      const read = await api.noteForSave(at.target, at.workspace)
      if (at.key === here().key) {
        known = { key: at.key, note: read.note ?? null, etag: read.etag }
        if (draft.editing && known.note !== shown) shown = known.note
      }
      return null
    } catch (e) {
      return { error: e }
    }
  }

  // A draft set aside answers the note it was written over (`draft.base`):
  // resumed over a newer one, the notice shows it
  function edit() {
    const answered = draft.kept ? draft.base : null
    draft.open()
    base = answered === null ? shown : answered || null
    error = null
    void learn(here())
  }

  async function useTheirs() {
    if (
      draftAtRisk(draft.text, base ?? null) &&
      !(await confirmDialog('Replace your draft with the newer note?', {
        confirmLabel: 'Use theirs',
        cancelLabel: 'Keep editing',
        danger: true,
      }))
    )
      return
    draft.text = shown ?? ''
    base = shown
    onresolved?.()
  }

  function keepMine() {
    base = shown
    onresolved?.()
  }

  /** Over a newer note the user has not taken in, ask first */
  const mayReplace = async () =>
    !newer ||
    (await confirmDialog(
      `${label} changed while you were editing. Replace it with yours?`,
      { confirmLabel: 'Replace', cancelLabel: 'Keep editing', danger: true },
    ))

  // Every wait below can see the box switched to another note: `at`, taken
  // first, is checked after each, so nothing lands on the wrong one
  async function save(): Promise<void> {
    if (pending) return
    pending = true
    error = null
    const at = here()
    try {
      if (!(await mayReplace()) || !still(at)) return
      const seen = shown
      if (!etagOf(seen, at)) {
        const failed = await learn(at)
        if (!still(at)) return
        // Not knowing what is there, write nothing: the draft stays
        if (failed) {
          error = failed.error
          return
        }
      }
      const etag = etagOf(seen, at)
      // The read found a note the user has not seen: the notice has it
      if (shown !== seen || !etag) return
      await writing.run(at, draft.text, etag)
    } finally {
      pending = false
    }
  }

  // The write is an action like every other button's; it answers its own
  // failures (refused), so the action never needs to
  const writing = action(write, { doing: 'save the note' })
  async function write(at: At, text: string, etag: string) {
    saving = true
    error = null
    try {
      const note = text.trim() ? text : null
      const stored = await setNote(at.target, note, at.workspace, etag)
      // Switched away meanwhile: the box, its draft and the page now
      // belong to another note
      if (at.key !== here().key) return
      shown = stored.note ?? null
      known = undefined
      // Words typed while it was out stay, as a draft over the stored note
      if (!draft.done(text)) base = shown
      onsaved?.(shown)
    } catch (e) {
      await refused(at, e)
    } finally {
      saving = false
    }
  }

  /** A write the server refused: one that changed since it was read shows
   * the newer note, and writes nothing. Said only while the box is on it */
  async function refused(at: At, e: unknown) {
    const stale = e instanceof ApiError && e.code === 'stale_write'
    if (stale && at.key === here().key && !(await learn(at))) return
    if (at.key === here().key) error = e
  }
</script>

<div class="note">
  {#if draft.editing}
    {#if newer}
      <div class="panel newer" role="status">
        <p>
          {label} changed while you were editing, in another tab or by an agent
        </p>
        {#if shown}
          <div class="theirs">
            <ClampedText text={shown} {lines} markdown />
          </div>
        {:else}
          <p class="muted">It is empty now.</p>
        {/if}
        <div class="actions">
          <button type="button" class="quiet danger" onclick={useTheirs}>
            Use theirs
          </button>
          <button type="button" onclick={keepMine}>Keep mine</button>
        </div>
      </div>
    {/if}
    <textarea
      bind:value={draft.text}
      maxlength={NOTE_MAX}
      rows="4"
      aria-label={label}
      {placeholder}
      onkeydown={(e) => draft.keydown(e)}></textarea>
    <p class="hint muted">
      Markdown: # heading · - list · **bold** · *italic* · `code` ·
      [link](https://…)
      {#if draft.counter}<span class="num">{draft.counter}</span>{/if}
    </p>
    <div class="actions">
      <button
        type="button"
        class="quiet"
        onclick={save}
        disabled={saving || pending}
        title="Ctrl+Enter"
      >
        Save
      </button>
      <button
        type="button"
        class="bare"
        onclick={() => draft.cancel()}
        disabled={saving || pending}
      >
        Cancel
      </button>
    </div>
    {#if error}<ErrorLine {error} doing="save the note" />{/if}
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
      {#if draft.kept}<p class="hint">You have an unsaved draft.</p>{/if}
      <button
        type="button"
        class="quiet icon"
        aria-label="Edit note"
        title="edit this note"
        onclick={edit}
      >
        <Pencil size={13} />
      </button>
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
    margin: 0.25rem 0 0;
  }
  .actions {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.25rem;
  }
  .newer {
    margin: 0 0 0.5rem;
    border-color: var(--warn);
  }
  .newer p {
    margin: 0;
    overflow-wrap: anywhere;
  }
  .theirs {
    margin: 0.4rem 0;
  }
</style>
