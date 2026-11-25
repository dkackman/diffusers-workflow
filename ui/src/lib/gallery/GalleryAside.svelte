<script lang="ts">
  // What the Gallery adds under a file's recipe in the viewer (RecipePanel's
  // `children`, handed its one read): the file's role, run and size, the
  // note on it or on its job, the findings, and the notes for the agent.
  // A saved note reads the file again through the panel, so the page can
  // carry what the server now says to the tile
  import type { GalleryFile, GalleryMetadata } from '../types'
  import { formatBytes } from '../format'
  import Time from '../ui/Time.svelte'
  import { versionTitle } from '../terms'
  import NoteEditor from '../ui/NoteEditor.svelte'
  import type { AssessState } from './assess.svelte'
  import FileAgentNotes from './FileAgentNotes.svelte'
  import GalleryFindings from './GalleryFindings.svelte'
  import RoleTag from './RoleTag.svelte'

  let {
    file,
    meta,
    reload,
    workspace,
    assessed,
    onnotesaved,
  }: {
    file: GalleryFile
    /** The panel's read of the file */
    meta: GalleryMetadata
    /** The panel's read again: the fresh one, or null when it failed or
     * was overtaken */
    reload: () => Promise<GalleryMetadata | null>
    /** The workspace the file belongs to */
    workspace: string
    assessed?: AssessState
    /** A note changed, and the file's effective note now */
    onnotesaved?: (
      kind: 'run' | 'file',
      note: string | null,
    ) => void | Promise<void>
  } = $props()

  /** The file's role in its run; null when the run does not say */
  const role = $derived(meta.role ?? file.role ?? null)
  // The run the file belongs to, as a note target; empty under the flat layout
  const runDir = $derived(file.run_id ? `${file.folder}/${file.run_id}` : '')
  // A file inheriting its run's note shows that one; "Note this file" swaps
  // in a box for the file's own, which then overrides it. The page keys
  // this by the file, so each file starts on the inherited one
  let ownNote = $state(false)
  const inherited = $derived(meta.note_source === 'run' && !ownNote && !!runDir)

  // Back to what the server says once it has answered: a cleared file note
  // returns the run's, a saved one is the file's own. A read that failed
  // or was overtaken hands on nothing from before the save
  async function noteSaved(kind: 'run' | 'file') {
    const fresh = await reload()
    if (!fresh) return
    await onnotesaved?.(kind, fresh.note ?? null)
  }
  async function saveFileNote() {
    await noteSaved('file')
    ownNote = false
  }
</script>

<div class="gallery-aside">
  <p class="facts">
    <RoleTag {role} />
    {#if file.version}
      <!-- The job this file came from, said in both the form a person is
           quoted ("v4") and the form every tool takes (the run id) -->
      <span class="num muted" title={versionTitle(file.version)}>
        v{file.version} · <code>{file.run_id}</code>
      </span>
    {/if}
    <span class="num muted">
      {formatBytes(file.size)} · <Time at={file.mtime} mode="exact" />
    </span>
  </p>
  {#if inherited}
    <NoteEditor
      note={meta.run_note}
      target={runDir}
      {workspace}
      label="Your note on this job"
      onsaved={() => noteSaved('run')}
    />
    <button type="button" class="bare" onclick={() => (ownNote = true)}>
      Note this file
    </button>
  {:else}
    <NoteEditor
      note={ownNote ? null : meta.note}
      target={file.name}
      {workspace}
      label="Your note on this file"
      onsaved={saveFileNote}
    />
  {/if}
  <GalleryFindings findings={meta.findings ?? []} {assessed} />
  <FileAgentNotes
    file={file.name}
    linkedJobs={meta.linked_jobs ?? []}
    {workspace}
  />
</div>

<style>
  .gallery-aside {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    font-size: var(--t-sm);
    min-width: 0;
  }
  .facts {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1) var(--space-2);
    margin: 0;
  }
</style>
