<script lang="ts">
  import LinkedNotes from '../request/LinkedNotes.svelte'
  import { LinkedRequest } from '../request/linkedRequest.svelte'
  import type { GalleryLinkedJob, LedgerNote } from '../types'

  // The gallery file's notes for the agent: one box per take that
  // published the file. A file a step-cache hit republished belongs to more
  // than one take, so the user says which first; a file no take of a
  // request item published has no box
  let {
    file,
    linkedJobs,
    workspace,
  }: {
    /** The file, as the gallery names it */
    file: string
    /** The metadata's linked_jobs, newest first */
    linkedJobs: GalleryLinkedJob[]
    workspace: string
  } = $props()

  const takes = $derived(linkedJobs.filter((job) => job.item))
  let chosen = $state('')
  $effect(() => {
    void file
    chosen = ''
  })
  const take = $derived(
    takes.length === 1
      ? takes[0]
      : (takes.find((job) => job.id === chosen) ?? null),
  )
  const takeLabel = (job: GalleryLinkedJob) =>
    `take ${job.attempt ?? '?'} of item ${job.item}`

  const requestId = $derived(take?.request)
  const linked = $derived(
    requestId ? new LinkedRequest(workspace, requestId) : null,
  )
  $effect(() => {
    void linked?.read()
  })
  const aboutFile = (note: LedgerNote) =>
    (note.refs ?? []).some((ref) => ref.output === file)
</script>

{#if takes.length > 1}
  <label class="take">
    <span class="muted">Note for the agent about</span>
    <select bind:value={chosen}>
      <option value="">Which take?</option>
      {#each takes as job (job.id)}
        <option value={job.id}>{takeLabel(job)}</option>
      {/each}
    </select>
  </label>
{/if}
{#if linked && take?.item}
  {#key take.id}
    <LinkedNotes
      {linked}
      item={take.item}
      filter={aboutFile}
      refs={[{ item: take.item, job: take.id, output: file }]}
    />
  {/key}
{/if}

<style>
  .take {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1) var(--space-2);
    font-size: var(--t-sm);
  }
  .take select {
    max-width: 100%;
  }
</style>
