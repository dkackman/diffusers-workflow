<script lang="ts">
  // The Results panel's head: Compare, which opens up to four of the job's
  // pictures and clips side by side, finals first, and Download all, every
  // file the job wrote as one archive
  import { Archive, Columns2 } from '@lucide/svelte'
  import { action } from '../action.svelte'
  import { api } from '../api'
  import { hasPicture, type MediaFile } from '../media'
  import { COMPARE_MAX } from '../compare'
  import { goHref } from '../router.svelte'
  import { compareHref, jobHref } from '../routes'

  let {
    files,
    items,
    workspace,
    jobId,
  }: {
    /** Every file, in the panel's order */
    files: string[]
    /** The files a tile shows, in the same order */
    items: MediaFile[]
    /** The job's own workspace */
    workspace: string
    /** The job, which Compare's back link returns to */
    jobId: string
  } = $props()

  const comparable = $derived(
    items.filter((item) => hasPicture(item.kind)).map((item) => item.name),
  )
  const compareTitle = $derived(
    comparable.length > COMPARE_MAX
      ? `Compares the first ${COMPARE_MAX}`
      : 'Compare these side by side',
  )
  function compare() {
    goHref(
      compareHref(workspace, comparable.slice(0, COMPARE_MAX), {
        href: jobHref(jobId, workspace),
        label: 'Job',
      }),
    )
  }
  const downloading = action((names: string[]) => api.archiveOutputs(names), {
    doing: "download this job's files",
  })
</script>

<div class="resultshead">
  <h2>Results</h2>
  {#if comparable.length >= 2}
    <button
      type="button"
      class="quiet withicon"
      title={compareTitle}
      onclick={compare}
    >
      <Columns2 size={14} />Compare
    </button>
  {/if}
  {#if files.length}
    <button
      type="button"
      class="quiet withicon"
      title="download every file this job wrote, as one archive"
      disabled={downloading.busy}
      onclick={() => downloading.run(files)}
    >
      <Archive size={14} />Download all
    </button>
  {/if}
</div>

<style>
  .resultshead {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: var(--space-2);
  }
  .resultshead h2 {
    margin: 0 auto 0 0;
  }
</style>
