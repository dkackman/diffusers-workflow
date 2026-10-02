<script lang="ts">
  import { untrack } from 'svelte'
  import { api, outputUrl } from '../api'
  import { describeOutputMeta } from '../outputMeta'
  import {
    sectionBySubfolder,
    type StepGroup,
    type UnsavedStep,
  } from '../results'
  import DownloadLink from '../DownloadLink.svelte'
  import type { JobDetail } from '../types'

  // The Results panel's body: each output beside what made it, sectioned by
  // subfolder and grouped by step, then the steps that wrote nothing. The
  // page keys this on the job id, so a switch starts with no lookups
  let {
    job,
    fileGroups,
    unsaved,
    kinds,
    seedVariable,
  }: {
    job: JobDetail
    fileGroups: StepGroup[]
    unsaved: UnsavedStep[]
    kinds: Record<string, string | null>
    seedVariable: string | null
  } = $props()

  const kindOf = (path: string) => kinds[path] ?? null

  // The generation metadata embedded in each image, keyed by file - per
  // file rather than per step, since each image of a batch carries its own
  // seed. Images only: the metadata route decodes an audio or video file
  // whole to probe it, and nothing it would report is shown here. A key
  // present with null is a lookup already in flight or failed
  let fileMeta = $state<Record<string, Record<string, unknown> | null>>({})
  $effect(() => {
    const workspace = job.workspace
    const pending = fileGroups
      .flatMap((group) => group.files)
      .filter(
        (file) =>
          kindOf(file) === 'image' && !(file in untrack(() => fileMeta)),
      )
    if (!pending.length) return
    untrack(() => {
      for (const file of pending) {
        fileMeta[file] = null
        api
          .galleryMetadata(file, workspace)
          .then((r) => {
            fileMeta[file] = r.metadata
          })
          .catch(() => {})
      }
    })
  })

  // Nothing at all was generated: every step the manifest lists was served
  // from the step cache. Worth saying outright - the page otherwise shows a
  // succeeded job full of images that are not this run's
  const allReused = $derived(
    fileGroups.length > 0 && fileGroups.every((group) => group.reused),
  )
  // Sections by result.subfolder - one root section, no heading, when no
  // step chose one, so an older run renders as it always did
  const sections = $derived(sectionBySubfolder(fileGroups))
  const sectioned = $derived(
    sections.length > 1 || sections[0].subfolder !== '',
  )
  // Two runs of a workflow write the same file names, so the job id rides
  // along - without it the browser shows this job the image it cached from
  // the previous one. The job's own workspace rides along too, so its media
  // still loads correctly if the picker has since moved elsewhere.
  const fileUrl = (path: string) => outputUrl(path, job.id, job.workspace)
</script>

{#if allReused}
  <p class="muted reusednote">
    Served from the step cache: every step matched an earlier run with the same
    seed and inputs, so these are that run's files and nothing was generated.{#if seedVariable}
      Use <strong>New seed</strong> for a different image.{/if}
  </p>
{/if}
{#each sections as section (section.subfolder)}
  {#if sectioned}
    <h3 class="subhead">
      {section.subfolder === '' ? '(run root)' : `${section.subfolder}/`}
    </h3>
  {/if}
  {#each section.groups as group (group.step)}
    {#if fileGroups.length > 1}
      <svelte:element this={sectioned ? 'h4' : 'h3'} class="stephead muted">
        {group.step}
        {#if group.reused && !allReused}
          <span
            class="muted"
            title="served from the step cache - an
                 earlier run's files, nothing generated for this step"
            >· reused</span
          >
        {/if}
      </svelte:element>
    {/if}
    {#each group.files as file (file)}
      {@const info = describeOutputMeta(fileMeta[file])}
      <!-- Laid out as the gallery detail is: the media on the left,
           what made it on the right -->
      <div class="output">
        {#if kindOf(file) === 'image'}
          <a
            class="frame plain"
            href={fileUrl(file)}
            target="_blank"
            title={file.split('/').pop()}
            ><img src={fileUrl(file)} alt={file.split('/').pop()} /></a
          >
        {:else if kindOf(file) === 'video'}
          <span class="frame">
            <!-- svelte-ignore a11y_media_has_caption -->
            <video src={fileUrl(file)} controls loop></video>
          </span>
        {:else if kindOf(file) === 'audio'}
          <span class="frame">
            <audio src={fileUrl(file)} controls></audio>
          </span>
        {:else}
          <a class="filelink" href={fileUrl(file)} target="_blank"
            >{file.split('/').pop()}</a
          >
        {/if}
        <div class="meta">
          <div class="metabar">
            <span class="filename">{file.split('/').pop()}</span>
            <DownloadLink href={api.outputDownloadUrl(file, job.workspace)} />
          </div>
          {#if info.model}
            <div>
              <span class="muted">model</span>
              <code>{info.model}</code>
            </div>
          {/if}
          {#if info.seed !== undefined}
            <div>
              <span class="muted">seed</span> <code>{info.seed}</code>
            </div>
          {/if}
          {#if info.prompt}
            <div class="prompt">
              <span class="muted">prompt</span>
              <p>{info.prompt}</p>
            </div>
          {/if}
          {#if info.negativePrompt}
            <div class="prompt">
              <span class="muted">negative prompt</span>
              <p>{info.negativePrompt}</p>
            </div>
          {/if}
        </div>
      </div>
    {/each}
  {/each}
{/each}
{#if unsaved.length}
  <h3 class="subhead">Steps that wrote nothing</h3>
  <ul class="unsaved">
    {#each unsaved as entry (entry.node)}
      <li>
        <code>{entry.node}</code>
        {#if entry.members.length}
          <span
            class="muted"
            title="this step has for_each: {entry.members.join(', ')}"
            >× {entry.members.length}</span
          >
        {/if}
        <span class="muted why">
          {#if entry.reason}<code>{entry.reason.key}</code>
            {entry.reason.detail}{:else}no file written{/if}
        </span>
      </li>
    {/each}
  </ul>
{/if}

<style>
  .reusednote {
    margin: 0 0 0.6rem;
  }

  /* One output per row, the gallery detail's shape: the proof on the left,
     the recipe that made it on the right */
  .output {
    display: flex;
    gap: var(--space-4);
    align-items: flex-start;
    flex-wrap: wrap;
  }
  .output + .output {
    margin-top: var(--space-3);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line);
  }
  /* What the run produced, framed the way the catalog and the gallery
     frame it - the picture flush to its edges, no rounding of its own */
  .output :global(.frame) {
    max-width: min(480px, 100%);
  }
  .output :global(.frame > video) {
    height: auto;
  }
  .output a.filelink {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
  }
  .meta {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    font-size: 0.85rem;
    max-width: 46ch;
    min-width: 0;
  }
  .meta .muted {
    margin-right: 0.4rem;
  }
  .metabar {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.3rem;
  }
  /* The name the engine wrote, and what you would type to reference it */
  .filename {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
    overflow-wrap: anywhere;
    flex: 1;
  }
  .prompt p {
    margin: 0.15rem 0 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .stephead {
    font-family: var(--font-mono);
    font-weight: 600;
    line-height: 1.15;
    letter-spacing: -0.01em;
    font-size: 0.78rem;
    text-transform: none;
    margin: var(--space-3) 0 var(--space-2);
  }
  .stephead:first-of-type {
    margin-top: 0;
  }
  .subhead {
    font-size: var(--t-sm);
    text-transform: none;
    margin: var(--space-3) 0 var(--space-1);
  }
  .subhead:first-of-type {
    margin-top: 0;
  }
  .subhead + .stephead {
    margin-top: 0;
  }
  /* The steps whose files are deliberately absent: the step name is what
     the engine resolves, the reason is written for a person */
  .unsaved {
    list-style: none;
    margin: 0;
    padding: 0;
    font-size: var(--t-sm);
  }
  .unsaved li {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.5rem;
    padding: 0.15rem 0;
  }
  .unsaved code {
    font-size: 0.8rem;
  }
  .unsaved .why {
    min-width: 0;
  }
</style>
