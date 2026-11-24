<script lang="ts">
  import { untrack } from 'svelte'
  import { api, outputUrl } from '../api'
  import { jobMedia, type MediaFile } from '../media'
  import { openInHash } from '../mediaViewer'
  import { leafName } from '../names'
  import { resource } from '../resource.svelte'
  import { isBundle, runMetadataReader } from './runMetadata'
  import { describeOutputMeta } from '../outputMeta'
  import {
    sectionByRole,
    sectionBySubfolder,
    type StepGroup,
    type UnsavedStep,
  } from '../results'
  import type { GalleryInput, JobDetail, OutputKind } from '../types'
  import MissingFile from '../ui/MissingFile.svelte'
  import OutputTile from '../ui/OutputTile.svelte'
  import RoleTag from '../gallery/RoleTag.svelte'
  import { ASSET, OUTPUT, referenceName } from '../references'
  import { wsHref } from '../routes'
  import { isTimeline } from '../timeline'
  import TimelineStrip from './TimelineStrip.svelte'
  import TimingBar from './TimingBar.svelte'
  import JobViewer from './JobViewer.svelte'
  import ResultsHead from './ResultsHead.svelte'

  // The Results panel: its head (ResultsHead), then each output
  // as a tile with one line of what made it, sectioned by subfolder (or
  // role) and grouped by step, then the steps that wrote nothing. A tile
  // opens the job's outputs in the viewer (JobViewer). The page keys this
  // on the job id, so a switch starts with no lookups
  let {
    job,
    fileGroups,
    unsaved,
    kinds,
    seedVariable,
    running = false,
    onnote,
  }: {
    job: JobDetail
    fileGroups: StepGroup[]
    unsaved: UnsavedStep[]
    kinds: Record<string, string | null>
    seedVariable: string | null
    /** The job is still running, so a file open= names may yet arrive */
    running?: boolean
    /** Given for a finished take of a request item: "Note this file"
     * opens the page's note for the agent about that file */
    onnote?: (file: string) => void
  } = $props()

  // A step's timing is the manifest's: a job opened from history has no
  // events. A reused step, a task or a composed step carries none
  const timingOf = (step: string) => {
    const entry = job.manifest?.find((e) => e.step === step)
    return entry && !entry.reused ? (entry.timing ?? null) : null
  }

  const kindOf = (path: string) => kinds[path] ?? null

  // The metadata each image and timeline embeds, and the run's inputs
  // (runMetadata.ts): read as the run's files arrive, and again when the
  // gallery moves, which asks only what is new or failed
  const readRun = runMetadataReader()
  const run = resource(
    () =>
      readRun(
        fileGroups.flatMap((group) => group.files),
        kindOf,
        job.workspace,
      ),
    { watch: ['gallery'], workspace: () => job.workspace },
  )
  const fileMeta = $derived(run.data?.meta ?? {})
  const fileMedia = $derived(run.data?.media ?? {})
  const runInputs = $derived(run.data?.inputs ?? [])

  // Nothing at all was generated: every step the manifest lists was served
  // from the step cache. Worth saying outright - the page otherwise shows a
  // succeeded job full of images that are not this run's
  const allReused = $derived(
    fileGroups.length > 0 && fileGroups.every((group) => group.reused),
  )
  // Sections by result.subfolder - one root section, no heading, when no
  // step chose one, so an older run renders as it always did
  const sections = $derived(sectionBySubfolder(fileGroups))
  // By role when the manifest states one: Final, Intermediate, Made from.
  // Null for a run that predates roles, which keeps the subfolder layout
  const byRole = $derived(sectionByRole(fileGroups))
  const inputName = (input: GalleryInput) =>
    referenceName(input.reference, input.kind === 'asset' ? ASSET : OUTPUT) ??
    input.reference
  const sectioned = $derived(
    sections.length > 1 || sections[0].subfolder !== '',
  )
  // Every file in the order the panel shows them, finals first: what the
  // viewer steps through, Compare picks from and Download all takes
  const ordered = $derived(
    (byRole
      ? [...byRole.final, ...byRole.intermediate]
      : sections.flatMap((section) => section.groups)
    ).flatMap((group) => group.files),
  )
  // A kind the gallery does not show (null) has no tile: its name links it
  const tileOf = (file: string): MediaFile | null => {
    const kind = kindOf(file) as OutputKind | null
    return kind ? jobMedia(file, kind, job.id, job.workspace) : null
  }
  const items = $derived(
    ordered.map(tileOf).filter((t): t is MediaFile => t !== null),
  )
  let open = $state<string | null>(openInHash())
  // An open= that names no output: while the job runs the file may still
  // arrive, and the viewer opens on it then; once it has finished, said
  // under the head and cleared, as the Gallery does
  let missing = $state('')
  $effect(() => {
    const name = open
    if (!name || running || items.some((item) => item.name === name)) return
    untrack(() => {
      missing = leafName(name)
      open = null
    })
  })

  // Two runs of a workflow write the same file names, so the job id rides
  // along - without it the browser shows this job the image it cached from
  // the previous one. The job's own workspace rides along too, so its media
  // still loads correctly if the picker has since moved elsewhere.
  const fileUrl = (path: string) => outputUrl(path, job.id, job.workspace)
  // A bundle member is linked only when the clip that plays it recorded an
  // `output:` source that is one of this job's files. Members are all
  // media/<name>, so a name says nothing: an asset that shares a name with a
  // job file has no source here and gets no link
  const memberHref = (_member: string, source: string | null) => {
    const path = referenceName(source, OUTPUT)
    if (!path) return null
    const match = fileGroups
      .flatMap((group) => group.files)
      .find((file) => file === path)
    return match ? api.outputDownloadUrl(match, job.workspace) : null
  }
</script>

<ResultsHead files={ordered} {items} workspace={job.workspace} />
<MissingFile bind:leaf={missing} />
{#if allReused}
  <p class="muted reusednote">
    Served from the step cache: every step matched an earlier job with the same
    seed and inputs, so these are that job's files and nothing was generated.{#if seedVariable}
      Use <strong>New seed</strong> for a different image.{/if}
  </p>
{/if}
{#if byRole}
  <div class="roles">
    <h3 class="subhead">Final</h3>
    {@render stepGroups(byRole.final, true)}
    {#if byRole.intermediate.length}
      <details open={byRole.final.length === 0}>
        <summary>Intermediate</summary>
        {@render stepGroups(byRole.intermediate, true)}
      </details>
    {/if}
    {#if runInputs.length}
      <h3 class="subhead">Made from</h3>
      <ul class="madefrom">
        {#each runInputs as input (input.kind + input.reference)}
          <li>
            <a
              href={input.kind === 'asset'
                ? wsHref(job.workspace, 'assets')
                : fileUrl(inputName(input))}
              target={input.kind === 'output' ? '_blank' : undefined}
            >
              {inputName(input)}
            </a>
            <RoleTag role="input" />
            <span class="muted">{input.kind}</span>
          </li>
        {/each}
      </ul>
    {/if}
  </div>
{:else}
  {#each sections as section (section.subfolder)}
    {#if sectioned}
      <h3 class="subhead">
        {section.subfolder === '' ? '(top folder)' : `${section.subfolder}/`}
      </h3>
    {/if}
    {@render stepGroups(section.groups, sectioned)}
  {/each}
{/if}
{#snippet stepGroups(groups: StepGroup[], nested: boolean)}
  {#each groups as group (group.step)}
    {#if fileGroups.length > 1}
      <svelte:element this={nested ? 'h4' : 'h3'} class="stephead muted">
        {group.step}
        {#if group.reused && !allReused}
          <span
            class="muted"
            title="served from the step cache - an
                 earlier job's files, nothing generated for this step"
          >
            · reused
          </span>
        {:else}
          {@const timing = timingOf(group.step)}
          {#if timing}<TimingBar {timing} />{/if}
        {/if}
      </svelte:element>
    {/if}
    <div class="tiles">
      {#each group.files as file (file)}
        {@const tile = tileOf(file)}
        {@const info = describeOutputMeta(fileMeta[file])}
        <!-- data-step is what a flow-graph node scrolls to -->
        <div class="output" data-step={group.step}>
          {#if tile}
            <OutputTile
              file={tile}
              size="card"
              caption="line"
              active={open === file}
              onclick={() => (open = file)}
              aria-label="Open {tile.label}"
              title="Open this file, with what made it"
            >
              {#snippet meta()}
                {#if info.model || info.seed !== undefined}
                  <span class="recipe">
                    {[
                      info.model,
                      info.seed !== undefined && `seed ${info.seed}`,
                    ]
                      .filter(Boolean)
                      .join(' · ')}
                  </span>
                {/if}
              {/snippet}
            </OutputTile>
          {:else}
            <a class="filelink" href={fileUrl(file)} target="_blank">
              {leafName(file)}
            </a>
          {/if}
          {#if isBundle(file) && isTimeline(fileMedia[file])}
            <TimelineStrip summary={fileMedia[file]} {memberHref} />
          {/if}
        </div>
      {/each}
    </div>
  {/each}
{/snippet}
{#if unsaved.length}
  <h3 class="subhead">Steps that wrote nothing</h3>
  <ul class="unsaved">
    {#each unsaved as entry (entry.node)}
      <li data-step={entry.node}>
        <code>{entry.node}</code>
        {#if entry.members.length}
          <span
            class="muted"
            title="this step has for_each: {entry.members.join(', ')}"
          >
            × {entry.members.length}
          </span>
        {/if}
        <span class="muted why">
          {#if entry.reason}<code>{entry.reason.key}</code>
            {entry.reason.detail}{:else}no file written{/if}
        </span>
      </li>
    {/each}
  </ul>
{/if}

<JobViewer {items} bind:open {onnote} />

<style>
  .roles details {
    margin-top: var(--space-3);
  }
  .roles summary {
    cursor: pointer;
    font-size: var(--t-sm);
    font-weight: 600;
    margin-bottom: var(--space-2);
  }
  .madefrom {
    list-style: none;
    margin: 0;
    padding: 0;
    font-size: var(--t-sm);
  }
  .madefrom li {
    padding: 0.15rem 0;
  }
  .reusednote {
    margin: 0 0 0.6rem;
  }

  /* A step's outputs as a grid of tiles; a timeline's strip under its
     file takes the row */
  .tiles {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(min(220px, 100%), 1fr));
    gap: var(--space-3);
  }
  .output {
    min-width: 0;
  }
  .output:has(:global(.timeline)) {
    grid-column: 1 / -1;
  }
  .filelink {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
    overflow-wrap: anywhere;
  }
  .recipe {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-family: var(--font-mono);
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
