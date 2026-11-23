<script lang="ts">
  import { Bookmark, FolderOpen, Gauge, Star, Trash2, X } from '@lucide/svelte'
  import DownloadLink from '../DownloadLink.svelte'
  import { ApiError, api } from '../api'
  import { goWs } from '../router.svelte'
  import { notify } from '../toast'
  import { confirmDialog } from '../confirm.svelte'
  import { describeOutputMeta } from '../outputMeta'
  import type { GalleryFile, GalleryInput } from '../types'
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import GalleryFindings from './GalleryFindings.svelte'
  import FileAgentNotes from './FileAgentNotes.svelte'
  import { AssessCache } from './assess.svelte'
  import RoleTag from './RoleTag.svelte'
  import NoteEditor from '../ui/NoteEditor.svelte'
  import { ASSET, OUTPUT, referenceName } from '../references'
  import { outputUrl } from '../apiUrls'
  import { wsHref } from '../routes'
  import { workspace } from '../workspace.svelte'
  import { formatBytes, formatMtime } from '../format'

  // The detail panel under the grid for the one file picked. The page owns
  // the pick (and keys this panel by it, so a new pick starts a new read);
  // this reads what made the file and owns what the panel does with it
  let {
    file,
    assessCache = new AssessCache(),
    onnotesaved,
    starred = false,
    onstar,
    onremove,
    onclose,
  }: {
    file: GalleryFile
    /** Assess results by output for the page's life; the page passes its
     * own so one survives the panel closing */
    assessCache?: AssessCache
    /** A note changed, and the file's effective note now, so the page can
     * carry it to the tile */
    onnotesaved?: (
      kind: 'run' | 'file',
      note: string | null,
    ) => void | Promise<void>
    /** Whether the file is starred, and the toggle */
    starred?: boolean
    onstar?: () => void
    onremove: () => void
    onclose: () => void
  } = $props()

  const meta = resource(() => api.galleryMetadata(file.name), {
    watch: ['gallery'],
  })
  const metadata = $derived(meta.data?.metadata ?? null)
  const sourceJob = $derived(meta.data?.job ?? null)
  /** The file's role in its run; null when the run does not say */
  const role = $derived(meta.data?.role ?? file.role ?? null)
  const inputs = $derived(meta.data?.inputs ?? [])
  const findings = $derived(meta.data?.findings ?? [])
  const linkedJobs = $derived(meta.data?.linked_jobs ?? [])
  const note = $derived(meta.data?.note ?? null)
  const runNote = $derived(meta.data?.run_note ?? null)
  const noteSource = $derived(meta.data?.note_source ?? null)

  // A text file's contents, fetched when the pick changes
  let textBody = $state<string | null>(null)
  $effect(() => {
    if (file.kind !== 'text') return
    const url = file.url
    textBody = null
    let stale = false
    fetch(url)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(r.statusText))))
      .then((t) => {
        if (!stale) textBody = t
      })
      .catch(() => {
        if (!stale) textBody = '(could not load this file)'
      })
    return () => {
      stale = true
    }
  })

  /** keep_output's 409 for a name already in the library. Its 409 also
   * means "no asset library", and neither refusal carries a code, so the
   * taken one is told apart by its wording. */
  function nameTaken(e: unknown): boolean {
    if (!(e instanceof ApiError) || e.status !== 409) return false
    return e.code === 'exists' || e.message.includes('already exists')
  }

  /** Keep this file as an input asset, so a later workflow can name it
   * without depending on the run that made it. */
  async function keepAsAsset() {
    const suggestion = file.name.split('/').pop() ?? file.name
    const assetName = window.prompt(
      'Keep as asset — name in the asset library:',
      suggestion,
    )
    if (!assetName) return
    try {
      const result = await api.keepOutput(file.name, assetName)
      notify.success(`Kept as ${result.reference}`)
    } catch (e) {
      // The server refuses an existing name rather than replacing it; the
      // choice to replace belongs to the person, not the button
      if (nameTaken(e)) {
        const replace = await confirmDialog(
          `${ASSET}${assetName} already exists. Replace it?`,
          { confirmLabel: 'Replace', title: 'Replace asset' },
        )
        // Declining is an answer, not a failure: nothing to report
        if (!replace) return
        try {
          const result = await api.keepOutput(file.name, assetName, true)
          notify.success(`Kept as ${result.reference}`)
        } catch (failure) {
          notify.failure(`keep ${file.name} as an asset`, failure)
        }
      } else {
        notify.failure(`keep ${file.name} as an asset`, e)
      }
    }
  }

  // Assess measures shots, seams and a soundtrack; a still has none, and
  // the title says so in the server's own words (dw/server/assess.py)
  const assessable = $derived(file.kind === 'video' || file.kind === 'audio')
  const assessKey = $derived(AssessCache.key(workspace.current, file.name))
  const assessed = $derived(assessCache.get(assessKey))
  const assessTitle = $derived(
    assessable
      ? 'measure this file for shots, seams, sync and level - places to look, not verdicts'
      : file.kind === 'image'
        ? 'a still has no shots, seams or soundtrack to measure'
        : 'not a video or audio file',
  )

  const embeddedWorkflow = $derived(
    (metadata?.workflow as Record<string, unknown> | undefined) ?? null,
  )

  const described = $derived(describeOutputMeta(metadata))
  const prompt = $derived(described.prompt)
  const negativePrompt = $derived(described.negativePrompt)
  const seed = $derived(described.seed)

  // Where a run input leads: an output opens its file, an asset the library
  const inputName = (input: GalleryInput) =>
    referenceName(input.reference, input.kind === 'asset' ? ASSET : OUTPUT) ??
    input.reference
  const inputHref = (input: GalleryInput) =>
    input.kind === 'asset'
      ? wsHref(workspace.current, 'assets')
      : outputUrl(inputName(input))
  // The folder the file sits in, as the engine wrote it
  const location = $derived(
    file.name.includes('/')
      ? file.name.slice(0, file.name.lastIndexOf('/'))
      : '',
  )

  // The run the file belongs to, as a note target; empty under the flat layout
  const runDir = $derived(file.run_id ? `${file.folder}/${file.run_id}` : '')
  // A file inheriting its run's note shows that one; "Note this file" swaps
  // in a box for the file's own, which then overrides it
  let ownNote = $state(false)
  $effect(() => {
    void file.name
    ownNote = false
  })
  const inherited = $derived(noteSource === 'run' && !ownNote && !!runDir)

  // Back to what the server says once it has answered: a cleared file note
  // returns the run's, a saved one is the file's own
  // A read that failed, or one a feed reload overtook (still loading),
  // leaves the note from before the save: not handed on. The listing's own
  // feed reload carries the note to the tile
  async function noteSaved(kind: 'run' | 'file') {
    await meta.reload()
    if (meta.status !== 'ready') return
    await onnotesaved?.(kind, meta.data?.note ?? null)
  }
  async function saveFileNote() {
    await noteSaved('file')
    ownNote = false
  }

  function openAsWorkflow() {
    if (!embeddedWorkflow) return
    // Pin the run's seed into the definition so reopening reproduces this
    // exact image; delete the seed field in the editor to re-randomize
    const definition = { ...embeddedWorkflow }
    if (typeof metadata?.seed === 'number') definition.seed = metadata.seed
    sessionStorage.setItem('dw-editor-import', JSON.stringify(definition))
    goWs('edit')
  }
</script>

<div class="detail panel">
  <div class="bar">
    <strong class="selname">{file.name}</strong>
    <span class="flex"></span>
    {#if embeddedWorkflow}
      <button
        class="quiet withicon"
        onclick={openAsWorkflow}
        title="open the embedded workflow definition in the editor"
      >
        <FolderOpen size={14} />Open as workflow
      </button>
    {/if}
    <button
      class="quiet withicon"
      onclick={keepAsAsset}
      title="keep this file as an input asset, under a name later workflows can use"
    >
      <Bookmark size={14} />Keep as asset
    </button>
    <button
      class="quiet withicon"
      onclick={() => assessCache.run(assessKey, file.name)}
      disabled={!assessable || assessed?.running}
      title={assessTitle}
    >
      <Gauge size={14} />{assessed?.running ? 'Assessing…' : 'Assess'}
    </button>
    <a
      href={file.url}
      target="_blank"
      class="muted"
      title="open the file itself in a new tab"
    >
      open file
    </a>
    {#if file.version}
      <!-- The run this file came from, said in both the form a person is
           quoted ("version 4") and the form every tool takes (the run
           id), so the two can be checked against each other here rather
           than back in the listing -->
      <span class="num muted">
        version {file.version} · <code>{file.run_id}</code>
      </span>
    {/if}
    <span class="num muted">
      {formatBytes(file.size)} · {formatMtime(file.mtime)}
    </span>
    <DownloadLink href={api.outputDownloadUrl(file.name)} />
    <button
      class="quiet icon"
      onclick={onstar}
      aria-pressed={starred}
      aria-label={starred ? 'unstar this file' : 'star this file'}
      title={starred ? 'starred - press S to unstar' : 'star to keep - press S'}
    >
      <Star size={14} fill={starred ? 'currentColor' : 'none'} />
    </button>
    <button
      class="quiet icon danger"
      onclick={onremove}
      title="delete this file from the output directory"
      aria-label="delete this file from the output directory"
    >
      <Trash2 size={14} />
    </button>
    <span class="flex"></span>
    <button
      class="quiet icon"
      onclick={onclose}
      title="close details"
      aria-label="close details"
    >
      <X size={14} />
    </button>
  </div>
  <div class="body">
    {#if file.kind === 'image'}
      <img src={file.url} alt={file.name} />
    {:else if file.kind === 'video'}
      <!-- svelte-ignore a11y_media_has_caption -->
      <video src={file.url} controls loop></video>
    {:else if file.kind === 'audio'}
      <audio src={file.url} controls></audio>
    {:else if file.kind === 'text'}
      <pre class="text">{textBody ?? 'loading…'}</pre>
    {/if}
    <LoadState of={meta} label="what made this file">
      {#if metadata || role || inputs.length || findings.length || assessed || linkedJobs.length}
        <div class="meta">
          <div><RoleTag {role} /></div>
          {#if inherited}
            <NoteEditor
              note={runNote}
              target={runDir}
              label="Run note"
              onsaved={() => noteSaved('run')}
            />
            <button type="button" class="bare" onclick={() => (ownNote = true)}>
              Note this file
            </button>
          {:else}
            <NoteEditor
              note={ownNote ? null : note}
              target={file.name}
              label={ownNote || noteSource === 'file' ? 'File note' : 'Note'}
              onsaved={saveFileNote}
            />
          {/if}
          <GalleryFindings {findings} {assessed} />
          <FileAgentNotes
            file={file.name}
            {linkedJobs}
            workspace={workspace.current}
          />
          {#if metadata}
            <details class="settings">
              <summary>Settings</summary>
              {#if metadata.step_name}<div>
                  <span class="muted">step</span>
                  {metadata.step_name}
                </div>{/if}
              {#if metadata.model_name}<div>
                  <span class="muted">model</span>
                  <code>{metadata.model_name}</code>
                </div>{/if}
              {#if seed !== undefined}
                <div>
                  <span class="muted">seed</span> <code>{seed}</code>
                </div>
              {/if}
              {#if prompt}
                <div class="prompt">
                  <span class="muted">prompt</span>
                  <p>{prompt}</p>
                </div>
              {/if}
              {#if negativePrompt}
                <div class="prompt">
                  <span class="muted">negative prompt</span>
                  <p>{negativePrompt}</p>
                </div>
              {/if}
              {#if sourceJob}
                <div>
                  <span class="muted">job</span>
                  <a
                    href={'#/jobs/' + sourceJob.id}
                    title="open the job that produced this file"
                  >
                    {sourceJob.id}
                  </a>
                </div>
              {/if}
              {#if embeddedWorkflow}
                <div>
                  <span class="muted">workflow</span>
                  {embeddedWorkflow.id}
                </div>
              {:else}
                <div class="muted">
                  no embedded workflow - enable embed_metadata in the step's
                  result
                </div>
              {/if}
            </details>
          {/if}
          {#if inputs.length}
            <div class="madefrom">
              <span class="muted">Made from</span>
              <ul>
                {#each inputs as input (input.kind + input.reference)}
                  <li>
                    <a
                      href={inputHref(input)}
                      target={input.kind === 'output' ? '_blank' : undefined}
                      title={input.kind === 'asset'
                        ? 'open the asset library'
                        : 'open this earlier output'}
                    >
                      {inputName(input)}
                    </a>
                    <RoleTag role="input" />
                    <span class="muted">{input.kind}</span>
                  </li>
                {/each}
              </ul>
            </div>
          {/if}
          {#if location}
            <div class="location">
              <span class="muted">Location</span>
              <code>{location}/</code>
            </div>
          {/if}
        </div>
      {:else if file.kind === 'image'}
        <div class="meta muted">
          no embedded metadata - enable embed_metadata in the step's result
        </div>
      {/if}
    </LoadState>
  </div>
</div>

<style>
  .body pre.text {
    margin: 0;
    max-height: 24rem;
    overflow: auto;
    white-space: pre-wrap;
    font-family: var(--font-mono);
    font-size: var(--t-xs);
  }
  .detail {
    position: sticky;
    bottom: 1rem;
    margin-top: 1rem;
    box-shadow: 0 6px 24px rgb(0 0 0 / 0.35);
  }
  .bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.8rem;
    margin-bottom: 0.7rem;
  }
  /* The path the engine wrote, and what you would type to reference it */
  .selname {
    font-family: var(--font-mono);
    font-size: var(--t-sm);
    overflow-wrap: anywhere;
  }
  .body {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    flex-wrap: wrap;
  }
  .body img,
  .body video {
    max-width: min(480px, 100%);
    border: 1px solid var(--line);
    border-radius: var(--radius-frame);
  }
  .meta {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    font-size: 0.85rem;
    max-width: 46ch;
  }
  .meta .muted {
    margin-right: 0.4rem;
  }
  .settings summary {
    cursor: pointer;
    color: var(--muted);
  }
  .madefrom ul {
    margin: 0.15rem 0 0;
    padding-left: 1.1rem;
  }
  .location code {
    overflow-wrap: anywhere;
  }
  .prompt p {
    margin: 0.15rem 0 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
</style>
