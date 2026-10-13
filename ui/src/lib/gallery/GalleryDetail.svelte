<script lang="ts">
  import { Bookmark, FolderOpen, Trash2, X } from '@lucide/svelte'
  import DownloadLink from '../DownloadLink.svelte'
  import { api } from '../api'
  import { goWs } from '../router.svelte'
  import { notify } from '../toast'
  import { confirmDialog } from '../confirm.svelte'
  import { describeOutputMeta } from '../outputMeta'
  import type { GalleryFile, GalleryInput, OutputRole } from '../types'
  import RoleTag from './RoleTag.svelte'
  import { ASSET, OUTPUT, referenceName } from '../references'
  import { outputUrl } from '../apiUrls'
  import { wsHref } from '../routes'
  import { workspace } from '../workspace.svelte'
  import { formatBytes, formatMtime } from '../format'

  // The detail panel under the grid for the one file picked. The page owns
  // the pick and the metadata fetch, so a re-click refetches; this owns
  // what the panel does with them
  let {
    file,
    metadata,
    metadataLoading,
    sourceJob,
    role = null,
    inputs = [],
    onremove,
    onclose,
  }: {
    file: GalleryFile
    metadata: Record<string, unknown> | null
    metadataLoading: boolean
    sourceJob: { id: string; status: string } | null
    /** The file's role in its run; null when the run does not say */
    role?: OutputRole | null
    /** The assets and outputs the run read */
    inputs?: GalleryInput[]
    onremove: () => void
    onclose: () => void
  } = $props()

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
      const message = e instanceof Error ? e.message : String(e)
      // The server refuses an existing name rather than replacing it; the
      // choice to replace belongs to the person, not the button
      if (
        message.includes('already exists') &&
        (await confirmDialog(`${message}\n\nReplace it?`, {
          confirmLabel: 'Replace',
        }))
      ) {
        try {
          const result = await api.keepOutput(file.name, assetName, true)
          notify.success(`Kept as ${result.reference}`)
        } catch (failure) {
          notify.error(
            failure instanceof Error ? failure.message : String(failure),
          )
        }
      } else {
        notify.error(message)
      }
    }
  }

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
    <a
      href={file.url}
      target="_blank"
      class="muted"
      title="open the file itself in a new tab">open file</a
    >
    {#if file.version}
      <!-- The run this file came from, said in both the form a person is
           quoted ("version 4") and the form every tool takes (the run
           id), so the two can be checked against each other here rather
           than back in the listing -->
      <span class="num muted"
        >version {file.version} · <code>{file.run_id}</code></span
      >
    {/if}
    <span class="num muted"
      >{formatBytes(file.size)} · {formatMtime(file.mtime)}</span
    >
    <DownloadLink href={api.outputDownloadUrl(file.name)} />
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
      aria-label="close details"><X size={14} /></button
    >
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
    {#if metadata || role || inputs.length}
      <div class="meta">
        <div><RoleTag {role} /></div>
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
                      : 'open this earlier output'}>{inputName(input)}</a
                  >
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
    {:else if file.kind === 'image' && metadataLoading}
      <div class="meta muted">reading metadata…</div>
    {:else if file.kind === 'image'}
      <div class="meta muted">
        no embedded metadata - enable embed_metadata in the step's result
      </div>
    {/if}
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
