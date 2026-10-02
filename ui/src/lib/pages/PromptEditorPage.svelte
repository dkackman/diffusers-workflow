<script lang="ts">
  import Suggest from '../ui/Suggest.svelte'
  import { isNameSegment } from '../names'
  import { DocumentEditor, NEW_FOLDER } from '../editorShell.svelte'
  import ViewSwitch from '../editor/ViewSwitch.svelte'
  import FolderPicker from '../editor/FolderPicker.svelte'
  import EditorBody from '../editor/EditorBody.svelte'
  import { PROMPT, reference } from '../references'
  import {
    Braces,
    CircleCheck,
    Columns2,
    Copy,
    Download,
    LayoutList,
    Save,
    Sparkles,
    Trash2,
  } from '@lucide/svelte'
  import { api, fetchOutputText, streamJobEvents } from '../api'
  import { writableRoot } from '../libraries'
  import DownloadLink from '../DownloadLink.svelte'
  import { go } from '../router.svelte'
  import { sharedHref } from '../routes'
  import { phaseLabel } from '../progress'
  import { notify } from '../toast'
  import { confirmDialog } from '../confirm.svelte'
  import { loadPromptLibrary } from '../promptlib.svelte'
  import { groupOf, leafOf } from '../grouping'
  import {
    emptyPrompt,
    knownIntendedModels,
    manifestTextFile,
    parseTags,
    presetForIntendedModel,
    workflowsReferencing,
  } from '../prompts'
  import CopyButton from '../CopyButton.svelte'
  import type {
    EnhancerPreset,
    ModelRepo,
    PromptDefinition,
    PromptDetail,
  } from '../types'

  let { name = '' }: { name?: string } = $props()

  const shell = new DocumentEditor({
    viewKey: 'dw-prompt-editor-view',
    views: ['form', 'split', 'json'],
  })
  shell.doc = emptyPrompt()
  let promptDir = $state('')
  let promptFiles = $state<string[]>([])
  let promptDetails = $state<Record<string, PromptDetail>>({})
  // A prompt from a read-only examples library: it can be edited and saved
  // (the save lands in this workspace's library, shadowing it) but not
  // deleted, the way a read-only workflow behaves
  let readOnly = $state(false)

  // Existing folders, from the listing - one level is the designed depth
  const folders = $derived(
    [...new Set(promptFiles.map(groupOf).filter(Boolean))].sort(),
  )

  const VIEWS = [
    {
      view: 'form',
      label: 'form',
      title: 'edit with a form',
      icon: LayoutList,
    },
    {
      view: 'split',
      label: 'split',
      title: 'form beside the JSON - both editable, blur applies',
      icon: Columns2,
    },
    {
      view: 'json',
      label: 'JSON',
      title: 'edit the raw JSON, schema-aware',
      icon: Braces,
    },
  ] as const

  // ------------------------------------------------------------- enhancer

  let presets = $state<EnhancerPreset[]>([])
  let presetKey = $state('')
  let enhanceModel = $state('')
  let idea = $state('')
  let device = $state('')
  let models = $state<ModelRepo[]>([])
  let downloading = $state(false)
  let enhanceBusy = $state(false)
  let enhanceStatus = $state('')
  let enhanceError = $state('')
  let enhanceResult = $state('')
  let enhanceJobId = $state('')
  let enhancersDown = $state(false)
  let stopStream: (() => void) | null = null

  const preset = $derived(presets.find((p) => p.key === presetKey))
  const modelCached = $derived(
    models.some((repo) => repo.repo_id === enhanceModel),
  )
  const intendedModels = $derived(
    knownIntendedModels(
      presets,
      Object.values(promptDetails).map((detail) => detail.intended_model),
    ),
  )

  // Switching preset resets the model to that preset's default; re-picking
  // the current one leaves a hand-typed model alone
  function pickPreset(key: string) {
    const picked = presets.find((p) => p.key === key)
    if (!picked || key === presetKey) return
    presetKey = key
    enhanceModel = picked.default_model
  }

  async function refreshModels() {
    try {
      models = (await api.listModels()).repos
    } catch {
      /* cache indicator stays pessimistic */
    }
  }

  async function downloadModel() {
    if (!enhanceModel) return
    downloading = true
    enhanceError = ''
    try {
      await api.startDownload(enhanceModel)
      // Poll until this repo's download leaves the active list
      while (downloading) {
        await new Promise((resolve) => setTimeout(resolve, 2000))
        const { downloads } = await api.listDownloads()
        const mine = downloads.find((d) => d.repo_id === enhanceModel)
        if (!mine || mine.status !== 'downloading') {
          if (mine?.status === 'failed')
            enhanceError = mine.error ?? 'download failed'
          break
        }
      }
    } catch (e) {
      enhanceError = e instanceof Error ? e.message : String(e)
    } finally {
      downloading = false
      refreshModels()
    }
  }

  async function generate() {
    if (!idea.trim()) {
      enhanceError = 'Describe the idea to expand first'
      return
    }
    enhanceBusy = true
    enhanceError = ''
    enhanceResult = ''
    enhanceStatus = 'queueing…'
    try {
      const job = await api.enhance({
        idea,
        preset: presetKey,
        model_name: enhanceModel || undefined,
        device: device || undefined,
      })
      enhanceJobId = job.id
      if (job.queue_position !== undefined) {
        enhanceStatus = `queued · #${job.queue_position + 1} in line`
      }
      stopStream = streamJobEvents(
        job.id,
        -1,
        (event) => {
          if (event.event === 'log') enhanceStatus = String(event.message)
          // The enhancer is one task step - its phase is the whole story,
          // and 'loading' is most of the wait on a cold model
          else if (event.event === 'phase')
            enhanceStatus = phaseLabel(event) + '…'
          else if (event.event === 'job_status')
            enhanceStatus = String(event.status)
        },
        () => finishEnhance(job.id),
      )
    } catch (e) {
      enhanceError = e instanceof Error ? e.message : String(e)
      enhanceBusy = false
      enhanceStatus = ''
    }
  }

  async function finishEnhance(jobId: string) {
    stopStream = null
    try {
      const detail = await api.getJob(jobId)
      if (detail.status !== 'succeeded') {
        enhanceError = detail.error ?? `enhancement ${detail.status}`
        return
      }
      const file = manifestTextFile(detail.manifest)
      if (!file) {
        enhanceError = 'The enhancement produced no text'
        return
      }
      enhanceResult = (await fetchOutputText(file, detail.workspace)).trim()
    } catch (e) {
      enhanceError = e instanceof Error ? e.message : String(e)
    } finally {
      enhanceBusy = false
      enhanceStatus = ''
      enhanceJobId = ''
    }
  }

  async function cancelEnhance() {
    if (!enhanceJobId) return
    try {
      await api.cancelJob(enhanceJobId)
    } catch {
      /* already finished */
    }
  }

  function useResult() {
    shell.doc.text = enhanceResult
    shell.doc.enhanced = { model: enhanceModel, idea }
    enhanceResult = ''
  }

  // ---------------------------------------------------------------- lifecycle

  $effect(() => {
    api
      .listPrompts()
      .then((r) => {
        promptFiles = r.prompts
        promptDir = writableRoot(r.libraries)
        promptDetails = r.details ?? {}
      })
      .catch((e) => notify.error(e.message))
    refreshModels()
    if (name) {
      shell.saveName = leafOf(name)
      shell.folder = groupOf(name)
      api
        .getPrompt(name)
        // the definition comes back beside where it was found: a prompt
        // from a read-only examples library can be edited and saved (the
        // copy lands here) but not deleted
        .then(({ prompt, writable }) => {
          readOnly = !writable
          shell.load(prompt)
          idea = prompt.enhanced?.idea ?? ''
          preselect()
        })
        .catch((e) => notify.error(e.message))
    } else {
      const imported = shell.takeImport(
        'dw-prompt-editor-import',
        'dw-prompt-editor-folder',
      )
      if (imported) notify.success('Duplicated - save under a new name')
      shell.load(imported ?? emptyPrompt())
      shell.saveName = ''
    }
    api
      .listEnhancers()
      .then((r) => {
        presets = r.presets
        enhancersDown = presets.length === 0
        preselect()
      })
      .catch(() => {
        // Editing still works without the enhancer - but say so, rather
        // than leaving a permanently disabled Generate button unexplained
        enhancersDown = true
      })
    return () => {
      stopStream?.()
      stopStream = null
      downloading = false
    }
  })

  function preselect() {
    if (!presets.length) return
    // A matching intended model picks its preset; with no match the current
    // selection stands, so an ltx-2 prompt doesn't get the H3 enhancer
    const picked = presetForIntendedModel(presets, shell.doc.intended_model)
    if (picked) pickPreset(picked.key)
    else if (!presetKey) pickPreset(presets[0].key)
  }

  function onKeydown(event: KeyboardEvent) {
    if ((event.ctrlKey || event.metaKey) && event.key === 's') {
      event.preventDefault()
      save()
    }
  }

  // ---------------------------------------------------------------- editing

  function setField(key: string, value: string) {
    if (value) shell.doc[key] = value
    else delete shell.doc[key]
  }

  function setTags(raw: string) {
    const tags = parseTags(raw)
    if (tags.length) shell.doc.tags = tags
    else delete shell.doc.tags
  }

  // Validation failures are errors (red), not statuses (green checkmark)
  function saveBlocker(): string | null {
    if (!shell.saveName) return 'Give the prompt a file name first'
    if (!isNameSegment(shell.saveName))
      return 'Prompt names: letters, numbers, dot, dash, underscore'
    if (shell.folder === NEW_FOLDER) {
      if (!shell.directory()) return 'Name the new folder first'
      if (!isNameSegment(shell.directory()))
        return 'Folder names: letters, numbers, dot, dash, underscore'
    }
    if (!String(shell.doc.text ?? '').trim())
      return 'The prompt needs text before it can be saved'
    return null
  }

  async function save() {
    const blocker = saveBlocker()
    if (blocker) {
      notify.error(blocker)
      return
    }
    const path = shell.savePath()!
    shell.busy = true
    try {
      await api.savePrompt(path, $state.snapshot(shell.doc) as PromptDefinition)
      shell.commitNewFolder()
      // Same as the workflow editor: the picker lists folders from the
      // listing, so a newly created one must be added or the select resets
      if (!promptFiles.includes(path)) promptFiles = [...promptFiles, path]
      shell.markSaved()
      notify.success(`Saved to ${path}`)
      loadPromptLibrary()
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    } finally {
      shell.busy = false
    }
  }

  async function remove() {
    if (!name) return
    // Say what the delete is about to break: any workflow holding a
    // prompt:name reference fails at run time once the file is gone
    let warning = ''
    try {
      const { details } = await api.listWorkflows()
      const referencing = workflowsReferencing(name, details ?? {})
      if (referencing.length) {
        const shown = referencing.slice(0, 5).join(', ')
        const more = referencing.length > 5 ? ', …' : ''
        warning =
          `\n\nReferenced by ${referencing.length} workflow` +
          `${referencing.length === 1 ? '' : 's'} (${shown}${more}) - ` +
          'they will fail at run time until updated.'
      }
    } catch {
      /* the confirm still protects the file itself */
    }
    if (
      !(await confirmDialog(
        `Delete ${name}.json? This removes the file on disk.${warning}`,
        { confirmLabel: 'Delete' },
      ))
    )
      return
    try {
      await api.deletePrompt(name)
      loadPromptLibrary()
      go('shared', 'prompts')
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    }
  }

  function duplicate() {
    sessionStorage.setItem(
      'dw-prompt-editor-import',
      JSON.stringify($state.snapshot(shell.doc)),
    )
    if (shell.folder && shell.folder !== NEW_FOLDER)
      sessionStorage.setItem('dw-prompt-editor-folder', shell.folder)
    go('shared', 'prompt-edit')
  }
</script>

<svelte:window onkeydown={onKeydown} />

<div class="head">
  <a href={sharedHref('prompts')} class="muted">← prompts</a>
  <h1>{name || 'New prompt'}</h1>
  <span class="flex"></span>
  <ViewSwitch
    view={shell.view}
    options={[...VIEWS]}
    onselect={(view) => shell.setView(view)}
  />
  {#if name}
    <button
      class="quiet withicon"
      onclick={duplicate}
      title="open a copy in the editor to save under a new name"
    >
      <Copy size={14} />Duplicate
    </button>
    <DownloadLink href={api.promptDownloadUrl(name)} />
    {#if !readOnly}
      <button
        class="quiet withicon danger"
        onclick={remove}
        title="delete this prompt from the library"
      >
        <Trash2 size={14} />Delete
      </button>
    {/if}
  {/if}
  <button
    class="withicon"
    class:dirtybtn={shell.dirty}
    onclick={save}
    disabled={shell.busy}
    title="write to the prompt directory under the name below (Ctrl+S)"
  >
    <Save size={14} />Save{#if shell.dirty}<span class="dirtydot"></span>{/if}
  </button>
</div>

<div class="savebar muted">
  saving as
  <FolderPicker
    bind:folder={shell.folder}
    bind:newFolder={shell.newFolder}
    {folders}
    newFolderTitle="name for the new folder at the root of the prompt directory"
  />
  {#if shell.folder === NEW_FOLDER}<span>/</span>{/if}
  <input class="savename" bind:value={shell.saveName} placeholder="MyPrompt" />
  <span class="dirhint">.json in {promptDir}</span>
  <span class="flex"></span>
  {#if shell.savePath()}
    <code class="refhint" title="use the stored prompt from any workflow"
      >prompt:{shell.savePath()}</code
    >
    <CopyButton
      text={reference(PROMPT, shell.savePath() ?? '')}
      title="copy reference to clipboard"
    />
  {/if}
</div>

<EditorBody
  view={shell.view}
  jsonDraft={shell.jsonDraft}
  onjson={(raw) => shell.applyJson(raw)}
  schema="prompt"
  hint="Schema-aware: completion, hover docs and validation come from the prompt schema. Changes apply when the editor loses focus."
  stickyTop="66px"
>
  {#snippet form()}
    <!-- Queried by .panelgrid below, so the panels stack on their own
         column's width - the split view narrows the form well before the
         viewport -->
    <div class="promptform">
      <div class="panelgrid">
        <div class="panel">
          <h2>Prompt</h2>
          <label class="fieldlabel" for="prompt-text">text</label>
          <textarea
            id="prompt-text"
            class="prompttext"
            rows="8"
            spellcheck="true"
            value={shell.doc.text ?? ''}
            placeholder="the prompt itself - what prompt:{shell.savePath() ??
              'name'} resolves to"
            onchange={(e) => (shell.doc.text = e.currentTarget.value)}
          ></textarea>
          <label class="fieldlabel" for="prompt-negative">negative prompt</label
          >
          <textarea
            id="prompt-negative"
            rows="2"
            spellcheck="true"
            value={shell.doc.negative_prompt ?? ''}
            placeholder="optional - for models that take one"
            onchange={(e) => setField('negative_prompt', e.currentTarget.value)}
          ></textarea>
          <div class="metagrid">
            <label for="prompt-desc">description</label>
            <input
              id="prompt-desc"
              spellcheck="true"
              value={shell.doc.description ?? ''}
              placeholder="shown on the prompt's library card"
              onchange={(e) => setField('description', e.currentTarget.value)}
            />
            <label for="prompt-model">intended model</label>
            <Suggest
              id="prompt-model"
              suggestions={intendedModels}
              value={shell.doc.intended_model ?? ''}
              placeholder="e.g. minimax-h3 - badges the card, preselects the enhancer"
              onchange={(value) => {
                setField('intended_model', value)
                preselect()
              }}
            />
            <label for="prompt-tags">tags</label>
            <input
              id="prompt-tags"
              value={(shell.doc.tags ?? []).join(', ')}
              placeholder="comma-separated, for filtering the library"
              onchange={(e) => setTags(e.currentTarget.value)}
            />
          </div>
          {#if shell.doc.enhanced?.model}
            <p class="muted provenance" title={shell.doc.enhanced.idea}>
              <Sparkles size={13} /> enhanced by {shell.doc.enhanced.model}
            </p>
          {/if}
        </div>

        <div class="panel">
          <h2><Sparkles size={15} /> Enhance with AI</h2>
          <p class="muted hint">
            Expand an idea into a full prompt with a local language model. Runs
            as an ordinary job - it waits its turn behind anything generating.
          </p>
          {#if enhancersDown}
            <p class="muted hint">
              Enhancement is unavailable - the server reported no enhancer
              presets. Editing and saving still work.
            </p>
          {/if}
          <div class="metagrid">
            <label for="enhance-preset">preset</label>
            <select
              id="enhance-preset"
              value={presetKey}
              onchange={(e) => pickPreset(e.currentTarget.value)}
            >
              {#each presets as p (p.key)}
                <option value={p.key}>{p.label}</option>
              {/each}
            </select>
            <label for="enhance-model">model</label>
            <span class="modelrow">
              <Suggest
                id="enhance-model"
                suggestions={preset?.models ?? []}
                bind:value={enhanceModel}
                placeholder="Hugging Face repo id"
              />
              {#if enhanceModel}
                {#if modelCached}
                  <span
                    class="chip good"
                    title="already in the local model cache">cached</span
                  >
                {:else if downloading}
                  <span
                    class="chip"
                    title="downloading to the local model cache"
                    >downloading…</span
                  >
                {:else}
                  <button
                    class="quiet withicon"
                    onclick={downloadModel}
                    title="download this model to the local cache now - otherwise the first enhancement downloads it"
                  >
                    <Download size={13} />get
                  </button>
                {/if}
              {/if}
            </span>
            <label for="enhance-device">device</label>
            <select
              id="enhance-device"
              bind:value={device}
              title="where the language model runs - cpu keeps VRAM free for generation"
            >
              <option value="">preset default (cpu)</option>
              <option value="cuda">cuda</option>
              <option value="mps">mps</option>
              <option value="cpu">cpu</option>
            </select>
            <label for="enhance-idea">idea</label>
            <textarea
              id="enhance-idea"
              rows="3"
              spellcheck="true"
              bind:value={idea}
              placeholder={preset?.placeholder ?? 'describe what to generate'}
            ></textarea>
          </div>
          <div class="enhanceactions">
            <button
              class="withicon"
              onclick={generate}
              disabled={enhanceBusy || !presets.length}
              title="expand the idea with the selected model"
            >
              <Sparkles size={14} />Generate
            </button>
            {#if enhanceBusy && enhanceJobId}
              <button class="quiet" onclick={cancelEnhance}>cancel</button>
              <a class="muted joblink" href={'#/jobs/' + enhanceJobId}
                >watch job</a
              >
            {/if}
            {#if enhanceStatus}
              <span class="muted enhancestatus"
                ><span class="pulse-dot"></span>{enhanceStatus}</span
              >
            {/if}
          </div>
          {#if enhanceError}<p class="error">{enhanceError}</p>{/if}
          {#if enhanceResult}
            <textarea class="resultbox" rows="8" readonly value={enhanceResult}
            ></textarea>
            <div class="enhanceactions">
              <button class="withicon" onclick={useResult}>
                <CircleCheck size={14} />Use as prompt text
              </button>
              <button class="quiet" onclick={() => (enhanceResult = '')}>
                discard
              </button>
            </div>
          {/if}
        </div>
      </div>
    </div>
  {/snippet}
</EditorBody>

<style>
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.6rem;
    margin-bottom: 0.4rem;
  }
  .head h1 {
    font-size: 1.1rem;
    margin: 0;
  }
  .savebar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.4rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
  }
  .savename {
    max-width: 200px;
  }
  /* a directory is one unbroken token, so let it wrap anywhere rather than
     push the bar past a phone-width viewport */
  .dirhint {
    min-width: 0;
    overflow-wrap: anywhere;
  }
  .refhint {
    font-size: 0.8rem;
    color: var(--accent);
  }
  .promptform {
    container-type: inline-size;
  }
  .panelgrid {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 1.1rem;
    align-items: start;
  }
  @container (max-width: 900px) {
    .panelgrid {
      grid-template-columns: 1fr;
    }
  }
  .panel {
    margin-bottom: 1rem;
    min-width: 0;
  }
  .panel h2 {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .fieldlabel {
    display: block;
    font-weight: 600;
    color: var(--muted);
    font-size: 0.85rem;
    margin: 0.6rem 0 0.25rem;
  }
  textarea {
    width: 100%;
    resize: vertical;
  }
  .prompttext {
    font-size: 0.95rem;
  }
  .metagrid {
    display: grid;
    grid-template-columns: minmax(120px, 40%) minmax(0, 1fr);
    gap: 0.5rem 0.8rem;
    align-items: center;
    margin-top: 0.6rem;
  }
  @container (max-width: 420px) {
    .metagrid {
      grid-template-columns: minmax(0, 1fr);
      gap: 0.2rem;
    }
  }
  .metagrid label {
    font-weight: 600;
    color: var(--muted);
    font-size: 0.85rem;
  }
  .metagrid textarea {
    align-self: stretch;
  }
  .modelrow {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .modelrow :global(input) {
    flex: 1;
  }
  .chip.good {
    color: var(--good);
    border-color: var(--good);
  }
  .provenance {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.8rem;
    margin: 0.6rem 0 0;
  }
  .enhanceactions {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-top: 0.7rem;
  }
  .enhancestatus {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.85rem;
  }
  .joblink {
    font-size: 0.85rem;
  }
  .resultbox {
    margin-top: 0.7rem;
    font-size: 0.9rem;
  }
  .dirtydot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--warn);
    margin-left: 0.15rem;
  }
  .error {
    color: var(--bad);
  }
  .hint {
    font-size: 0.8rem;
  }
</style>
