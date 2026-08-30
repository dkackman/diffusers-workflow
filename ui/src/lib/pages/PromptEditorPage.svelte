<script lang="ts">
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
  import { go } from '../router.svelte'
  import { loadPromptLibrary } from '../promptlib.svelte'
  import {
    emptyPrompt,
    knownIntendedModels,
    manifestTextFile,
    parseTags,
    presetForIntendedModel,
    workflowsReferencing,
  } from '../prompts'
  import JsonEditor from '../editor/JsonEditor.svelte'
  import type {
    EnhancerPreset,
    ModelRepo,
    PromptDefinition,
    PromptDetail,
  } from '../types'

  let { name = '' }: { name?: string } = $props()

  let doc = $state<Record<string, any>>(emptyPrompt())
  let promptDir = $state('')
  let promptFiles = $state<string[]>([])
  let promptDetails = $state<Record<string, PromptDetail>>({})
  let saveName = $state('')
  let folder = $state('')
  let newFolder = $state('')
  let status = $state('')
  let error = $state('')
  let busy = $state(false)
  let baseline = $state('')

  // Existing folders, from the listing - one level is the designed depth
  const folders = $derived(
    [
      ...new Set(
        promptFiles
          .filter((file) => file.includes('/'))
          .map((file) => file.split('/').slice(0, -1).join('/')),
      ),
    ].sort(),
  )

  type EditorView = 'form' | 'split' | 'json'
  let view = $state<EditorView>(
    (() => {
      try {
        const stored = localStorage.getItem('dw-prompt-editor-view')
        return stored === 'form' || stored === 'split' || stored === 'json'
          ? stored
          : 'form'
      } catch {
        return 'form'
      }
    })(),
  )

  function setView(next: EditorView) {
    view = next
    try {
      localStorage.setItem('dw-prompt-editor-view', next)
    } catch {
      /* session only */
    }
  }

  let jsonDraft = $state('')
  let jsonParseFailed = $state(false)

  // Mirror the prompt into the JSON surfaces; a failed parse pins the raw
  // text so a broken edit isn't regenerated out from under the user
  $effect(() => {
    const pretty = JSON.stringify($state.snapshot(doc), null, 2)
    if (!jsonParseFailed) jsonDraft = pretty
  })

  const serialized = $derived(JSON.stringify($state.snapshot(doc)))
  const dirty = $derived(baseline !== '' && serialized !== baseline)

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
          else if (event.event === 'step_start') enhanceStatus = 'generating…'
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
      enhanceResult = (await fetchOutputText(file)).trim()
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
    doc.text = enhanceResult
    doc.enhanced = { model: enhanceModel, idea }
    enhanceResult = ''
  }

  // ---------------------------------------------------------------- lifecycle

  $effect(() => {
    error = ''
    status = ''
    api
      .listPrompts()
      .then((r) => {
        promptFiles = r.prompts
        promptDir = r.prompt_dir
        promptDetails = r.details ?? {}
      })
      .catch((e) => (error = e.message))
    refreshModels()
    if (name) {
      const segments = name.split('/')
      saveName = segments[segments.length - 1]
      folder = segments.slice(0, -1).join('/')
      api
        .getPrompt(name)
        .then((definition) => {
          doc = definition as PromptDefinition
          baseline = JSON.stringify(definition)
          idea = definition.enhanced?.idea ?? ''
          preselect()
        })
        .catch((e) => (error = e.message))
    } else {
      folder = sessionStorage.getItem('dw-prompt-editor-folder') ?? ''
      sessionStorage.removeItem('dw-prompt-editor-folder')
      const imported = sessionStorage.getItem('dw-prompt-editor-import')
      let fresh = emptyPrompt() as Record<string, any>
      if (imported) {
        sessionStorage.removeItem('dw-prompt-editor-import')
        try {
          fresh = JSON.parse(imported)
          status = 'Duplicated - save under a new name'
        } catch {
          /* unreadable hand-off - stay with the blank slate */
        }
      }
      doc = fresh
      baseline = JSON.stringify(fresh)
      saveName = ''
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
    const picked = presetForIntendedModel(presets, doc.intended_model)
    if (picked) pickPreset(picked.key)
    else if (!presetKey) pickPreset(presets[0].key)
  }

  $effect(() => {
    const guard = (event: BeforeUnloadEvent) => {
      if (dirty) event.preventDefault()
    }
    window.addEventListener('beforeunload', guard)
    return () => {
      window.removeEventListener('beforeunload', guard)
    }
  })

  function onKeydown(event: KeyboardEvent) {
    if ((event.ctrlKey || event.metaKey) && event.key === 's') {
      event.preventDefault()
      save()
    }
  }

  // ---------------------------------------------------------------- editing

  function setField(key: string, value: string) {
    if (value) doc[key] = value
    else delete doc[key]
  }

  function setTags(raw: string) {
    const tags = parseTags(raw)
    if (tags.length) doc.tags = tags
    else delete doc.tags
  }

  function applyJson(raw: string) {
    jsonDraft = raw
    try {
      doc = JSON.parse(raw)
      jsonParseFailed = false
      error = ''
    } catch (e) {
      jsonParseFailed = true
      error = `JSON: ${e instanceof Error ? e.message : e}`
    }
  }

  // Pure - it renders in the save bar, so it must not touch status/error
  function savePath(): string | null {
    if (!saveName) return null
    const directory = folder === '__new__' ? newFolder.trim() : folder
    if (folder === '__new__' && !/^[\w][\w.-]*$/.test(directory)) return null
    return directory ? `${directory}/${saveName}` : saveName
  }

  // Validation failures are errors (red), not statuses (green checkmark)
  function saveBlocker(): string | null {
    if (!saveName) return 'Give the prompt a file name first'
    if (!/^[\w][\w.-]*$/.test(saveName))
      return 'Prompt names: letters, numbers, dot, dash, underscore'
    if (folder === '__new__') {
      if (!newFolder.trim()) return 'Name the new folder first'
      if (!/^[\w][\w.-]*$/.test(newFolder.trim()))
        return 'Folder names: letters, numbers, dot, dash, underscore'
    }
    if (!String(doc.text ?? '').trim())
      return 'The prompt needs text before it can be saved'
    return null
  }

  async function save() {
    status = ''
    const blocker = saveBlocker()
    if (blocker) {
      error = blocker
      return
    }
    const path = savePath()!
    busy = true
    error = ''
    try {
      const result = await api.savePrompt(
        path,
        $state.snapshot(doc) as PromptDefinition,
      )
      if (folder === '__new__') {
        folder = newFolder.trim()
        newFolder = ''
      }
      // Same as the workflow editor: the picker lists folders from the
      // listing, so a newly created one must be added or the select resets
      if (!promptFiles.includes(path)) promptFiles = [...promptFiles, path]
      baseline = JSON.stringify($state.snapshot(doc))
      status = `Saved to ${result.path}`
      loadPromptLibrary()
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    } finally {
      busy = false
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
      !window.confirm(
        `Delete ${name}.json? This removes the file on disk.${warning}`,
      )
    )
      return
    try {
      await api.deletePrompt(name)
      loadPromptLibrary()
      go('prompts')
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    }
  }

  function duplicate() {
    sessionStorage.setItem(
      'dw-prompt-editor-import',
      JSON.stringify($state.snapshot(doc)),
    )
    if (folder && folder !== '__new__')
      sessionStorage.setItem('dw-prompt-editor-folder', folder)
    go('prompt-edit')
  }
</script>

<svelte:window onkeydown={onKeydown} />

<datalist id="intended-models">
  {#each intendedModels as model (model)}<option value={model}></option>{/each}
</datalist>
<datalist id="enhancer-models">
  {#each preset?.models ?? [] as model (model)}<option value={model}
    ></option>{/each}
</datalist>

<div class="head">
  <a href="#/prompts" class="muted">← prompts</a>
  <h1>{name || 'New prompt'}</h1>
  <span class="flex"></span>
  <div class="viewswitch" role="group" aria-label="editor view">
    <button
      class="quiet withicon"
      class:activebtn={view === 'form'}
      onclick={() => setView('form')}
      title="edit with a form"
    >
      <LayoutList size={14} />form
    </button>
    <button
      class="quiet withicon"
      class:activebtn={view === 'split'}
      onclick={() => setView('split')}
      title="form beside the JSON - both editable, blur applies"
    >
      <Columns2 size={14} />split
    </button>
    <button
      class="quiet withicon"
      class:activebtn={view === 'json'}
      onclick={() => setView('json')}
      title="edit the raw JSON, schema-aware"
    >
      <Braces size={14} />JSON
    </button>
  </div>
  {#if name}
    <button
      class="quiet withicon"
      onclick={duplicate}
      title="open a copy in the editor to save under a new name"
    >
      <Copy size={14} />Duplicate
    </button>
    <button
      class="quiet withicon danger"
      onclick={remove}
      title="delete this prompt from the library"
    >
      <Trash2 size={14} />Delete
    </button>
  {/if}
  <button
    class="withicon"
    class:dirtybtn={dirty}
    onclick={save}
    disabled={busy}
    title="write to the prompt directory under the name below (Ctrl+S)"
  >
    <Save size={14} />Save{#if dirty}<span class="dirtydot"></span>{/if}
  </button>
</div>

<div class="savebar muted">
  saving as
  <select class="folderpick" bind:value={folder} title="folder to save into">
    <option value="">(root)</option>
    {#each folders as existing (existing)}<option value={existing}
        >{existing}/</option
      >{/each}
    <option value="__new__">new folder…</option>
  </select>
  {#if folder === '__new__'}
    <input
      class="newfolder"
      bind:value={newFolder}
      placeholder="folder name"
      title="name for the new folder at the root of the prompt directory"
    />
    <span>/</span>
  {/if}
  <input class="savename" bind:value={saveName} placeholder="MyPrompt" />
  <span>.json in {promptDir}</span>
  <span class="flex"></span>
  {#if savePath()}
    <code class="refhint" title="use the stored prompt from any workflow"
      >prompt:{savePath()}</code
    >
  {/if}
</div>

{#if error}<p class="error">{error}</p>{/if}
{#if status}
  <p class="status"><CircleCheck size={14} />{status}</p>
{/if}

{#if view === 'json'}
  <JsonEditor
    value={jsonDraft}
    onchange={applyJson}
    height="560px"
    schema="prompt"
  />
  <p class="muted hint">
    Schema-aware: completion, hover docs and validation come from the prompt
    schema. Changes apply when the editor loses focus.
  </p>
{:else}
  <div class="editwrap" class:splitcols={view === 'split'}>
    <div class="formcol">
      <div class="panel">
        <h2>Prompt</h2>
        <label class="fieldlabel" for="prompt-text">text</label>
        <textarea
          id="prompt-text"
          class="prompttext"
          rows="8"
          spellcheck="true"
          value={doc.text ?? ''}
          placeholder="the prompt itself - what prompt:{savePath() ??
            'name'} resolves to"
          onchange={(e) => (doc.text = e.currentTarget.value)}></textarea>
        <label class="fieldlabel" for="prompt-negative">negative prompt</label>
        <textarea
          id="prompt-negative"
          rows="2"
          spellcheck="true"
          value={doc.negative_prompt ?? ''}
          placeholder="optional - for models that take one"
          onchange={(e) => setField('negative_prompt', e.currentTarget.value)}
        ></textarea>
        <div class="metagrid">
          <label for="prompt-desc">description</label>
          <input
            id="prompt-desc"
            spellcheck="true"
            value={doc.description ?? ''}
            placeholder="shown on the prompt's library card"
            onchange={(e) => setField('description', e.currentTarget.value)}
          />
          <label for="prompt-model">intended model</label>
          <input
            id="prompt-model"
            list="intended-models"
            value={doc.intended_model ?? ''}
            placeholder="e.g. minimax-h3 - badges the card, preselects the enhancer"
            onchange={(e) => {
              setField('intended_model', e.currentTarget.value)
              preselect()
            }}
          />
          <label for="prompt-tags">tags</label>
          <input
            id="prompt-tags"
            value={(doc.tags ?? []).join(', ')}
            placeholder="comma-separated, for filtering the library"
            onchange={(e) => setTags(e.currentTarget.value)}
          />
        </div>
        {#if doc.enhanced?.model}
          <p class="muted provenance" title={doc.enhanced.idea}>
            <Sparkles size={13} /> enhanced by {doc.enhanced.model}
          </p>
        {/if}
      </div>

      <div class="panel">
        <h2><Sparkles size={15} /> Enhance with AI</h2>
        <p class="muted hint">
          Expand an idea into a full prompt with a local language model. Runs as
          an ordinary job - it waits its turn behind anything generating.
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
            <input
              id="enhance-model"
              list="enhancer-models"
              bind:value={enhanceModel}
              placeholder="Hugging Face repo id"
            />
            {#if enhanceModel}
              {#if modelCached}
                <span class="chip good" title="already in the local model cache"
                  >cached</span
                >
              {:else if downloading}
                <span class="chip" title="downloading to the local model cache"
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
    {#if view === 'split'}
      <div class="jsoncol">
        <JsonEditor
          value={jsonDraft}
          onchange={applyJson}
          height="calc(100vh - 200px)"
          schema="prompt"
        />
      </div>
    {/if}
  </div>
{/if}

<style>
  .head {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.4rem;
  }
  .head h1 {
    font-size: 1.1rem;
    margin: 0;
  }
  .flex {
    flex: 1;
  }
  .withicon {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  .savebar {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
  }
  .savename {
    max-width: 200px;
  }
  .folderpick {
    max-width: 160px;
  }
  .newfolder {
    max-width: 140px;
  }
  .refhint {
    font-size: 0.8rem;
    color: var(--accent);
  }
  .panel {
    margin-bottom: 1rem;
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
    grid-template-columns: minmax(120px, auto) 1fr;
    gap: 0.5rem 0.8rem;
    align-items: center;
    margin-top: 0.6rem;
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
  .modelrow input {
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
  .editwrap.splitcols {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(360px, 44%);
    gap: 1.1rem;
    align-items: start;
  }
  .jsoncol {
    position: sticky;
    top: 66px;
  }
  @media (max-width: 1100px) {
    .editwrap.splitcols {
      grid-template-columns: 1fr;
    }
    .jsoncol {
      position: static;
    }
  }
  .viewswitch {
    display: inline-flex;
  }
  .viewswitch button {
    border-radius: 0;
  }
  .viewswitch button:first-child {
    border-radius: 6px 0 0 6px;
  }
  .viewswitch button:last-child {
    border-radius: 0 6px 6px 0;
  }
  .viewswitch button + button {
    margin-left: -1px;
  }
  .activebtn {
    border-color: var(--accent);
    color: var(--accent);
    position: relative;
    z-index: 1;
  }
  .status {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--good);
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
