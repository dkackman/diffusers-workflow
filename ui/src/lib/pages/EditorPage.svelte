<script lang="ts">
  import { untrack } from 'svelte'
  import { editorLists, loadEditorLists } from '../editorLists.svelte'
  import {
    DocumentEditor,
    EDITOR_VIEWS,
    NEW_FOLDER,
    editorKeys,
    type EditorView,
  } from '../editorShell.svelte'
  import Segmented from '../ui/Segmented.svelte'
  import EditorBody from '../editor/EditorBody.svelte'
  import StepList from '../editor/StepList.svelte'
  import { StepModes } from '../stepModes.svelte'
  import ValidationPanel from '../editor/ValidationPanel.svelte'
  import { EditorJumps, editorCheck } from '../editor/editorCheck.svelte'
  import DocumentFileBar from '../editor/DocumentFileBar.svelte'
  import WorkflowOrigin, { IMPORTED } from '../editor/WorkflowOrigin.svelte'
  import { CircleCheck, Play, Save } from '@lucide/svelte'
  import { api } from '../api'
  import { ApiError } from '../apiErrors'
  import {
    openWorkflow,
    listedWorkflows,
    startingWorkflow,
    validatedRun,
    validatedSave,
    watchWorkflow,
    withSaved,
  } from '../editor/workflowFile'
  import { resource } from '../resource.svelte'
  import LoadFailed from '../ui/LoadFailed.svelte'
  import ErrorLine from '../ui/ErrorLine.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import { notify } from '../toast'
  import { confirmDialog } from '../confirm.svelte'
  import { action } from '../action.svelte'
  import { go, goWs, renameRoute } from '../router.svelte'
  import { SECTION_LABELS, wsHref } from '../routes'
  import { loadWorkspaces, workspace } from '../workspace.svelte'
  import {
    attentionBackendVariables,
    constraintHints,
    emptyWorkflow,
  } from '../editor'
  import { referenceSuggestions } from '../referenceSuggestions'
  import { danglingReferenceDetails, flowGraph } from '../flow'
  import { groupOf, leafOf } from '../grouping'
  import { loadPromptLibrary, promptLibrary } from '../promptlib.svelte'
  import VariablesForm from '../editor/VariablesForm.svelte'
  import FlowView from '../editor/FlowView.svelte'
  import type { WorkflowDefinition } from '../types'

  let { name = '' }: { name?: string } = $props()

  const ed = new DocumentEditor({
    viewKey: 'dw-editor-view',
    views: ['form', 'split', 'json', 'flow'],
    // migrate the old boolean split preference
    legacyView: () =>
      localStorage.getItem('dw-editor-split') === '1' ? 'split' : null,
    label: (): string => ed.savePath() ?? 'this workflow',
  })
  watchWorkflow(ed)
  ed.doc = emptyWorkflow()
  // The file fields collapse behind the path chip once the workflow has a
  // name - they are settings, not something you retouch on every edit
  let fileOpen = $state(false)
  // The path being edited: the route's, until a save gives the document
  // another (or its first) - the URL follows a beat later
  const current = $derived(
    ed.loadState === 'loading' || ed.loadState === 'failed'
      ? name
      : (ed.loadedPath ?? name),
  )
  // Which workspace and file the document on screen came from or went to,
  // so the route catching up with a save does not read it again
  let opened = ''

  const modesKey = $derived(`step-modes:${current || '(unsaved)'}`)
  // Keyed by step name; existing steps open compact, new ones full
  const modes = new StepModes(() => modesKey)
  const jumps = new EditorJumps(ed, modes)
  // The server's copy replaced the steps: each keeps its mode by name
  $effect(() => {
    if (!ed.remoteLoads) return
    untrack(() =>
      modes.retain(
        ((ed.doc.steps as Record<string, any>[]) ?? []).map((s) => s?.name),
      ),
    )
  })

  // Existing folders, from the listing - one level is the designed depth,
  // but any deeper directories that exist still appear and keep working
  const folders = $derived(
    [...new Set(editorLists.workflowFiles.map(groupOf).filter(Boolean))].sort(),
  )
  // The verdict shown: the live check's, or the Validate button's
  const check = editorCheck(ed)
  // A new workflow's starting point: '' for blank, an example's name, or
  // IMPORTED for a definition handed over from an image's metadata
  let startFrom = $state('')
  // The listing: the folder picker's files, where a new file saves, and the
  // read-only examples a new workflow can start from (workflowFile.ts)
  const catalog = resource(() => api.listWorkflows(), { watch: ['workflows'] })
  const listed = $derived(listedWorkflows(catalog.data))
  $effect(() => {
    if (catalog.data) editorLists.workflowFiles = listed.files
  })
  // Beside the definition, a hand-off carries the draft's file name and
  // what it started from, so moving it to another workspace keeps both
  const NAME_KEY = 'dw-editor-import-name'
  const FROM_KEY = 'dw-editor-import-from'
  // the workspace picker's names - outside the load effect, so the listing
  // landing cannot re-run it and reset the draft
  loadWorkspaces()

  // The way out: a saved workflow's own page, else the Workflows list
  const back = $derived({
    href: wsHref(
      workspace.current,
      'workflows',
      ...current.split('/').filter(Boolean),
    ),
    label: current || SECTION_LABELS.ws.workflows,
  })

  // The data-flow graph and per-step reference problems drive the rail's
  // producer/consumer chips and inline warnings - promptLibrary.names stays
  // undefined until the listing lands, so a missing library must not flag
  // every prompt: reference as dangling
  const flow = $derived(flowGraph($state.snapshot(ed.doc)))
  const problemsByStep = $derived.by(() => {
    const details = danglingReferenceDetails(
      $state.snapshot(ed.doc),
      promptLibrary.names,
    )
    const grouped: Record<number, string[]> = {}
    for (const d of details) {
      grouped[d.stepIndex] = [...(grouped[d.stepIndex] ?? []), d.message]
    }
    return new Map(
      Object.entries(grouped).map(([index, messages]) => [
        Number(index),
        messages,
      ]),
    )
  })
  // Memoized so datalist options keep stable DOM identity - churn on every
  // render made the browser's suggestion dropdown flaky on first focus
  const stepReferences = $derived.by(() => {
    const snapshot = $state.snapshot(ed.doc)
    return ((snapshot.steps as unknown[]) ?? []).map((_, index) =>
      referenceSuggestions(snapshot, index, promptLibrary.names ?? []),
    )
  })

  $effect(() => {
    // A new draft moved to another workspace stays on this page; the
    // workspace is read here so the move reloads the listing and takes the
    // draft back up from the hand-off
    void workspace.current
    loadEditorLists()
    loadPromptLibrary()
    // the route catching up with a save: this is the document already
    if (name && opened === `${workspace.current}:${name}`) return
    const path = name
    // Only the workspace and the route start a load. Untracked, because
    // what a load reads on the way - the step modes' key follows the
    // editor's own path - would otherwise re-run this when a first save
    // names the document, and start the draft over the one just saved
    untrack(() => {
      if (path) {
        ed.saveName = leafOf(path)
        ed.folder = groupOf(path)
        fileOpen = false
        open(path)
      } else {
        startDraft()
      }
    })
  })

  /** A new workflow: a hand-off's definition, else the blank starter. */
  function startDraft() {
    opened = ''
    // A gallery "open as workflow" hands the definition over in
    // sessionStorage - one-shot, so a plain "New" stays a blank slate
    const imported = ed.takeImport('dw-editor-import', 'dw-editor-folder')
    // Built as a plain local object and assigned once: reading the
    // workflow proxy here would subscribe this effect to every edit and
    // re-fire all the listing calls on each keystroke
    const fresh = imported ?? (emptyWorkflow() as Record<string, any>)
    const from = sessionStorage.getItem(FROM_KEY)
    startFrom = from ?? (imported ? IMPORTED : '')
    if (imported && from === null)
      notify.success('Imported from image metadata')
    ed.startNew(fresh)
    ed.saveName = sessionStorage.getItem(NAME_KEY) ?? ''
    sessionStorage.removeItem(FROM_KEY)
    sessionStorage.removeItem(NAME_KEY)
    // A new workflow has nowhere to save to yet - show the fields
    fileOpen = true
    if (imported) modes.restore()
    else modes.reset({ [fresh.steps[0].name]: 'full' })
  }

  /** Read `path` into the editor; a failure leaves nothing to save. */
  async function open(path: string) {
    const where = `${workspace.current}:${path}`
    // a new read has begun: until it lands nothing is opened, so a return
    // to the previous file reads it again instead of trusting the key
    opened = ''
    if (!(await openWorkflow(ed, path, () => path !== name))) return
    opened = where
    modes.restore()
  }

  /** The missing file `name`, begun as a blank workflow. */
  function startMissing() {
    const fresh = emptyWorkflow() as Record<string, any>
    ed.startNew(fresh)
    fileOpen = true
    modes.reset({ [fresh.steps[0].name]: 'full' })
  }

  /** Hand the draft to another workspace's editor - nothing is saved yet,
   * so where it saves is only which workspace the route names. */
  function moveToWorkspace(target: string) {
    sessionStorage.setItem(
      'dw-editor-import',
      JSON.stringify($state.snapshot(ed.doc)),
    )
    sessionStorage.setItem(NAME_KEY, ed.saveName)
    sessionStorage.setItem(FROM_KEY, startFrom)
    go('ws', target, 'edit')
  }

  /** A name the user typed stays; one the last example supplied follows
   * the new choice. */
  function followName(example: string) {
    const supplied = listed.examples.includes(startFrom)
      ? leafOf(startFrom)
      : ''
    if (ed.saveName && ed.saveName !== supplied) return
    ed.saveName = example ? leafOf(example) : ''
  }

  async function chooseStart(example: string) {
    const message = `Start over from ${example || 'a blank workflow'}? Your edits are lost.`
    const ask = { confirmLabel: 'Start over', danger: true }
    if (ed.unsaved() && !(await confirmDialog(message, ask))) return
    try {
      const fresh = await startingWorkflow(example, emptyWorkflow)
      ed.startNew(fresh)
      modes.reset(example ? {} : { [fresh.steps[0].name]: 'full' })
      followName(example)
      startFrom = example
    } catch (e) {
      notify.failure('start a new workflow', e)
    }
  }

  async function validate(): Promise<boolean> {
    // Validate, save and run all see what is on screen, typed or not
    ed.flush()
    if (!ed.canWrite) return false
    if (ed.jsonParseFailed) {
      notify.error('Fix the JSON first', 'json-parse')
      return false
    }
    ed.busy = true
    try {
      check.validation = await api.validate(
        $state.snapshot(ed.doc) as WorkflowDefinition,
      )
      return check.validation.valid
    } catch (e) {
      notify.failure('validate the workflow', e)
      return false
    } finally {
      ed.busy = false
    }
  }

  async function save() {
    ed.flush()
    const path = ed.savePath()
    if (!path) {
      if (!ed.saveName) notify.error('Give the workflow a file name first')
      else if (!ed.directory()) notify.error('Name the new folder first')
      else notify.error('Folder names: letters, numbers, dot, dash, underscore')
      // The message names a field the user cannot see while collapsed
      fileOpen = true
      return
    }
    if (!(await saving.run(path))) return
    // The file is written: what follows only brings the page up to it
    try {
      // The URL names the file now on screen: a first save, or a save as.
      // It goes first so a failure after it leaves page and URL agreeing
      opened = `${workspace.current}:${path}`
      if (path !== name)
        renameRoute(wsHref(workspace.current, 'edit', ...path.split('/')))
      // A folder this save created must be in the picker at once, or the
      // select falls back to "(root)" while the state still names it
      if (catalog.data) catalog.adopt(withSaved(catalog.data, path))
      // and the time it was saved at is the server's, read back quietly: the
      // save names no mtime, and the listing has no rows, so nothing is marked
      void catalog.refresh()
    } catch (e) {
      notify.failure('update the editor after saving', e)
    }
  }
  // Save and Run are actions: pressed twice, each does one thing
  const saving = action((path: string) => validatedSave(ed, path, validate), {
    doing: (path) => `save ${path}`,
    success: (path) => `Saved to ${path}`,
  })
  const running = action(
    async () => {
      const job = await validatedRun(ed, listed.dir, validate)
      if (job) goWs('jobs', job.id)
      return job
    },
    { doing: 'run the workflow' },
  )
  const run = () => void running.run()
  const onKeydown = editorKeys(ed, { save: () => void save(), run })
</script>

<svelte:window onkeydown={onKeydown} />

<!-- The editing surface: a flush commits the field focused in it -->
<div class="editing" bind:this={ed.root}>
  <div class="head">
    <PageHeader title={current || 'New workflow'} {back}>
      <label class="wfidwrap">
        <span class="wfidlabel">id</span>
        <input
          class="wfid"
          bind:value={ed.doc.id}
          placeholder="workflow id"
          aria-label="workflow id"
          title="the workflow's id - how it names itself, independent of the file name"
        />
      </label>
      {#if ed.dirty}
        <span
          class="chip unsaved"
          title="this definition differs from the last save"
        >
          unsaved
        </span>
      {/if}
      {#snippet actions()}
        <Segmented
          label="editor view"
          value={ed.view}
          options={EDITOR_VIEWS}
          onselect={(view) => ed.setView(view as EditorView)}
        />
        <button
          class="quiet withicon"
          onclick={validate}
          disabled={ed.busy || !ed.canWrite}
          title="check against the schema and real pipeline signatures, without running"
        >
          <CircleCheck size={14} />Validate
        </button>
        <button
          class="quiet withicon"
          onclick={save}
          disabled={ed.busy || !ed.canWrite}
          title="validate, then write to the workflow directory under the name below (Ctrl+S)"
        >
          <Save size={14} />Save{#if ed.dirty}<span class="dirtydot"
            ></span>{/if}
        </button>
        <button
          class="withicon"
          onclick={run}
          disabled={ed.busy || !ed.canWrite}
          title="validate, then queue this definition as a job - no save needed (Ctrl+Enter)"
        >
          <Play size={14} />Run
        </button>
      {/snippet}
    </PageHeader>
  </div>

  {#if ed.loadState === 'failed'}
    <LoadFailed
      label={name}
      error={ed.loadError}
      onretry={() => open(name)}
      oncreate={ed.loadError instanceof ApiError && ed.loadError.status === 404
        ? startMissing
        : undefined}
    />
  {:else}
    <DocumentFileBar
      {ed}
      dir={listed.dir}
      {folders}
      changed={catalog.data?.details?.[current]?.mtime}
      bind:fileOpen
    >
      {#snippet extra()}
        {#if !current}
          <WorkflowOrigin
            workspace={workspace.current}
            workspaces={workspace.names ?? [workspace.current]}
            {startFrom}
            examples={listed.examples}
            onworkspace={moveToWorkspace}
            onstartfrom={chooseStart}
          />
        {/if}
      {/snippet}
    </DocumentFileBar>
    {#if catalog.error}
      <ErrorLine error={catalog.error} doing="list the workflows" />
    {/if}

    {#if check.validation || check.error}
      <ValidationPanel
        validation={check.validation}
        error={check.error}
        doc={ed.doc}
        onjump={jumps.goTo}
        ondismiss={() => {
          check.validation = null
          check.error = null
        }}
      />
    {/if}

    {#if ed.view === 'flow'}
      <FlowView workflow={$state.snapshot(ed.doc)} onselect={jumps.showStep} />
    {:else}
      <EditorBody
        view={ed.view}
        jsonDraft={ed.jsonDraft}
        onjson={(raw, o) => ed.applyJson(raw, o)}
        register={(f) => ed.registerFlush(f)}
        hint="Schema-aware: completion, hover docs and validation come from the workflow schema. Changes apply as you type."
        stickyTop="136px"
      >
        {#snippet form()}
          <div class="panel">
            <h2>Variables</h2>
            <VariablesForm
              mode="define"
              bind:variables={ed.doc.variables}
              idPrefix="wfvar-"
              backendVariables={attentionBackendVariables(ed.doc)}
              constraints={constraintHints(ed.doc)}
              presets={ed.doc.presets}
            />
          </div>

          <StepList
            bind:steps={ed.doc.steps}
            bind:workflow={ed.doc}
            {modes}
            folder={ed.folder === NEW_FOLDER ? '' : ed.folder}
            {flow}
            {problemsByStep}
            {stepReferences}
            bind:hovered={jumps.hovered}
          />
        {/snippet}
      </EditorBody>
    {/if}
  {/if}
</div>

<style>
  /* No box of its own: the sticky head keeps its containing block */
  .editing {
    display: contents;
  }
  /* Save, Run and the way out stay reachable while a long workflow
     scrolls. Below 900px the app header wraps to a second row and its
     height stops being predictable, so the pinning is dropped there. */
  .head {
    position: sticky;
    top: 76px;
    z-index: 5;
    background: var(--bg);
    padding: 0.5rem 0;
    margin-bottom: 0.2rem;
    border-bottom: 1px solid var(--line);
  }
  @media (max-width: 900px) {
    .head {
      position: static;
    }
  }
  .head :global(.pagehead) {
    margin-bottom: 0;
  }
  .wfidwrap {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    max-width: 280px;
  }
  .wfidlabel {
    color: var(--muted);
    font-size: 0.75rem;
    text-transform: none;
  }
  /* Reads as a title until you reach for it, rather than as an unexplained
     text box sitting next to the title */
  .wfid {
    font-weight: 700;
    background: transparent;
    border-color: transparent;
  }
  .wfid:hover {
    border-color: var(--line);
  }
  .wfid:focus {
    background: var(--panel-2);
  }
  .panel {
    margin-bottom: 1rem;
  }
</style>
