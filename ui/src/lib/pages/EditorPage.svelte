<script lang="ts">
  import { editorLists, loadEditorLists } from '../editorLists.svelte'
  import { DocumentEditor, NEW_FOLDER } from '../editorShell.svelte'
  import ViewSwitch from '../editor/ViewSwitch.svelte'
  import EditorBody from '../editor/EditorBody.svelte'
  import StepList from '../editor/StepList.svelte'
  import { StepModes } from '../stepModes.svelte'
  import ValidationPanel from '../editor/ValidationPanel.svelte'
  import WorkflowFileBar from '../editor/WorkflowFileBar.svelte'
  import {
    CircleCheck,
    Columns2,
    Braces,
    LayoutList,
    Play,
    Save,
    Workflow,
  } from '@lucide/svelte'
  import { api } from '../api'
  import { writableRoot } from '../libraries'
  import { notify } from '../toast'
  import { confirmDialog } from '../confirm.svelte'
  import { goWs } from '../router.svelte'
  import { wsHref } from '../routes'
  import { workspace } from '../workspace.svelte'
  import { emptyWorkflow, referenceSuggestions } from '../editor'
  import { danglingReferenceDetails, flowGraph } from '../flow'
  import { groupOf, leafOf } from '../grouping'
  import { loadPromptLibrary, promptLibrary } from '../promptlib.svelte'
  import VariablesForm from '../editor/VariablesForm.svelte'
  import FlowView from '../editor/FlowView.svelte'
  import type { ValidationResult, WorkflowDefinition } from '../types'

  let { name = '' }: { name?: string } = $props()

  const ed = new DocumentEditor({
    viewKey: 'dw-editor-view',
    views: ['form', 'split', 'json', 'flow'],
    // migrate the old boolean split preference
    legacyView: () =>
      localStorage.getItem('dw-editor-split') === '1' ? 'split' : null,
  })
  ed.doc = emptyWorkflow()
  let workflowDir = $state('')
  // The file fields collapse behind the path chip once the workflow has a
  // name - they are settings, not something you retouch on every edit
  let fileOpen = $state(false)

  const modesKey = $derived(`step-modes:${name || '(unsaved)'}`)
  // Keyed by step name; existing steps open compact, new ones full
  const modes = new StepModes(() => modesKey)

  // Existing folders, from the listing - one level is the designed depth,
  // but any deeper directories that exist still appear and keep working
  const folders = $derived(
    [...new Set(editorLists.workflowFiles.map(groupOf).filter(Boolean))].sort(),
  )
  let validation = $state<ValidationResult | null>(null)
  const VIEWS = [
    {
      view: 'form',
      label: 'form',
      title: 'edit with introspection-driven forms',
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
    {
      view: 'flow',
      label: 'flow',
      title:
        'read-only data-flow diagram: steps as boxes, previous_result as edges',
      icon: Workflow,
    },
  ] as const

  // Crumbs back out of the editor. The read-only page is where an edit
  // usually starts, and until now the only exit landed on the list - so
  // the last crumb links back to the workflow itself, when one is saved.
  const crumbFolders = $derived(
    name ? groupOf(name).split('/').filter(Boolean) : [],
  )
  const crumbName = $derived(name ? leafOf(name) : '')
  const workflowHref = $derived(
    wsHref(workspace.current, 'workflows', ...name.split('/')),
  )

  // Leaving with unsaved edits used to drop them without a word. The
  // confirm is async, so the default navigation is always prevented first
  // and replayed by hand once the answer comes back.
  async function confirmLeave(event: MouseEvent) {
    if (!ed.dirty) return
    event.preventDefault()
    const target = (event.currentTarget as HTMLAnchorElement).href
    if (await confirmDialog('Discard unsaved changes?')) {
      window.location.href = target
    }
  }

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
  let hovered = $state<string | null>(null)
  // Memoized so datalist options keep stable DOM identity - churn on every
  // render made the browser's suggestion dropdown flaky on first focus
  const stepReferences = $derived.by(() => {
    const snapshot = $state.snapshot(ed.doc)
    return ((snapshot.steps as unknown[]) ?? []).map((_, index) =>
      referenceSuggestions(snapshot, index, promptLibrary.names ?? []),
    )
  })

  $effect(() => {
    loadEditorLists()
    api.listWorkflows().then((r) => {
      editorLists.workflowFiles = r.workflows.map((file) => `${file}.json`)
      workflowDir = writableRoot(r.libraries)
    })
    loadPromptLibrary()
    validation = null
    if (name) {
      ed.saveName = leafOf(name)
      ed.folder = groupOf(name)
      fileOpen = false
      api
        .getWorkflow(name)
        .then((fetched) => {
          ed.load(fetched.definition as WorkflowDefinition)
          modes.restore()
        })
        .catch((e) => notify.error(e.message))
    } else {
      // A gallery "open as workflow" hands the definition over in
      // sessionStorage - one-shot, so a plain "New" stays a blank slate
      const imported = ed.takeImport('dw-editor-import', 'dw-editor-folder')
      // Built as a plain local object and assigned once: reading the
      // workflow proxy here would subscribe this effect to every edit and
      // re-fire all the listing calls on each keystroke
      const fresh = imported ?? (emptyWorkflow() as Record<string, any>)
      if (imported) notify.success('Imported from image metadata')
      ed.load(fresh)
      ed.saveName = ''
      // A new workflow has nowhere to save to yet - show the fields
      fileOpen = true
      if (imported) modes.restore()
      else modes.reset({ [fresh.steps[0].name]: 'full' })
    }
  })

  function onKeydown(event: KeyboardEvent) {
    if (!(event.ctrlKey || event.metaKey)) return
    if (event.key === 's') {
      event.preventDefault()
      save()
    } else if (event.key === 'Enter') {
      event.preventDefault()
      run()
    }
  }

  async function validate(): Promise<boolean> {
    ed.busy = true
    try {
      validation = await api.validate(
        $state.snapshot(ed.doc) as WorkflowDefinition,
      )
      return validation.valid
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
      return false
    } finally {
      ed.busy = false
    }
  }

  async function save() {
    const path = ed.savePath()
    if (!path) {
      if (!ed.saveName) notify.error('Give the workflow a file name first')
      else if (!ed.directory()) notify.error('Name the new folder first')
      else notify.error('Folder names: letters, numbers, dot, dash, underscore')
      // The message names a field the user cannot see while collapsed
      fileOpen = true
      return
    }
    if (!(await validate())) return
    ed.busy = true
    try {
      await api.saveWorkflow(
        path,
        $state.snapshot(ed.doc) as WorkflowDefinition,
      )
      ed.commitNewFolder()
      // The folder picker's options come from the listing - a folder this
      // save just created must appear there, or the select falls back to
      // "(root)" while the state still names the folder
      if (!editorLists.workflowFiles.includes(`${path}.json`)) {
        editorLists.workflowFiles = [
          ...editorLists.workflowFiles,
          `${path}.json`,
        ]
      }
      ed.markSaved()
      notify.success(`Saved to ${path}`)
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    } finally {
      ed.busy = false
    }
  }

  async function run() {
    if (!(await validate())) return
    ed.busy = true
    try {
      // base_dir anchors relative paths (images, sub-workflow files) the
      // way running the saved file would - at the workflow's own folder
      const directory =
        ed.folder && ed.folder !== NEW_FOLDER ? `/${ed.folder}` : ''
      const job = await api.submitJob({
        workflow: $state.snapshot(ed.doc) as WorkflowDefinition,
        base_dir: `${workflowDir}${directory}`,
      })
      goWs('jobs', job.id)
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    } finally {
      ed.busy = false
    }
  }

  // The flow view's click-to-navigate: land back on the form view with the
  // step's row scrolled into place and briefly lit, the same highlight the
  // rail chips already use on hover - a reasonable cross-reference without
  // teaching the flow view anything about editing.
  function focusStep(stepName: string) {
    ed.setView('form')
    hovered = stepName
    requestAnimationFrame(() => {
      document
        .getElementById('step-' + stepName)
        ?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    })
    setTimeout(() => {
      if (hovered === stepName) hovered = null
    }, 2000)
  }
</script>

<svelte:window onkeydown={onKeydown} />

<div class="head">
  <nav class="crumbs muted" aria-label="breadcrumb">
    <a href={wsHref(workspace.current, 'workflows')} onclick={confirmLeave}
      >← workflows</a
    >
    {#each crumbFolders as part (part)}
      <span class="sep">/</span><span>{part}</span>
    {/each}
    {#if crumbName}
      <span class="sep">/</span>
      <a
        href={workflowHref}
        onclick={confirmLeave}
        title="back to the read-only view of this workflow"
      >
        {crumbName}
      </a>
    {/if}
  </nav>
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
      title="this definition differs from the last save">unsaved</span
    >
  {/if}
  <span class="flex"></span>
  <ViewSwitch
    view={ed.view}
    options={[...VIEWS]}
    onselect={(view) => ed.setView(view)}
  />
  <button
    class="quiet withicon"
    onclick={validate}
    disabled={ed.busy}
    title="check against the schema and real pipeline signatures, without running"
  >
    <CircleCheck size={14} />Validate
  </button>
  <button
    class="quiet withicon"
    class:dirtybtn={ed.dirty}
    onclick={save}
    disabled={ed.busy}
    title="validate, then write to the workflow directory under the name below (Ctrl+S)"
  >
    <Save size={14} />Save
  </button>
  <button
    class="withicon"
    onclick={run}
    disabled={ed.busy}
    title="validate, then queue this definition as a job - no save needed (Ctrl+Enter)"
  >
    <Play size={14} />Run
  </button>
</div>

<WorkflowFileBar {ed} {workflowDir} {folders} bind:fileOpen />

{#if validation}
  <ValidationPanel {validation} ondismiss={() => (validation = null)} />
{/if}

{#if ed.view === 'flow'}
  <FlowView workflow={$state.snapshot(ed.doc)} onselect={focusStep} />
{:else}
  <EditorBody
    view={ed.view}
    jsonDraft={ed.jsonDraft}
    onjson={(raw) => ed.applyJson(raw)}
    hint="Schema-aware: completion, hover docs and validation come from the workflow schema. Changes apply when the editor loses focus."
    stickyTop="136px"
  >
    {#snippet form()}
      <div class="panel">
        <h2>Variables</h2>
        <VariablesForm
          mode="define"
          bind:variables={ed.doc.variables}
          idPrefix="wfvar-"
        />
      </div>

      <StepList
        bind:steps={ed.doc.steps}
        {modes}
        folder={ed.folder === NEW_FOLDER ? '' : ed.folder}
        {flow}
        {problemsByStep}
        {stepReferences}
        bind:hovered
      />
    {/snippet}
  </EditorBody>
{/if}

<style>
  /* Save, Run and the way out stay reachable while a long workflow
     scrolls. Below 900px the app header wraps to a second row and its
     height stops being predictable, so the pinning is dropped there. */
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.6rem;
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
  .crumbs {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.85rem;
    min-width: 0;
  }
  .crumbs .sep {
    opacity: 0.5;
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
     text box sitting next to a breadcrumb */
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
  .chip.unsaved {
    background: color-mix(in srgb, var(--warn) 22%, transparent);
    color: var(--warn);
  }
  .panel {
    margin-bottom: 1rem;
  }
  /* Unsaved work makes Save the thing to do next, so it stops looking
     like the two quiet buttons beside it */
  .dirtybtn {
    border-color: var(--warn);
    color: var(--warn);
  }
</style>
