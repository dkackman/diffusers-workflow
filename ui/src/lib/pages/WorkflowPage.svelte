<script lang="ts">
  import { Copy, Play, SquarePen, Trash2 } from '@lucide/svelte'
  import DownloadLink from '../DownloadLink.svelte'
  import JsonEditor from '../editor/JsonEditor.svelte'
  import VariablesForm from '../editor/VariablesForm.svelte'
  import { api } from '../api'
  import { go } from '../router.svelte'
  import { loadPromptLibrary, promptLibrary } from '../promptlib.svelte'
  import { PROMPT_LIST_ID } from '../prompts'
  import { notify } from '../toast'
  import type { WorkflowDefinition } from '../types'

  let { name }: { name: string } = $props()

  let workflow = $state<WorkflowDefinition | null>(null)
  let workflowDir = $state('')
  /** Where this workflow was read from, and whether it is the user's to
   * change - an examples or builtin source is read-only. */
  let origin = $state('')
  let writable = $state(true)
  let overrides = $state<Record<string, string>>({})
  let error = $state('')
  let submitting = $state(false)
  let showJson = $state(true)

  $effect(() => {
    overrides = {}
    workflow = null
    api.listWorkflows().then((r) => {
      workflowDir = r.workflow_dir
      origin = r.details[name]?.origin ?? ''
      writable = r.details[name]?.writable ?? true
    })
    loadPromptLibrary()
    api
      .getWorkflow(name)
      .then((definition) => (workflow = definition))
      .catch((e) => (error = e.message))
  })

  const variables = $derived(Object.entries(workflow?.variables ?? {}))

  function newFrom() {
    if (!workflow) return
    sessionStorage.setItem('dw-editor-import', JSON.stringify(workflow))
    go('edit')
  }

  async function remove() {
    if (!window.confirm(`Delete ${name}.json? This removes the file on disk.`))
      return
    try {
      await api.deleteWorkflow(name)
      go('workflows')
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      notify.error(msg)
    }
  }

  async function run() {
    submitting = true
    error = ''
    try {
      const args: Record<string, unknown> = {}
      for (const [key, value] of Object.entries(overrides)) {
        if (value !== '') args[key] = value
      }
      const job = await api.submitJob({
        // By name, not by composed path: the server resolves a name across
        // every source it can read, so an example runs where it lives
        workflow_path: name,
        arguments: args,
      })
      go('jobs', job.id)
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      notify.error(msg)
    } finally {
      submitting = false
    }
  }
</script>

<div class="head">
  <a href="#/workflows" class="muted">← workflows</a>
  <h1>{name}</h1>
  <button
    class="quiet"
    onclick={() => (showJson = !showJson)}
    title={showJson
      ? 'hide the workflow definition'
      : 'show the workflow definition'}
  >
    {showJson ? 'hide' : 'show'} JSON
  </button>
  <a
    class="editlink withicon"
    href={'#/edit/' + name.split('/').map(encodeURIComponent).join('/')}
    title="open this workflow in the editor"
  >
    <SquarePen size={14} />Edit
  </a>
  <button
    class="quiet withicon"
    onclick={newFrom}
    disabled={!workflow}
    title="open a copy in the editor"
  >
    <Copy size={14} />New from
  </button>
  <span class="spacer"></span>
  {#if !writable}
    <span class="readonly muted" title={`read-only: this workflow comes from the ${origin} directory. Saving an edit writes a copy into ${workflowDir}`}>
      read-only{origin ? ` (${origin})` : ''}
    </span>
  {/if}
  <DownloadLink href={api.workflowDownloadUrl(name)} />
  {#if writable}
    <button
      class="quiet icon danger"
      onclick={remove}
      title="delete this workflow file from disk"
      aria-label="delete this workflow file from disk"
    >
      <Trash2 size={14} />
    </button>
  {/if}
  <button
    class="withicon"
    onclick={run}
    disabled={submitting || !workflow}
    title="queue this workflow with the variables below"
  >
    <Play size={14} />{submitting ? 'Submitting…' : 'Run'}
  </button>
</div>

<datalist id={PROMPT_LIST_ID}>
  {#each promptLibrary.names ?? [] as promptName (promptName)}<option
      value={'prompt:' + promptName}
    ></option>{/each}
</datalist>

{#if error}<p class="error">{error}</p>{/if}

{#if workflow}
  {#if workflow.description}
    <p class="muted desc">{workflow.description}</p>
  {/if}
  {#if variables.length}
    <div class="panel">
      <h2>
        Variables
        <span class="muted"
          >(blank = workflow default · type prompt: to use a stored prompt)</span
        >
      </h2>
      <VariablesForm
        mode="override"
        variables={workflow.variables ?? {}}
        bind:overrides
        idPrefix="var-"
      />
    </div>
  {:else}
    <p class="muted">This workflow defines no variables.</p>
  {/if}

  {#if showJson}
    <div class="json">
      <JsonEditor
        value={JSON.stringify(workflow, null, 2)}
        readonly
        height="520px"
      />
    </div>
  {/if}
{/if}

<style>
  .readonly {
    font-size: 0.85em;
    align-self: center;
  }

  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 1rem;
    margin-bottom: 1rem;
  }
  .head h1 {
    flex: 1;
  }
  .editlink {
    font-weight: 600;
  }
  .withicon {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  a.withicon {
    gap: 0.3rem;
  }
  .spacer {
    width: 0.8rem;
  }
  .icon {
    display: inline-flex;
    padding: 0.4rem 0.5rem;
  }
  .json {
    margin-top: 1rem;
  }
  .error {
    color: var(--bad);
  }
  .desc {
    margin: -0.5rem 0 1rem;
    max-width: 75ch;
  }
</style>
