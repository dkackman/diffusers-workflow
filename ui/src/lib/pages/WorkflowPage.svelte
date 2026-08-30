<script lang="ts">
  import { Copy, Play, SquarePen, Trash2 } from '@lucide/svelte'
  import JsonEditor from '../editor/JsonEditor.svelte'
  import { api } from '../api'
  import { go } from '../router.svelte'
  import { displayValue as display, isReference } from '../editor'
  import { loadPromptLibrary, promptLibrary } from '../promptlib.svelte'
  import { PROMPT_LIST_ID, promptListId, promptTooltip } from '../prompts'
  import type { WorkflowDefinition } from '../types'

  let { name }: { name: string } = $props()

  let workflow = $state<WorkflowDefinition | null>(null)
  let workflowDir = $state('')
  let overrides = $state<Record<string, string>>({})
  let error = $state('')
  let submitting = $state(false)
  let showJson = $state(true)

  $effect(() => {
    overrides = {}
    workflow = null
    api.listWorkflows().then((r) => (workflowDir = r.workflow_dir))
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
      error = e instanceof Error ? e.message : String(e)
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
        workflow_path: `${workflowDir}/${name}.json`,
        arguments: args,
      })
      go('jobs', job.id)
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
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
  <button
    class="quiet icon danger"
    onclick={remove}
    title="delete this workflow file from disk"
    aria-label="delete this workflow file from disk"
  >
    <Trash2 size={14} />
  </button>
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
      <div class="vars">
        {#each variables as [key, defaultValue] (key)}
          <label for={'var-' + key}>{key}</label>
          {#if display(defaultValue).length > 60}
            <div class="fieldcol">
              <textarea
                id={'var-' + key}
                class:ref={isReference(overrides[key] || defaultValue)}
                rows="3"
                spellcheck="true"
                title={promptTooltip(
                  overrides[key] || defaultValue,
                  promptLibrary.texts,
                )}
                placeholder={display(defaultValue)}
                bind:value={overrides[key]}></textarea>
              {#if promptLibrary.names?.length}
                <select
                  class="promptpick"
                  title="override this variable with a stored prompt from the library"
                  onchange={(e) => {
                    if (!e.currentTarget.value) return
                    overrides[key] = 'prompt:' + e.currentTarget.value
                    e.currentTarget.value = ''
                  }}
                >
                  <option value="">use a stored prompt…</option>
                  {#each promptLibrary.names ?? [] as promptName (promptName)}
                    <option value={promptName}>{promptName}</option>
                  {/each}
                </select>
              {/if}
            </div>
          {:else}
            <input
              id={'var-' + key}
              class:ref={isReference(overrides[key] || defaultValue)}
              list={promptListId(overrides[key] || defaultValue)}
              autocomplete="off"
              title={promptTooltip(
                overrides[key] || defaultValue,
                promptLibrary.texts,
              )}
              placeholder={display(defaultValue)}
              bind:value={overrides[key]}
            />
          {/if}
        {/each}
      </div>
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
  .head {
    display: flex;
    align-items: center;
    gap: 1rem;
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
  .vars {
    display: grid;
    grid-template-columns: minmax(140px, auto) 1fr;
    gap: 0.5rem 1rem;
    align-items: start;
  }
  .vars label {
    padding-top: 0.45rem;
    font-weight: 600;
    color: var(--muted);
  }
  .fieldcol {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    align-items: flex-start;
  }
  .fieldcol textarea {
    width: 100%;
  }
  .promptpick {
    max-width: 260px;
    font-size: 0.8rem;
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
