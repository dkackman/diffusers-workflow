<script lang="ts">
  import { api } from '../api'
  import { go } from '../router.svelte'
  import type { WorkflowDefinition } from '../types'

  let { name }: { name: string } = $props()

  let workflow = $state<WorkflowDefinition | null>(null)
  let workflowDir = $state('')
  let overrides = $state<Record<string, string>>({})
  let error = $state('')
  let submitting = $state(false)
  let showJson = $state(false)

  $effect(() => {
    overrides = {}
    workflow = null
    api.listWorkflows().then((r) => (workflowDir = r.workflow_dir))
    api
      .getWorkflow(name)
      .then((definition) => (workflow = definition))
      .catch((e) => (error = e.message))
  })

  const variables = $derived(Object.entries(workflow?.variables ?? {}))

  function display(value: unknown): string {
    return typeof value === 'string' ? value : JSON.stringify(value)
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
  <button class="quiet" onclick={() => (showJson = !showJson)}>
    {showJson ? 'hide' : 'show'} JSON
  </button>
  <a class="editlink" href={'#/edit/' + name.split('/').map(encodeURIComponent).join('/')}>Edit</a>
  <button onclick={run} disabled={submitting || !workflow}>
    {submitting ? 'Submitting…' : 'Run'}
  </button>
</div>

{#if error}<p class="error">{error}</p>{/if}

{#if workflow}
  {#if variables.length}
    <div class="panel">
      <h2>Arguments <span class="muted">(blank = workflow default)</span></h2>
      <div class="vars">
        {#each variables as [key, defaultValue]}
          <label for={'var-' + key}>{key}</label>
          {#if display(defaultValue).length > 60}
            <textarea
              id={'var-' + key}
              rows="3"
              placeholder={display(defaultValue)}
              bind:value={overrides[key]}
            ></textarea>
          {:else}
            <input
              id={'var-' + key}
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
    <pre class="json">{JSON.stringify(workflow, null, 2)}</pre>
  {/if}
{/if}

<style>
  .head { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }
  .head h1 { flex: 1; }
  .editlink { font-weight: 600; }
  .vars {
    display: grid; grid-template-columns: minmax(140px, auto) 1fr;
    gap: 0.5rem 1rem; align-items: start;
  }
  .vars label { padding-top: 0.45rem; font-weight: 600; color: var(--muted); }
  .json { margin-top: 1rem; max-height: 480px; overflow: auto; }
  .error { color: var(--bad); }
</style>
