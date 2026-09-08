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
  import type { GalleryFile, WorkflowDefinition } from '../types'

  let { name }: { name: string } = $props()

  let workflow = $state<WorkflowDefinition | null>(null)
  /** Where this workflow was read from, and whether it is the user's to
   * change - an examples or builtin source is read-only. Both come back on
   * `getWorkflow` itself, off the response headers, so there is no separate
   * `listWorkflows` round trip just to learn them. */
  let origin = $state('')
  let writable = $state(true)
  let overrides = $state<Record<string, string>>({})
  let error = $state('')
  let submitting = $state(false)
  /** Collapsed by default: this page is where you set variables and run,
   * and a 520px read-only editor between the form and the rest of the page
   * pushed the actual work off the screen. The editor is one click away,
   * and Edit opens the real one. */
  let showJson = $state(false)
  /** What this workflow has already made, newest first - the same proof the
   * catalog card shows, in the place where you decide whether to run it
   * again. */
  let proofs = $state<GalleryFile[]>([])

  $effect(() => {
    overrides = {}
    workflow = null
    origin = ''
    writable = true
    loadPromptLibrary()
    api
      .getWorkflow(name)
      .then((definition) => {
        workflow = definition
        origin = definition.origin
        writable = definition.writable
      })
      .catch((e) => (error = e.message))
  })

  $effect(() => {
    // A run writes to <outputs>/<identity>/<run id>/, and the gallery
    // reports that identity as `folder`, so this workflow's name matches
    // its outputs directly. Loaded separately from the definition so the
    // page is usable before it answers.
    const wanted = name
    proofs = []
    api
      .gallery()
      .then((result) => {
        if (wanted !== name) return
        proofs = result.files
          .filter((file) => file.folder === wanted)
          .slice(0, 6)
      })
      .catch(() => {
        /* the page reads fine without them */
      })
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

<a href="#/workflows" class="back muted">← workflows</a>

<!-- One filled button: Run, the thing you came here to do. Everything else
     is outlined or bare, and Delete is pushed out of reach of Run rather
     than sitting beside it. -->
<div class="head">
  <h1>{name}</h1>
  {#if !writable}
    <span
      class="readonly muted"
      title={`read-only: this workflow comes from the ${origin} directory. Saving an edit writes a copy instead of overwriting it`}
    >
      {origin || 'read-only'}, read-only
    </span>
  {/if}
  <span class="flex"></span>
  <a
    class="plain editlink"
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
  <button
    class="withicon run"
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
    <p class="muted desc prose">{workflow.description}</p>
  {/if}

  <!-- What it made last time, before you decide to make another -->
  {#if proofs.length}
    <div class="proofs">
      {#each proofs as proof (proof.name)}
        <a
          class="plain frame proof"
          href="#/gallery"
          title="{proof.label} — open the gallery"
        >
          {#if proof.kind === 'image'}
            <img
              src={api.galleryThumbnailUrl(proof.name)}
              alt={proof.label}
              loading="lazy"
            />
          {:else}
            <video src={proof.url} muted playsinline preload="metadata"></video>
          {/if}
        </a>
      {/each}
    </div>
  {/if}

  {#if variables.length}
    <div class="panel">
      <!-- The heading names a thing the engine resolves, so it is mono; the
           hint is written for a person, so it is not -->
      <h2>Variables</h2>
      <p class="hint muted">
        Leave a field blank to use the workflow's own default. Type
        <code>prompt:</code> to pull in a stored prompt.
      </p>
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

  <!-- The definition, the file and the destructive action all live below
       the work, in that order of how often anyone wants them -->
  <div class="foot">
    <button
      class="bare"
      onclick={() => (showJson = !showJson)}
      aria-expanded={showJson}
      title={showJson
        ? 'hide the workflow definition'
        : 'show the workflow definition'}
    >
      {showJson ? 'hide' : 'show'} JSON
    </button>
    <DownloadLink href={api.workflowDownloadUrl(name)} />
    <span class="flex"></span>
    {#if writable}
      <button
        class="bare danger withicon"
        onclick={remove}
        title="delete this workflow file from disk"
        aria-label="delete this workflow file from disk"
      >
        <Trash2 size={14} />Delete
      </button>
    {/if}
  </div>

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
  .back {
    display: inline-block;
    font-size: var(--t-sm);
    margin-bottom: var(--space-3);
  }
  .readonly {
    font-family: var(--font-mono);
    font-size: var(--t-xs);
    align-self: center;
  }
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.6rem;
    margin-bottom: var(--space-4);
  }
  .head h1 {
    word-break: break-word;
  }
  .head .flex {
    flex: 1;
  }
  .editlink {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: var(--t-sm);
    font-weight: 600;
    padding: 0.4rem 0.5rem;
  }
  .withicon {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  .run {
    padding: 0.45rem 1.1rem;
  }

  /* The proofs: contact-sheet frames, sized so a row of them reads at a
     glance without taking the page over */
  .proofs {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    margin-bottom: var(--space-4);
  }
  /* .frame carries the look; this only says how big a proof is here */
  .proof {
    width: 116px;
    aspect-ratio: 1;
  }

  .foot {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
    margin-top: var(--space-5);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line);
    font-size: var(--t-sm);
  }
  .foot .flex {
    flex: 1;
  }
  .json {
    margin-top: var(--space-3);
  }
  .error {
    color: var(--bad);
  }
  .desc {
    margin: 0 0 var(--space-4);
  }
  .hint {
    margin: -0.4rem 0 var(--space-3);
    font-size: var(--t-sm);
  }
</style>
