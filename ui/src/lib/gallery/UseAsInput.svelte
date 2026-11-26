<script lang="ts">
  // The dialog behind Use as input… (GA-10): the workspace's workflows that
  // take the open file's kind, one button each. It sits beside the viewer,
  // not inside its More menu, which closes (and unmounts what it holds)
  // the moment an action is chosen. The workflow list is read only while
  // the dialog is open
  import { api } from '../api'
  import { distinctLeaves, leafName } from '../names'
  import type { MediaFile } from '../media'
  import type { MediaKind } from '../mediaArguments'
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import Modal from '../ui/Modal.svelte'
  import { inputTargets, useAsInput } from './useAsInput'

  // The file it was last chosen on, and whether the dialog is up
  let file = $state.raw<MediaFile | null>(null)
  let open = $state(false)

  /** Use as input… was chosen on `chosen` */
  export function show(chosen: MediaFile): void {
    file = chosen
    open = true
  }

  const kind = $derived<MediaKind | null>(
    file?.kind === 'image' || file?.kind === 'video' ? file.kind : null,
  )
  const article = $derived(kind === 'image' ? 'an' : 'a')

  // Closed holds undefined, which LoadState reads as nothing held: null
  // would count as a list, and say no workflow takes it while loading
  const catalog = resource(
    () => (open ? api.listWorkflows() : Promise.resolve(undefined)),
    { watch: ['workflows'] },
  )
  const targets = $derived(
    kind && catalog.data ? inputTargets(catalog.data.details, kind) : [],
  )
  // Two workflows of one leaf in different folders keep the folders that
  // tell them apart
  const shown = $derived(distinctLeaves(targets.map((t) => t.workflow)))

  function choose(target: { workflow: string; variable: string }) {
    if (!file) return
    open = false
    useAsInput(target, file.name)
  }
</script>

{#if file && kind}
  <Modal bind:open label={`Use ${leafName(file.name)} as input`}>
    <h2>Use {leafName(file.name)} as input</h2>
    <LoadState of={catalog} label="the workflows" quiet={false}>
      {#if targets.length}
        <p class="hint muted">
          Workflows in this workspace that take {article}
          {kind}:
        </p>
        <ul class="targets">
          {#each targets as target (target.workflow)}
            <li>
              <button
                type="button"
                class="quiet"
                onclick={() => choose(target)}
                title={`${target.workflow} fills ${target.variable} with this file`}
              >
                {shown.get(target.workflow)}
              </button>
              <span class="muted"><code>{target.variable}</code></span>
            </li>
          {/each}
        </ul>
      {:else}
        <p class="hint muted">
          No workflow in this workspace takes {article}
          {kind}.
        </p>
      {/if}
    </LoadState>
    <div class="actions">
      <button type="button" class="quiet" onclick={() => (open = false)}>
        Cancel
      </button>
    </div>
  </Modal>
{/if}

<style>
  .targets {
    list-style: none;
    margin: 0 0 var(--space-2);
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  .targets li {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
  }
</style>
