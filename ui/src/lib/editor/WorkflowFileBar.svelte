<script module lang="ts">
  /** The start-from value for a definition handed over from elsewhere (an
   * image's embedded metadata) rather than chosen here. */
  export const IMPORTED = '(imported)'
</script>

<script lang="ts">
  import { ChevronUp, FileCog } from '@lucide/svelte'
  import FolderPicker from './FolderPicker.svelte'
  import type { DocumentEditor } from '../editorShell.svelte'
  import { groupOf } from '../grouping'

  // Where the workflow saves, and what it says about itself: the full
  // fields while open, a path chip once the workflow has a name
  let {
    ed,
    workflowDir,
    folders,
    fileOpen = $bindable(),
    origin = undefined,
  }: {
    ed: DocumentEditor
    workflowDir: string
    folders: string[]
    fileOpen: boolean
    /** A workflow not yet saved: which workspace it saves into and what it
     * starts from - blank, or one of the read-only examples */
    origin?: {
      workspace: string
      workspaces: string[]
      startFrom: string
      examples: string[]
      onworkspace: (workspace: string) => void
      onstartfrom: (example: string) => void
    }
  } = $props()

  // The examples by folder, for the start-from picker's option groups
  const exampleGroups = $derived.by(() => {
    const groups: Record<string, string[]> = {}
    for (const example of origin?.examples ?? [])
      (groups[groupOf(example)] ??= []).push(example)
    return Object.entries(groups)
  })

  const savePreview = $derived.by(() => {
    const directory = ed.directory()
    const file = ed.saveName || 'unnamed'
    return `${workflowDir}/${directory ? directory + '/' : ''}${file}.json`
  })
</script>

{#if fileOpen}
  <div class="filebar panel">
    <div class="filegrid">
      {#if origin}
        <label for="wf-workspace">workspace</label>
        <div>
          <select
            id="wf-workspace"
            class="pick"
            value={origin.workspace}
            onchange={(e) => {
              const picked = e.currentTarget.value
              // the route change re-renders it; until then show the
              // workspace the draft is still in
              e.currentTarget.value = origin.workspace
              origin.onworkspace(picked)
            }}
            title="the workspace this workflow saves into"
          >
            {#each origin.workspaces as name (name)}<option value={name}
                >{name}</option
              >{/each}
          </select>
        </div>

        <label for="wf-startfrom">start from</label>
        <div>
          <select
            id="wf-startfrom"
            class="pick"
            value={origin.startFrom}
            onchange={(e) => {
              const picked = e.currentTarget.value
              // held at the current start until the swap lands - a
              // cancelled confirm leaves it where it was
              e.currentTarget.value = origin.startFrom
              origin.onstartfrom(picked)
            }}
            title="a blank workflow, or a copy of one of the examples"
          >
            <option value="">blank</option>
            {#if origin.startFrom && !origin.examples.includes(origin.startFrom)}
              <!-- handed over from elsewhere: an image's metadata, or a copy
                   of one of this workspace's own workflows -->
              <option value={origin.startFrom} disabled
                >{origin.startFrom === IMPORTED
                  ? 'image metadata'
                  : `copy of ${origin.startFrom}`}</option
              >
            {/if}
            {#each exampleGroups as [group, names] (group)}
              <optgroup label={group ? `${group}/` : 'examples'}>
                {#each names as example (example)}<option value={example}
                    >{example}</option
                  >{/each}
              </optgroup>
            {/each}
          </select>
        </div>
      {/if}

      <label for="wf-folder">folder</label>
      <div class="folderrow">
        <FolderPicker
          id="wf-folder"
          bind:folder={ed.folder}
          bind:newFolder={ed.newFolder}
          {folders}
          newFolderTitle="name for the new folder at the root of the workflow directory"
        />
      </div>

      <label for="wf-savename">file name</label>
      <div class="namerow">
        <input
          id="wf-savename"
          class="savename"
          bind:value={ed.saveName}
          placeholder="MyWorkflow"
        />
        <span class="muted">.json</span>
      </div>

      <label for="wf-description">description</label>
      <input
        id="wf-description"
        spellcheck="true"
        value={ed.doc.description ?? ''}
        placeholder="shown on the workflow card"
        title="a short description of what this workflow does"
        onchange={(e) => {
          const v = e.currentTarget.value
          if (v) ed.doc.description = v
          else delete ed.doc.description
        }}
      />
    </div>
    <div class="filefoot">
      <span class="muted path">{savePreview}</span>
      <button
        class="quiet withicon"
        onclick={() => (fileOpen = false)}
        disabled={!ed.saveName}
        title="collapse the file settings"
      >
        <ChevronUp size={14} />done
      </button>
    </div>
  </div>
{:else}
  <div class="savebar">
    <button
      class="quiet withicon pathchip"
      onclick={() => (fileOpen = true)}
      title="change the folder, file name or description"
    >
      <FileCog size={14} /><span class="path">{savePreview}</span>
    </button>
    {#if ed.doc.description}
      <span class="muted desc">{ed.doc.description}</span>
    {/if}
  </div>
{/if}

<style>
  .savebar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.6rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
    min-width: 0;
  }
  .pathchip {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    padding: 0.25rem 0.6rem;
    max-width: 100%;
  }
  .pathchip:hover {
    color: var(--ink);
    border-color: var(--accent);
  }
  .path {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .savebar .desc {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .filebar {
    margin-bottom: 1rem;
  }
  .filegrid {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    gap: 0.5rem 0.8rem;
    align-items: center;
  }
  .filegrid label {
    font-weight: 600;
    color: var(--muted);
    font-size: 0.85rem;
  }
  @container (max-width: 400px) {
    .filegrid {
      grid-template-columns: minmax(0, 1fr);
      gap: 0.2rem 0.5rem;
    }
  }
  .folderrow,
  .namerow {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
  }
  .filefoot {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.4rem 0.8rem;
    margin-top: 0.8rem;
  }
  .filefoot .path {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    min-width: 0;
  }
  .savename {
    max-width: 200px;
  }
  .pick {
    max-width: 320px;
  }
</style>
