<script lang="ts">
  import { ChevronUp, FileCog } from '@lucide/svelte'
  import FolderPicker from './FolderPicker.svelte'
  import type { DocumentEditor } from '../editorShell.svelte'

  // Where the workflow saves, and what it says about itself: the full
  // fields while open, a path chip once the workflow has a name
  let {
    ed,
    workflowDir,
    folders,
    fileOpen = $bindable(),
  }: {
    ed: DocumentEditor
    workflowDir: string
    folders: string[]
    fileOpen: boolean
  } = $props()

  const savePreview = $derived.by(() => {
    const directory = ed.directory()
    const file = ed.saveName || 'unnamed'
    return `${workflowDir}/${directory ? directory + '/' : ''}${file}.json`
  })
</script>

{#if fileOpen}
  <div class="filebar panel">
    <div class="filegrid">
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
</style>
