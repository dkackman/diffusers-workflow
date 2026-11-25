<script lang="ts">
  import type { Snippet } from 'svelte'
  import { ChevronUp, FileCog } from '@lucide/svelte'
  import FolderPicker from './FolderPicker.svelte'
  import RemoteChanged from '../ui/RemoteChanged.svelte'
  import CopyButton from '../CopyButton.svelte'
  import Time from '../ui/Time.svelte'
  import type { DocumentEditor } from '../editorShell.svelte'
  import { workspace } from '../workspace.svelte'

  // Where a document saves, and what it says about itself: the full fields
  // while open, a path chip once it has a name - and, above them, that the
  // saved file changed under unsaved edits, or was deleted. A bar given no
  // `fileOpen` stays open; one given it collapses.
  let {
    ed,
    dir,
    folders,
    reference = '',
    extra = undefined,
    fileOpen = $bindable(undefined),
    noun = 'workflow',
    described = true,
    changed = undefined,
  }: {
    ed: DocumentEditor
    /** The directory the document saves under, for the path shown */
    dir: string
    folders: string[]
    /** How another document names this one, shown beside a copy button */
    reference?: string
    /** Fields ahead of the folder: a workflow's workspace and start-from */
    extra?: Snippet
    fileOpen?: boolean
    noun?: 'workflow' | 'prompt'
    /** Whether the bar holds the description field (a prompt's is in its form) */
    described?: boolean
    /** When the saved file last changed (unix seconds), from the listing */
    changed?: number
  } = $props()

  const open = $derived(fileOpen ?? true)

  const savePreview = $derived.by(() => {
    const directory = ed.directory()
    const file = ed.saveName || 'unnamed'
    return `${dir}/${directory ? directory + '/' : ''}${file}.json`
  })
</script>

{#snippet stamp()}
  {#if changed}
    <span class="muted stamp">
      <Time at={changed} mode="exact" label="changed" />
    </span>
  {/if}
{/snippet}

{#snippet refcopy()}
  {#if reference}
    <code class="refhint" title="use the stored {noun} from any workflow">
      {reference}
    </code>
    <CopyButton text={reference} title="copy reference to clipboard" />
  {/if}
{/snippet}

{#if ed.remote !== 'same'}
  <RemoteChanged
    label={ed.loadedPath ?? 'this file'}
    gone={ed.remote === 'gone'}
    ontheirs={() => ed.takeTheirs()}
    onmine={() => ed.keepMine()}
  />
{/if}
{#if ed.loadState === 'loaded' && !ed.writable}
  <p class="hint">
    This is an example, so it can't be changed in place. Save makes your own
    copy in {workspace.current}.
  </p>
{/if}
{#if open}
  <div class="filebar panel">
    <div class="formgrid filegrid">
      {@render extra?.()}

      <label for="doc-folder">folder</label>
      <div class="folderrow">
        <FolderPicker
          id="doc-folder"
          bind:folder={ed.folder}
          bind:newFolder={ed.newFolder}
          {folders}
          newFolderTitle="name for the new folder at the root of the {noun} directory"
        />
      </div>

      <label for="doc-savename">file name</label>
      <div class="namerow">
        <input
          id="doc-savename"
          class="savename"
          bind:value={ed.saveName}
          placeholder={noun === 'prompt' ? 'MyPrompt' : 'MyWorkflow'}
        />
        <span class="muted">.json</span>
      </div>

      {#if described}
        <label for="wf-description">description</label>
        <input
          id="wf-description"
          spellcheck="true"
          value={ed.doc.description ?? ''}
          placeholder="shown on the workflow card"
          title="a short description of what this file does"
          oninput={(e) => {
            const v = e.currentTarget.value
            if (v) ed.doc.description = v
            else delete ed.doc.description
          }}
        />
      {/if}
    </div>
    <div class="filefoot">
      <span class="muted path">{savePreview}</span>
      {@render stamp()}
      {@render refcopy()}
      {#if fileOpen !== undefined}
        <button
          class="quiet withicon"
          onclick={() => (fileOpen = false)}
          disabled={!ed.saveName}
          title="collapse the file settings"
        >
          <ChevronUp size={14} />done
        </button>
      {/if}
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
    {@render stamp()}
    {@render refcopy()}
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
  .stamp {
    font-size: var(--t-xs);
    white-space: nowrap;
  }
  .filegrid {
    --label-w: auto;
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
  .refhint {
    font-size: 0.8rem;
    color: var(--accent);
  }
</style>
