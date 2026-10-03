<script lang="ts">
  import Suggest from '../ui/Suggest.svelte'
  import { isNameSegment } from '../names'
  import { DocumentEditor, NEW_FOLDER } from '../editorShell.svelte'
  import ViewSwitch from '../editor/ViewSwitch.svelte'
  import FolderPicker from '../editor/FolderPicker.svelte'
  import EditorBody from '../editor/EditorBody.svelte'
  import EnhancePanel from '../editor/EnhancePanel.svelte'
  import { Enhancer } from '../enhancer.svelte'
  import { PROMPT, reference } from '../references'
  import {
    Braces,
    Columns2,
    Copy,
    LayoutList,
    Save,
    Sparkles,
    Trash2,
  } from '@lucide/svelte'
  import { api } from '../api'
  import { writableRoot } from '../libraries'
  import DownloadLink from '../DownloadLink.svelte'
  import { go } from '../router.svelte'
  import { sharedHref } from '../routes'
  import { notify } from '../toast'
  import { confirmDialog } from '../confirm.svelte'
  import { loadPromptLibrary } from '../promptlib.svelte'
  import { groupOf, leafOf } from '../grouping'
  import { emptyPrompt, parseTags, workflowsReferencing } from '../prompts'
  import CopyButton from '../CopyButton.svelte'
  import type { PromptDefinition, PromptDetail } from '../types'

  let { name = '' }: { name?: string } = $props()

  const shell = new DocumentEditor({
    viewKey: 'dw-prompt-editor-view',
    views: ['form', 'split', 'json'],
  })
  shell.doc = emptyPrompt()
  let promptDir = $state('')
  let promptFiles = $state<string[]>([])
  let promptDetails = $state<Record<string, PromptDetail>>({})
  // A prompt from a read-only examples library: it can be edited and saved
  // (the save lands in this workspace's library, shadowing it) but not
  // deleted, the way a read-only workflow behaves
  let readOnly = $state(false)

  // Existing folders, from the listing - one level is the designed depth
  const folders = $derived(
    [...new Set(promptFiles.map(groupOf).filter(Boolean))].sort(),
  )

  const VIEWS = [
    {
      view: 'form',
      label: 'form',
      title: 'edit with a form',
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
  ] as const

  // ------------------------------------------------------------- enhancer

  const enhancer = new Enhancer()
  const intendedModels = $derived(enhancer.intendedModels(promptDetails))

  function useResult() {
    const { text, enhanced } = enhancer.takeResult()
    shell.doc.text = text
    shell.doc.enhanced = enhanced
  }

  // ---------------------------------------------------------------- lifecycle

  $effect(() => {
    api
      .listPrompts()
      .then((r) => {
        promptFiles = r.prompts
        promptDir = writableRoot(r.libraries)
        promptDetails = r.details ?? {}
      })
      .catch((e) => notify.error(e.message))
    enhancer.refreshModels()
    if (name) {
      shell.saveName = leafOf(name)
      shell.folder = groupOf(name)
      api
        .getPrompt(name)
        // the definition comes back beside where it was found: a prompt
        // from a read-only examples library can be edited and saved (the
        // copy lands here) but not deleted
        .then(({ prompt, writable }) => {
          readOnly = !writable
          shell.load(prompt)
          enhancer.idea = prompt.enhanced?.idea ?? ''
          enhancer.preselect(shell.doc.intended_model)
        })
        .catch((e) => notify.error(e.message))
    } else {
      const imported = shell.takeImport(
        'dw-prompt-editor-import',
        'dw-prompt-editor-folder',
      )
      if (imported) notify.success('Duplicated - save under a new name')
      shell.load(imported ?? emptyPrompt())
      shell.saveName = ''
    }
    enhancer.load().then(() => enhancer.preselect(shell.doc.intended_model))
    return () => enhancer.stop()
  })

  function onKeydown(event: KeyboardEvent) {
    if ((event.ctrlKey || event.metaKey) && event.key === 's') {
      event.preventDefault()
      save()
    }
  }

  // ---------------------------------------------------------------- editing

  function setField(key: string, value: string) {
    if (value) shell.doc[key] = value
    else delete shell.doc[key]
  }

  function setTags(raw: string) {
    const tags = parseTags(raw)
    if (tags.length) shell.doc.tags = tags
    else delete shell.doc.tags
  }

  // Validation failures are errors (red), not statuses (green checkmark)
  function saveBlocker(): string | null {
    if (!shell.saveName) return 'Give the prompt a file name first'
    if (!isNameSegment(shell.saveName))
      return 'Prompt names: letters, numbers, dot, dash, underscore'
    if (shell.folder === NEW_FOLDER) {
      if (!shell.directory()) return 'Name the new folder first'
      if (!isNameSegment(shell.directory()))
        return 'Folder names: letters, numbers, dot, dash, underscore'
    }
    if (!String(shell.doc.text ?? '').trim())
      return 'The prompt needs text before it can be saved'
    return null
  }

  async function save() {
    const blocker = saveBlocker()
    if (blocker) {
      notify.error(blocker)
      return
    }
    const path = shell.savePath()!
    shell.busy = true
    try {
      await api.savePrompt(path, $state.snapshot(shell.doc) as PromptDefinition)
      shell.commitNewFolder()
      // Same as the workflow editor: the picker lists folders from the
      // listing, so a newly created one must be added or the select resets
      if (!promptFiles.includes(path)) promptFiles = [...promptFiles, path]
      shell.markSaved()
      notify.success(`Saved to ${path}`)
      loadPromptLibrary()
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    } finally {
      shell.busy = false
    }
  }

  async function remove() {
    if (!name) return
    // Say what the delete is about to break: any workflow holding a
    // prompt:name reference fails at run time once the file is gone
    let warning = ''
    try {
      const { details } = await api.listWorkflows()
      const referencing = workflowsReferencing(name, details ?? {})
      if (referencing.length) {
        const shown = referencing.slice(0, 5).join(', ')
        const more = referencing.length > 5 ? ', …' : ''
        warning =
          `\n\nReferenced by ${referencing.length} workflow` +
          `${referencing.length === 1 ? '' : 's'} (${shown}${more}) - ` +
          'they will fail at run time until updated.'
      }
    } catch {
      /* the confirm still protects the file itself */
    }
    if (
      !(await confirmDialog(
        `Delete ${name}.json? This removes the file on disk.${warning}`,
        { confirmLabel: 'Delete' },
      ))
    )
      return
    try {
      await api.deletePrompt(name)
      loadPromptLibrary()
      go('shared', 'prompts')
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    }
  }

  function duplicate() {
    sessionStorage.setItem(
      'dw-prompt-editor-import',
      JSON.stringify($state.snapshot(shell.doc)),
    )
    if (shell.folder && shell.folder !== NEW_FOLDER)
      sessionStorage.setItem('dw-prompt-editor-folder', shell.folder)
    go('shared', 'prompt-edit')
  }
</script>

<svelte:window onkeydown={onKeydown} />

<div class="head">
  <a href={sharedHref('prompts')} class="muted">← prompts</a>
  <h1>{name || 'New prompt'}</h1>
  <span class="flex"></span>
  <ViewSwitch
    view={shell.view}
    options={[...VIEWS]}
    onselect={(view) => shell.setView(view)}
  />
  {#if name}
    <button
      class="quiet withicon"
      onclick={duplicate}
      title="open a copy in the editor to save under a new name"
    >
      <Copy size={14} />Duplicate
    </button>
    <DownloadLink href={api.promptDownloadUrl(name)} />
    {#if !readOnly}
      <button
        class="quiet withicon danger"
        onclick={remove}
        title="delete this prompt from the library"
      >
        <Trash2 size={14} />Delete
      </button>
    {/if}
  {/if}
  <button
    class="withicon"
    class:dirtybtn={shell.dirty}
    onclick={save}
    disabled={shell.busy}
    title="write to the prompt directory under the name below (Ctrl+S)"
  >
    <Save size={14} />Save{#if shell.dirty}<span class="dirtydot"></span>{/if}
  </button>
</div>

<div class="savebar muted">
  saving as
  <FolderPicker
    bind:folder={shell.folder}
    bind:newFolder={shell.newFolder}
    {folders}
    newFolderTitle="name for the new folder at the root of the prompt directory"
  />
  {#if shell.folder === NEW_FOLDER}<span>/</span>{/if}
  <input class="savename" bind:value={shell.saveName} placeholder="MyPrompt" />
  <span class="dirhint">.json in {promptDir}</span>
  <span class="flex"></span>
  {#if shell.savePath()}
    <code class="refhint" title="use the stored prompt from any workflow"
      >prompt:{shell.savePath()}</code
    >
    <CopyButton
      text={reference(PROMPT, shell.savePath() ?? '')}
      title="copy reference to clipboard"
    />
  {/if}
</div>

<EditorBody
  view={shell.view}
  jsonDraft={shell.jsonDraft}
  onjson={(raw) => shell.applyJson(raw)}
  schema="prompt"
  hint="Schema-aware: completion, hover docs and validation come from the prompt schema. Changes apply when the editor loses focus."
  stickyTop="66px"
>
  {#snippet form()}
    <!-- Queried by .panelgrid below, so the panels stack on their own
         column's width - the split view narrows the form well before the
         viewport -->
    <div class="promptform">
      <div class="panelgrid">
        <div class="panel">
          <h2>Prompt</h2>
          <label class="fieldlabel" for="prompt-text">text</label>
          <textarea
            id="prompt-text"
            class="prompttext"
            rows="8"
            spellcheck="true"
            value={shell.doc.text ?? ''}
            placeholder="the prompt itself - what prompt:{shell.savePath() ??
              'name'} resolves to"
            onchange={(e) => (shell.doc.text = e.currentTarget.value)}
          ></textarea>
          <label class="fieldlabel" for="prompt-negative">negative prompt</label
          >
          <textarea
            id="prompt-negative"
            rows="2"
            spellcheck="true"
            value={shell.doc.negative_prompt ?? ''}
            placeholder="optional - for models that take one"
            onchange={(e) => setField('negative_prompt', e.currentTarget.value)}
          ></textarea>
          <div class="metagrid">
            <label for="prompt-desc">description</label>
            <input
              id="prompt-desc"
              spellcheck="true"
              value={shell.doc.description ?? ''}
              placeholder="shown on the prompt's library card"
              onchange={(e) => setField('description', e.currentTarget.value)}
            />
            <label for="prompt-model">intended model</label>
            <Suggest
              id="prompt-model"
              suggestions={intendedModels}
              value={shell.doc.intended_model ?? ''}
              placeholder="e.g. minimax-h3 - badges the card, preselects the enhancer"
              onchange={(value) => {
                setField('intended_model', value)
                enhancer.preselect(shell.doc.intended_model)
              }}
            />
            <label for="prompt-tags">tags</label>
            <input
              id="prompt-tags"
              value={(shell.doc.tags ?? []).join(', ')}
              placeholder="comma-separated, for filtering the library"
              onchange={(e) => setTags(e.currentTarget.value)}
            />
          </div>
          {#if shell.doc.enhanced?.model}
            <p class="muted provenance" title={shell.doc.enhanced.idea}>
              <Sparkles size={13} /> enhanced by {shell.doc.enhanced.model}
            </p>
          {/if}
        </div>

        <EnhancePanel {enhancer} onuse={useResult} />
      </div>
    </div>
  {/snippet}
</EditorBody>

<style>
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.6rem;
    margin-bottom: 0.4rem;
  }
  .head h1 {
    font-size: 1.1rem;
    margin: 0;
  }
  .savebar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.4rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
  }
  .savename {
    max-width: 200px;
  }
  /* a directory is one unbroken token, so let it wrap anywhere rather than
     push the bar past a phone-width viewport */
  .dirhint {
    min-width: 0;
    overflow-wrap: anywhere;
  }
  .refhint {
    font-size: 0.8rem;
    color: var(--accent);
  }
  .promptform {
    container-type: inline-size;
  }
  .panelgrid {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 1.1rem;
    align-items: start;
  }
  @container (max-width: 900px) {
    .panelgrid {
      grid-template-columns: 1fr;
    }
  }
  .panel {
    margin-bottom: 1rem;
    min-width: 0;
  }
  .panel h2 {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .fieldlabel {
    display: block;
    font-weight: 600;
    color: var(--muted);
    font-size: 0.85rem;
    margin: 0.6rem 0 0.25rem;
  }
  textarea {
    width: 100%;
    resize: vertical;
  }
  .prompttext {
    font-size: 0.95rem;
  }
  .metagrid {
    display: grid;
    grid-template-columns: minmax(120px, 40%) minmax(0, 1fr);
    gap: 0.5rem 0.8rem;
    align-items: center;
    margin-top: 0.6rem;
  }
  @container (max-width: 420px) {
    .metagrid {
      grid-template-columns: minmax(0, 1fr);
      gap: 0.2rem;
    }
  }
  .metagrid label {
    font-weight: 600;
    color: var(--muted);
    font-size: 0.85rem;
  }
  .metagrid textarea {
    align-self: stretch;
  }
  .provenance {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.8rem;
    margin: 0.6rem 0 0;
  }
  .dirtydot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--warn);
    margin-left: 0.15rem;
  }
</style>
