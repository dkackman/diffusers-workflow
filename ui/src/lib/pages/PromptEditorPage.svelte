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
  import { ApiError } from '../apiErrors'
  import { openPrompt, savePromptAt } from '../promptFile'
  import LoadFailed from '../ui/LoadFailed.svelte'
  import { writableRoot } from '../libraries'
  import DownloadLink from '../DownloadLink.svelte'
  import { go, renameRoute } from '../router.svelte'
  import { sharedHref } from '../routes'
  import { notify } from '../toast'
  import { confirmDialog } from '../confirm.svelte'
  import { loadPromptLibrary } from '../promptlib.svelte'
  import { groupOf, leafOf } from '../grouping'
  import { emptyPrompt, parseTags, workflowsReferencing } from '../prompts'
  import CopyButton from '../CopyButton.svelte'
  import type { PromptDetail } from '../types'

  let { name = '' }: { name?: string } = $props()

  const shell = new DocumentEditor({
    viewKey: 'dw-prompt-editor-view',
    views: ['form', 'split', 'json'],
    label: (): string => shell.savePath() ?? 'this prompt',
  })
  shell.doc = emptyPrompt()
  let promptDir = $state('')
  let promptFiles = $state<string[]>([])
  let promptDetails = $state<Record<string, PromptDetail>>({})
  // A prompt from a read-only examples library: it can be edited and saved
  // (the save lands in this workspace's library, shadowing it) but not
  // deleted, the way a read-only workflow behaves
  const readOnly = $derived(!shell.writable)
  // The prompt being edited: the route's, until a save names another (or
  // its first) - the URL follows a beat later
  const current = $derived(
    shell.loadState === 'loading' || shell.loadState === 'failed'
      ? name
      : (shell.loadedPath ?? name),
  )
  // The file on screen, so the route catching up with a save does not
  // read it again
  let opened = ''

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
    if (name && opened === name) {
      // the route catching up with a save: this is the prompt already
    } else if (name) {
      shell.saveName = leafOf(name)
      shell.folder = groupOf(name)
      open(name)
    } else {
      opened = ''
      const imported = shell.takeImport(
        'dw-prompt-editor-import',
        'dw-prompt-editor-folder',
      )
      if (imported) notify.success('Duplicated - save under a new name')
      shell.startNew(imported ?? emptyPrompt())
      shell.saveName = ''
    }
  })

  // Apart from the route: the URL following a save must not stop an
  // enhancement in flight
  $effect(() => {
    enhancer.refreshModels()
    enhancer.load().then(() => enhancer.preselect(shell.doc.intended_model))
    return () => enhancer.stop()
  })

  /** Read `path` into the editor; a failure leaves nothing to save. The
   * definition comes back beside where it was found: a prompt from a
   * read-only examples library can be edited and saved (the copy lands
   * here) but not deleted. */
  async function open(path: string) {
    const prompt = await openPrompt(shell, path, () => path !== name)
    if (!prompt) return
    opened = path
    enhancer.idea = prompt.enhanced?.idea ?? ''
    enhancer.preselect(shell.doc.intended_model)
  }

  function onKeydown(event: KeyboardEvent) {
    if (!shell.canWrite) return
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

  // The tags as typed, while the field has focus: the list they parse to
  // drops a trailing comma, which would otherwise vanish mid-word
  let tagsText = $state<string | null>(null)

  function setTags(raw: string) {
    tagsText = raw
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
    if (!shell.canWrite) return
    shell.flush()
    const blocker = saveBlocker()
    if (blocker) {
      notify.error(blocker)
      return
    }
    const path = shell.savePath()!
    shell.busy = true
    try {
      const outcome = await savePromptAt(shell, path)
      if (outcome === 'cancelled') return
      shell.commitNewFolder()
      // Same as the workflow editor: the picker lists folders from the
      // listing, so a newly created one must be added or the select resets
      if (!promptFiles.includes(path)) promptFiles = [...promptFiles, path]
      notify.success(`Saved to ${path}`)
      loadPromptLibrary()
      // The URL names the prompt now on screen: a first save, or a save as
      opened = path
      if (path !== name)
        renameRoute(sharedHref('prompt-edit', ...path.split('/')))
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e))
    } finally {
      shell.busy = false
    }
  }

  async function remove() {
    const name = current
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
  <h1>{current || 'New prompt'}</h1>
  <span class="flex"></span>
  <ViewSwitch
    view={shell.view}
    options={[...VIEWS]}
    onselect={(view) => shell.setView(view)}
  />
  {#if current}
    <button
      class="quiet withicon"
      onclick={duplicate}
      title="open a copy in the editor to save under a new name"
    >
      <Copy size={14} />Duplicate
    </button>
    <DownloadLink href={api.promptDownloadUrl(current)} />
    {#if !readOnly}
      <button
        class="quiet icon danger"
        onclick={remove}
        title="delete this prompt from the library"
        aria-label="delete this prompt from the library"
      >
        <Trash2 size={14} />
      </button>
    {/if}
  {/if}
  <button
    class="withicon"
    onclick={save}
    disabled={shell.busy || !shell.canWrite}
    title="write to the prompt directory under the name below (Ctrl+S)"
  >
    <Save size={14} />Save{#if shell.dirty}<span class="dirtydot"></span>{/if}
  </button>
</div>

{#if shell.loadState === 'failed'}
  <LoadFailed
    label={name}
    error={shell.loadError}
    onretry={() => open(name)}
    oncreate={shell.loadError instanceof ApiError &&
    shell.loadError.status === 404
      ? () => shell.startNew(emptyPrompt())
      : undefined}
  />
{:else}
  <div class="savebar muted">
    saving as
    <FolderPicker
      bind:folder={shell.folder}
      bind:newFolder={shell.newFolder}
      {folders}
      newFolderTitle="name for the new folder at the root of the prompt directory"
    />
    {#if shell.folder === NEW_FOLDER}<span>/</span>{/if}
    <input
      class="savename"
      bind:value={shell.saveName}
      placeholder="MyPrompt"
    />
    <span class="dirhint">.json in {promptDir}</span>
    <span class="flex"></span>
    {#if shell.savePath()}
      <code class="refhint" title="use the stored prompt from any workflow">
        prompt:{shell.savePath()}
      </code>
      <CopyButton
        text={reference(PROMPT, shell.savePath() ?? '')}
        title="copy reference to clipboard"
      />
    {/if}
  </div>

  <EditorBody
    view={shell.view}
    jsonDraft={shell.jsonDraft}
    onjson={(raw, o) => shell.applyJson(raw, o)}
    register={(f) => shell.registerFlush(f)}
    schema="prompt"
    hint="Schema-aware: completion, hover docs and validation come from the prompt schema. Changes apply as you type."
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
              oninput={(e) => (shell.doc.text = e.currentTarget.value)}
            ></textarea>
            <label class="fieldlabel" for="prompt-negative">
              negative prompt
            </label>
            <textarea
              id="prompt-negative"
              rows="2"
              spellcheck="true"
              value={shell.doc.negative_prompt ?? ''}
              placeholder="optional - for models that take one"
              oninput={(e) =>
                setField('negative_prompt', e.currentTarget.value)}></textarea>
            <div class="metagrid">
              <label for="prompt-desc">description</label>
              <input
                id="prompt-desc"
                spellcheck="true"
                value={shell.doc.description ?? ''}
                placeholder="shown on the prompt's library card"
                oninput={(e) => setField('description', e.currentTarget.value)}
              />
              <label for="prompt-model">intended model</label>
              <Suggest
                id="prompt-model"
                suggestions={intendedModels}
                value={shell.doc.intended_model ?? ''}
                placeholder="e.g. minimax-h3 - badges the card, preselects the enhancer"
                oninput={(value) => setField('intended_model', value)}
                onchange={(value) => {
                  setField('intended_model', value)
                  enhancer.preselect(shell.doc.intended_model)
                }}
              />
              <label for="prompt-tags">tags</label>
              <input
                id="prompt-tags"
                value={tagsText ?? (shell.doc.tags ?? []).join(', ')}
                placeholder="comma-separated, for filtering the library"
                oninput={(e) => setTags(e.currentTarget.value)}
                onblur={() => (tagsText = null)}
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
{/if}

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
</style>
