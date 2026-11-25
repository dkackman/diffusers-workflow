<script lang="ts">
  import type { PromptDefinition } from '../types'
  import { untrack } from 'svelte'
  import Suggest from '../ui/Suggest.svelte'
  import { isNameSegment } from '../names'
  import {
    DocumentEditor,
    EDITOR_VIEWS,
    NEW_FOLDER,
    editorKeys,
    type EditorView,
  } from '../editorShell.svelte'
  import Segmented from '../ui/Segmented.svelte'
  import DocumentFileBar from '../editor/DocumentFileBar.svelte'
  import EditorBody from '../editor/EditorBody.svelte'
  import EnhancePanel from '../editor/EnhancePanel.svelte'
  import { Enhancer } from '../enhancer.svelte'
  import { PROMPT, reference } from '../references'
  import { Copy, Save, Sparkles, Trash2 } from '@lucide/svelte'
  import { api } from '../api'
  import { ApiError } from '../apiErrors'
  import {
    deleteWarning,
    openPrompt,
    savePromptAt,
    watchPrompt,
    withSavedPrompt,
  } from '../promptFile'
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import LoadFailed from '../ui/LoadFailed.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import { writableRoot } from '../libraries'
  import DownloadLink from '../DownloadLink.svelte'
  import { go } from '../router.svelte'
  import { SECTION_LABELS, sharedHref } from '../routes'
  import { notify } from '../toast'
  import { action } from '../action.svelte'
  import { destroyConfirm } from '../destructive'
  import { loadPromptLibrary } from '../promptlib.svelte'
  import { groupOf } from '../grouping'
  import { DocumentRoute } from '../documentRoute.svelte'
  import { emptyPrompt, parseTags } from '../prompts'
  import type { PromptDetail } from '../types'

  let { name = '' }: { name?: string } = $props()

  const shell = watchPrompt(
    new DocumentEditor<PromptDefinition>({
      viewKey: 'dw-prompt-editor-view',
      views: ['form', 'split', 'json'],
      label: (): string => shell.savePath() ?? 'this prompt',
    }),
  )
  shell.doc = emptyPrompt()
  // The listing: the folder picker's files, where a new prompt saves, and
  // each prompt's intended model for the enhancer
  const listing = resource(() => api.listPrompts(), { watch: ['prompts'] })
  const promptDir = $derived(
    listing.data ? writableRoot(listing.data.libraries) : '',
  )
  const promptFiles = $derived(listing.data?.prompts ?? [])
  const promptDetails = $derived<Record<string, PromptDetail>>(
    listing.data?.details ?? {},
  )
  // A prompt from a read-only examples library: it can be edited and saved
  // (the save lands in this workspace's library, shadowing it) but not
  // deleted, the way a read-only workflow behaves
  const readOnly = $derived(!shell.writable)
  // Existing folders, from the listing - one level is the designed depth
  const folders = $derived(
    [...new Set(promptFiles.map(groupOf).filter(Boolean))].sort(),
  )

  const views = EDITOR_VIEWS.filter((v) => v.value !== 'flow')

  // ------------------------------------------------------------- enhancer

  const enhancer = new Enhancer()
  const intendedModels = $derived(enhancer.intendedModels(promptDetails))

  // The idea the last read put in the enhancer: an idea still equal to it was
  // not typed, so a newer read may replace it
  let readIdea = ''

  // The server's copy replaced the prompt: its idea is the enhancer's, unless
  // the person has typed one of their own
  $effect(() => {
    if (!shell.remoteLoads) return
    untrack(() => {
      if (enhancer.idea !== readIdea) return
      readIdea = enhancer.idea = shell.doc.enhanced?.idea ?? ''
    })
  })

  function useResult() {
    const { text, enhanced } = enhancer.takeResult()
    shell.doc.text = text
    shell.doc.enhanced = enhanced
  }

  // ---------------------------------------------------------------- lifecycle

  const place = new DocumentRoute(shell, {
    name: () => name,
    href: (path) => sharedHref('prompt-edit', ...path.split('/')),
    open,
    startDraft,
  })

  /** A new prompt: a duplicate's hand-off, else the blank starter. */
  function startDraft() {
    const imported = shell.takeImport(
      'dw-prompt-editor-import',
      'dw-prompt-editor-folder',
    )
    if (imported) notify.success('Duplicated - save under a new name')
    shell.startNew(imported ?? emptyPrompt())
    shell.saveName = ''
  }

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
  async function open(path: string, stale: () => boolean) {
    const prompt = await openPrompt(shell, path, stale)
    if (!prompt) return false
    readIdea = enhancer.idea = prompt.enhanced?.idea ?? ''
    enhancer.preselect(shell.doc.intended_model)
    return true
  }

  // ---------------------------------------------------------------- editing

  function setField(
    key: 'description' | 'intended_model' | 'negative_prompt',
    value: string,
  ) {
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

  function save() {
    if (!shell.canWrite) return
    shell.flush()
    const blocker = saveBlocker()
    if (blocker) {
      notify.error(blocker)
      return
    }
    void saving.run(shell.savePath()!)
  }
  const onKeydown = editorKeys(shell, { save })
  // An action: Save twice (or Ctrl+S twice) writes once
  const saving = action(
    async (path: string) => {
      shell.busy = true
      try {
        const outcome = await savePromptAt(shell, path)
        if (outcome === 'cancelled') return null
        shell.commitNewFolder()
      } finally {
        shell.busy = false
      }
      // Same as the workflow editor: the picker lists folders from the
      // listing, so a newly created one must be added or the select resets
      if (listing.data) listing.adopt(withSavedPrompt(listing.data, path))
      // and its saved time read back quietly, as the workflow editor does
      void listing.refresh()
      loadPromptLibrary()
      // The URL names the prompt now on screen: a first save, or a save as
      place.saved(path)
      return path
    },
    {
      doing: (path) => `save ${path}`,
      success: (path) => `Saved to ${path}`,
    },
  )

  // The workflows a delete would break are read on the click that asks
  const removing = action<[name: string, warning: string], void>(
    async (name) => {
      await api.deletePrompt(name)
      loadPromptLibrary()
      go('shared', 'prompts')
    },
    {
      doing: (name) => `delete ${name}`,
      confirm: (name, warning) =>
        destroyConfirm({
          name: `${name}.json`,
          noun: 'prompt',
          consequence: warning.trim(),
        }),
    },
  )
  // A Delete pressed again while the warning is read does nothing; one
  // pressed while the delete runs is the action's busy guard
  let warning = false
  async function remove() {
    const name = place.current
    if (!name || removing.busy || warning) return
    warning = true
    try {
      const said = await deleteWarning(name)
      if (!removing.busy) await removing.run(name, said)
    } finally {
      warning = false
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

<!-- The editing surface: a flush commits the field focused in it -->
<div class="editing" bind:this={shell.root}>
  <PageHeader
    title={place.current || 'New prompt'}
    back={{ href: sharedHref('prompts'), label: SECTION_LABELS.shared.prompts }}
  >
    {#if shell.dirty}
      <span
        class="chip unsaved"
        title="this definition differs from the last save"
      >
        unsaved
      </span>
    {/if}
    {#snippet actions()}
      <Segmented
        label="editor view"
        value={shell.view}
        options={views}
        onselect={(view) => shell.setView(view as EditorView)}
      />
      {#if place.current}
        <button
          class="quiet withicon"
          onclick={duplicate}
          title="open a copy in the editor to save under a new name"
        >
          <Copy size={14} />Duplicate
        </button>
        <DownloadLink href={api.promptDownloadUrl(place.current)} />
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
        <Save size={14} />Save{#if shell.dirty}<span class="dirtydot"
          ></span>{/if}
      </button>
    {/snippet}
  </PageHeader>

  {#if shell.loadState === 'failed'}
    <LoadFailed
      label={name}
      error={shell.loadError}
      onretry={() => place.retry()}
      oncreate={shell.loadError instanceof ApiError &&
      shell.loadError.status === 404
        ? () => shell.startNew(emptyPrompt())
        : undefined}
    />
  {:else}
    <!-- The listing only feeds the folder picker: loading shows nothing -->
    <LoadState of={listing} label="the prompt list">
      {#snippet skeleton()}{/snippet}
    </LoadState>
    <DocumentFileBar
      ed={shell}
      dir={promptDir}
      {folders}
      noun="prompt"
      described={false}
      changed={listing.data?.details?.[place.current]?.mtime}
      reference={shell.savePath()
        ? reference(PROMPT, shell.savePath() ?? '')
        : ''}
    />

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
                  setField('negative_prompt', e.currentTarget.value)}
              ></textarea>
              <div class="formgrid metagrid">
                <label for="prompt-desc">description</label>
                <input
                  id="prompt-desc"
                  spellcheck="true"
                  value={shell.doc.description ?? ''}
                  placeholder="shown on the prompt's library card"
                  oninput={(e) =>
                    setField('description', e.currentTarget.value)}
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
</div>

<style>
  /* No box of its own: the sticky head keeps its containing block */
  .editing {
    display: contents;
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
    margin-top: 0.6rem;
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
