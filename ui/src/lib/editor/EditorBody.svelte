<script lang="ts">
  import type { Snippet } from 'svelte'
  import JsonEditor from './JsonEditor.svelte'
  import type { EditorView } from '../editorShell.svelte'

  // The editor's body for the view picked: the JSON alone, the form alone,
  // or the two side by side
  let {
    view,
    jsonDraft,
    onjson,
    register,
    schema = 'workflow',
    hint,
    stickyTop,
    form,
  }: {
    view: EditorView
    jsonDraft: string
    onjson: (raw: string, options?: { quiet?: boolean }) => void
    register?: (flush: () => void) => () => void
    schema?: 'workflow' | 'prompt'
    hint: string
    // Where the split view's JSON column pins: below the app header, and
    // below the editor's own toolbar when it has a sticky one
    stickyTop: string
    form: Snippet
  } = $props()
</script>

{#if view === 'json'}
  <JsonEditor
    value={jsonDraft}
    onchange={onjson}
    {register}
    height="560px"
    {schema}
  />
  <p class="muted hint">{hint}</p>
{:else}
  <div class="editwrap" class:splitcols={view === 'split'}>
    <div class="formcol">
      {@render form()}
    </div>
    {#if view === 'split'}
      <div class="jsoncol" style:top={stickyTop}>
        <JsonEditor
          value={jsonDraft}
          onchange={onjson}
          {register}
          height="calc(100vh - 200px)"
          {schema}
        />
      </div>
    {/if}
  </div>
{/if}

<style>
  .editwrap.splitcols {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(360px, 44%);
    gap: 1.1rem;
    align-items: start;
  }
  .jsoncol {
    position: sticky;
  }
  @media (max-width: 1100px) {
    .editwrap.splitcols {
      grid-template-columns: 1fr;
    }
    .jsoncol {
      position: static;
    }
  }
</style>
