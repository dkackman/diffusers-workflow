<script lang="ts">
  import { MessageSquareText, Plus } from '@lucide/svelte'
  import { api } from '../api'
  import Empty from '../Empty.svelte'
  import FolderGroups from '../FolderGroups.svelte'
  import HintBar from '../HintBar.svelte'
  import type { PromptDetail } from '../types'

  let prompts = $state<string[]>([])
  let details = $state<Record<string, PromptDetail>>({})
  let promptDir = $state('')
  let filter = $state('')
  let error = $state('')
  let loaded = $state(false)

  $effect(() => {
    api
      .listPrompts()
      .then((result) => {
        prompts = result.prompts
        details = result.details ?? {}
        promptDir = result.prompt_dir
        loaded = true
      })
      .catch((e) => (error = e.message))
  })

  const visible = $derived(
    prompts.filter((name) => {
      const needle = filter.toLowerCase()
      const detail = details[name]
      return (
        name.toLowerCase().includes(needle) ||
        (detail?.description ?? '').toLowerCase().includes(needle) ||
        (detail?.intended_model ?? '').toLowerCase().includes(needle) ||
        (detail?.tags ?? []).some((tag) => tag.toLowerCase().includes(needle))
      )
    }),
  )

  const href = (name: string) =>
    '#/prompt-edit/' + name.split('/').map(encodeURIComponent).join('/')
</script>

<div class="head">
  <h1>Prompts</h1>
  <span class="muted">{promptDir}</span>
  <input placeholder="filter…" bind:value={filter} class="filter" />
  <a class="newlink" href="#/prompt-edit" title="new prompt"
    ><Plus size={15} /></a
  >
</div>

{#if error}
  <p class="muted">Could not load prompts: {error}</p>
{/if}

<HintBar storageKey="prompt-hint-dismissed">
  Write a prompt once, reuse it anywhere: a workflow argument set to prompt:name
  loads its text at run time.
</HintBar>

<FolderGroups
  names={visible}
  collapseKey="collapsed-prompt-folders"
  filterActive={filter !== ''}
  newHref="#/prompt-edit"
  onnewingroup={(group) =>
    sessionStorage.setItem('dw-prompt-editor-folder', group)}
>
  {#snippet card(name)}
    {@const detail = details[name]}
    {@const group = name.includes('/') ? name.split('/')[0] : ''}
    <!-- The link is an overlay rather than a wrapper: buttons may not
         nest inside an anchor, and the chips are real buttons -->
    <div class="card panel">
      <a
        class="cardcover"
        href={href(name)}
        aria-label="edit {name}"
        title={detail?.description || detail?.text || undefined}
      ></a>
      <span class="cardtop">
        <span class="cardname"
          >{group ? name.split('/').slice(1).join('/') : name}</span
        >
        {#if detail?.intended_model}
          <button
            class="chip modelchip"
            onclick={() => (filter = detail.intended_model)}
            title="written for {detail.intended_model} - click to filter"
          >
            {detail.intended_model}
          </button>
        {/if}
      </span>
      {#if detail?.description || detail?.text}
        <span class="carddesc muted">{detail.description || detail.text}</span>
      {/if}
      {#if detail?.tags?.length}
        <span class="cardtags">
          {#each detail.tags as tag (tag)}
            <button
              class="chip"
              onclick={() => (filter = tag)}
              title="click to filter by this tag"
            >
              {tag}
            </button>
          {/each}
        </span>
      {/if}
    </div>
  {/snippet}
</FolderGroups>

{#if loaded && prompts.length === 0}
  <Empty>
    {#snippet icon()}<MessageSquareText size={36} strokeWidth={1.5} />{/snippet}
    No prompts yet — the + above creates the first one.
  </Empty>
{:else if loaded && visible.length === 0}
  <p class="muted">Nothing matches "{filter}".</p>
{/if}

<style>
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 1rem;
    margin-bottom: 1rem;
  }
  .filter {
    max-width: 220px;
    margin-left: auto;
  }
  .newlink {
    display: inline-flex;
    align-items: center;
    padding: 0.4rem;
    border: 1px solid var(--line);
    border-radius: 6px;
    color: var(--muted);
  }
  .newlink:hover {
    border-color: var(--accent);
    color: var(--accent);
  }
  .card {
    position: relative;
    color: var(--ink);
    font-weight: 600;
    padding: 0.7rem 0.9rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .card:hover {
    border-color: var(--accent);
  }
  .cardcover {
    position: absolute;
    inset: 0;
    border-radius: inherit;
  }
  .chip {
    position: relative;
  }
  .cardtop {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
  }
  .cardname {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .carddesc {
    font-weight: 400;
    font-size: 0.78rem;
    line-height: 1.35;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .cardtags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
  }
  .chip {
    cursor: pointer;
    border: 1px solid var(--line);
    background: var(--panel-2);
    font-size: 0.7rem;
    font-weight: 500;
    padding: 0.05rem 0.45rem;
  }
  .chip:hover {
    border-color: var(--accent);
    color: var(--accent);
  }
  .modelchip {
    flex-shrink: 0;
    color: var(--accent);
  }
</style>
