<script lang="ts">
  import { MessageSquareText, Plus } from '@lucide/svelte'
  import { api } from '../api'
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import FolderGroups from '../FolderGroups.svelte'
  import { leafOf } from '../grouping'
  import HintBar from '../HintBar.svelte'
  import { sharedHref } from '../routes'
  import { writableRoot } from '../libraries'
  import type { PromptDetail } from '../types'

  let filter = $state('')

  // A prompt an agent saves shows within one feed interval, marked
  const list = resource(() => api.listPrompts(), {
    watch: ['prompts'],
    rows: (data) => data.prompts.map((name) => [name, data.details[name]]),
  })
  const prompts = $derived(list.data?.prompts ?? [])
  const details = $derived<Record<string, PromptDetail>>(
    list.data?.details ?? {},
  )
  const promptDir = $derived(writableRoot(list.data?.libraries ?? []))

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

  const href = (name: string) => sharedHref('prompt-edit', ...name.split('/'))
</script>

{#snippet search()}
  <input placeholder="filter…" bind:value={filter} class="filter" />
{/snippet}
<PageHeader title="Prompts" filter={search}>
  {#snippet actions()}<a
      class="quiet icon plain"
      href={sharedHref('prompt-edit')}
      title="new prompt"
      aria-label="new prompt"
    >
      <Plus size={15} />
    </a>{/snippet}
</PageHeader>

<HintBar storageKey="prompt-hint-dismissed">
  Write a prompt once, reuse it anywhere: a workflow argument set to prompt:name
  loads its text at run time.
</HintBar>

<LoadState
  of={list}
  label="prompts"
  empty={visible.length === 0}
  filtered={filter !== '' && prompts.length > 0}
  onclear={() => (filter = '')}
>
  {#snippet emptyIcon()}<MessageSquareText
      size={36}
      strokeWidth={1.5}
    />{/snippet}
  {#snippet emptyText()}No prompts yet — the + above creates the first one.{/snippet}
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
      <!-- The link is an overlay rather than a wrapper: buttons may not
         nest inside an anchor, and the chips are real buttons -->
      <div class="card panel" class:row-changed={list.changed.has(name)}>
        <a
          class="cardcover"
          href={href(name)}
          aria-label="edit {name}"
          title={detail?.description || detail?.text || undefined}
        ></a>
        <span class="cardtop">
          <span class="cardname">{leafOf(name)}</span>
          {#if details[name]?.origin && details[name].origin !== 'workspace'}
            <span
              class="origin muted"
              title="read-only: from the {details[name].origin} library"
            >
              {details[name].origin}
            </span>
          {/if}
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
          <span class="carddesc muted">
            {detail.description || detail.text}
          </span>
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
</LoadState>

<!-- Outside LoadState, so an empty, filtered or failed page still says it -->
{#if promptDir}
  <p class="dir muted">
    read from <span class="path">{promptDir}</span>
  </p>
{/if}

<style>
  .origin {
    font-size: 0.75rem;
  }
  /* An absolute path is worth knowing and not worth the slot beside the
     title - and as one unbreakable token it was pushing the page sideways
     on a phone */
  .dir {
    margin-top: 2.5rem;
    font-size: var(--t-xs);
  }
  .path {
    font-family: var(--font-mono);
    word-break: break-all;
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
  /* A tag is a button, and the global button rule would paint it accent-ink
     on the panel - invisible in either theme - so the text colour is set here */
  .chip {
    cursor: pointer;
    border: 1px solid var(--line);
    background: var(--panel-2);
    color: var(--muted);
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
