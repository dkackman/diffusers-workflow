<script lang="ts">
  import { MessageSquareText, Plus } from '@lucide/svelte'
  import { api } from '../api'
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import FolderGroups from '../FolderGroups.svelte'
  import { groupOf, leafOf } from '../grouping'
  import HintBar from '../HintBar.svelte'
  import { sharedHref } from '../routes'
  import { writableRoot } from '../libraries'
  import type { PromptDetail } from '../types'
  import { hashState } from '../router.svelte'
  import ListSort, { listSort, newestFirst } from '../ui/ListSort.svelte'
  import Time from '../ui/Time.svelte'

  // In the hash, so Back from a prompt - or a shared link - keeps it
  const filter = hashState('q', '')
  const sort = listSort()

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

  const matching = $derived(
    prompts.filter((name) => {
      const needle = filter.value.toLowerCase()
      const detail = details[name]
      return (
        name.toLowerCase().includes(needle) ||
        (detail?.description ?? '').toLowerCase().includes(needle) ||
        (detail?.intended_model ?? '').toLowerCase().includes(needle) ||
        (detail?.tags ?? []).some((tag) => tag.toLowerCase().includes(needle))
      )
    }),
  )
  // Newest first is one list across the folders
  const newest = $derived(sort.value === 'newest')
  const visible = $derived(
    newest ? newestFirst(matching, (name) => details[name]?.mtime) : matching,
  )

  const href = (name: string) => sharedHref('prompt-edit', ...name.split('/'))
</script>

{#snippet search()}
  <input placeholder="filter…" bind:value={filter.value} class="filter" />
{/snippet}
<PageHeader title="Prompts" filter={search}>
  {#snippet actions()}<ListSort {sort} /><a
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
  filtered={filter.value !== '' && prompts.length > 0}
  onclear={() => (filter.value = '')}
>
  {#snippet emptyIcon()}<MessageSquareText
      size={36}
      strokeWidth={1.5}
    />{/snippet}
  {#snippet emptyText()}No prompts yet — the + above creates the first one.{/snippet}
  <FolderGroups
    names={visible}
    collapseKey="collapsed-prompt-folders"
    filterActive={filter.value !== ''}
    groupOf={newest ? () => '' : undefined}
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
          <span class="cardname" title={name}>{leafOf(name)}</span>
          {#if details[name]?.origin && details[name].origin !== 'workspace'}
            <span
              class="origin muted"
              title="read-only: from the {details[name].origin} library"
            >
              {details[name].origin}
            </span>
          {/if}
        </span>
        <!-- One list across the folders moves the folder onto the card, as
           the Workflows page does -->
        {#if newest && groupOf(name)}
          <span class="cardfolder muted">{groupOf(name)}</span>
        {/if}
        {#if detail?.description || detail?.text}
          <span class="carddesc muted">
            {detail.description || detail.text}
          </span>
        {/if}
        {#if detail?.mtime}
          <span class="stamp muted">
            <Time at={detail.mtime} label="changed" />
          </span>
        {/if}
        {#if detail?.intended_model || detail?.tags?.length}
          <span class="cardtags">
            {#if detail?.intended_model}
              <button
                class="togglechip modelchip"
                class:on={filter.value === detail.intended_model}
                aria-pressed={filter.value === detail.intended_model}
                onclick={() =>
                  (filter.value =
                    filter.value === detail.intended_model
                      ? ''
                      : detail.intended_model)}
                title="written for {detail.intended_model} - click to filter, click again to show all"
              >
                {detail.intended_model}
              </button>
            {/if}
            {#each detail?.tags ?? [] as tag (tag)}
              <button
                class="togglechip"
                class:on={filter.value === tag}
                aria-pressed={filter.value === tag}
                onclick={() => (filter.value = filter.value === tag ? '' : tag)}
                title="click to filter by this tag, click again to show all"
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
  .togglechip {
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
    min-width: 0;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    overflow-wrap: anywhere;
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
  .stamp {
    font-weight: 400;
    font-size: var(--t-xs);
  }
  .modelchip {
    flex-shrink: 0;
    color: var(--accent);
  }
</style>
