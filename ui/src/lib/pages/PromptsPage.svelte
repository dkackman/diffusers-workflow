<script lang="ts">
  import { ChevronDown, ChevronRight, Plus, X } from '@lucide/svelte'
  import { api } from '../api'
  import type { PromptDetail } from '../types'

  let prompts = $state<string[]>([])
  let details = $state<Record<string, PromptDetail>>({})
  let promptDir = $state('')
  let filter = $state('')
  let error = $state('')

  const STORAGE_KEY = 'dw-collapsed-prompt-folders'

  function readCollapsed(): Record<string, boolean> {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}')
    } catch {
      return {}
    }
  }

  let collapsed = $state<Record<string, boolean>>(readCollapsed())

  let showHint = $state(
    (() => {
      try {
        return localStorage.getItem('dw-prompt-hint-dismissed') !== '1'
      } catch {
        return true
      }
    })(),
  )

  function dismissHint() {
    showHint = false
    try {
      localStorage.setItem('dw-prompt-hint-dismissed', '1')
    } catch {
      /* session-only dismissal */
    }
  }

  function toggle(group: string) {
    collapsed[group] = !collapsed[group]
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(collapsed))
    } catch {
      /* private mode etc. - collapse still works for the session */
    }
  }

  $effect(() => {
    api
      .listPrompts()
      .then((result) => {
        prompts = result.prompts
        details = result.details ?? {}
        promptDir = result.prompt_dir
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
  const groupOf = (name: string) =>
    name.includes('/') ? name.split('/')[0] : ''
  const groups = $derived(
    [...new Set(visible.map(groupOf))].sort((a, b) => a.localeCompare(b)),
  )
  const inGroup = (group: string) =>
    visible.filter((name) => groupOf(name) === group)
  // While filtering, everything stays visible - a collapsed folder hiding
  // matches would make the filter look broken
  const isOpen = (group: string) => filter !== '' || !collapsed[group]

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

{#if showHint}
  <div class="hintbar muted">
    <span
      >Write a prompt once, reuse it anywhere: a workflow argument set to
      prompt:name loads its text at run time.</span
    >
    <button
      class="quiet icon"
      onclick={dismissHint}
      title="dismiss"
      aria-label="dismiss this hint"
    >
      <X size={13} />
    </button>
  </div>
{/if}

{#each groups as group (group)}
  {#if group}
    <div class="grouprow">
      <button
        class="group"
        onclick={() => toggle(group)}
        title={isOpen(group) ? 'collapse this folder' : 'expand this folder'}
      >
        {#if isOpen(group)}<ChevronDown size={14} />{:else}<ChevronRight
            size={14}
          />{/if}
        {group}/ <span class="muted">({inGroup(group).length})</span>
      </button>
      <a
        class="groupnew"
        href="#/prompt-edit"
        onclick={() => sessionStorage.setItem('dw-prompt-editor-folder', group)}
        title="new prompt in {group}/"
        aria-label="new prompt in {group}/"
      >
        <Plus size={13} />
      </a>
    </div>
  {/if}
  {#if isOpen(group)}
    <div class="grid">
      {#each inGroup(group) as name (name)}
        {@const detail = details[name]}
        <a
          class="card panel"
          href={href(name)}
          title={detail?.description || undefined}
        >
          <span class="cardtop">
            <span class="cardname"
              >{group ? name.split('/').slice(1).join('/') : name}</span
            >
            {#if detail?.intended_model}
              <button
                class="chip modelchip"
                onclick={(e) => {
                  e.preventDefault()
                  filter = detail.intended_model
                }}
                title="written for {detail.intended_model} - click to filter"
              >
                {detail.intended_model}
              </button>
            {/if}
          </span>
          {#if detail?.description}
            <span class="carddesc muted">{detail.description}</span>
          {/if}
          {#if detail?.tags?.length}
            <span class="cardtags">
              {#each detail.tags as tag (tag)}
                <button
                  class="chip"
                  onclick={(e) => {
                    e.preventDefault()
                    filter = tag
                  }}
                  title="click to filter by this tag"
                >
                  {tag}
                </button>
              {/each}
            </span>
          {/if}
        </a>
      {/each}
    </div>
  {/if}
{/each}

<style>
  .head {
    display: flex;
    align-items: center;
    gap: 1rem;
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
  .hintbar {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    border: 1px dashed var(--line);
    border-radius: 6px;
    padding: 0.45rem 0.7rem;
    font-size: 0.85rem;
    margin-bottom: 1rem;
  }
  .hintbar span {
    flex: 1;
  }
  .hintbar .icon {
    display: inline-flex;
    padding: 0.2rem 0.3rem;
  }
  .grouprow {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 1.2rem 0 0.5rem;
  }
  .grouprow .group {
    margin: 0;
  }
  .groupnew {
    display: inline-flex;
    align-items: center;
    padding: 0.15rem;
    color: var(--muted);
    border: 1px solid transparent;
    border-radius: 4px;
    opacity: 0;
    transition: opacity 0.15s ease;
  }
  .grouprow:hover .groupnew {
    opacity: 1;
  }
  .groupnew:hover {
    color: var(--accent);
    border-color: var(--line);
  }
  .group {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    background: none;
    border: none;
    color: var(--muted);
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0;
    margin: 1.2rem 0 0.5rem;
    cursor: pointer;
  }
  .group:hover {
    color: var(--ink);
    filter: none;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 0.6rem;
  }
  .card {
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
