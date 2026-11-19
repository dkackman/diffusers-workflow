<script lang="ts">
  import { setNumber } from '../editor'
  import { api } from '../api'
  import type { LoraRow } from '../types'
  import { Plus, Trash2 } from '@lucide/svelte'

  let { pipeline = $bindable() }: { pipeline: Record<string, any> } = $props()

  const loras = $derived(pipeline.loras ?? [])

  // Catalog suggestions (GET /api/loras, catalog-only - never the Hub). Only
  // a literal Hub repo id is asked about: a `variable:` base resolves only
  // for a saved workflow, never for the draft being edited
  const REPO_ID = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/
  const STATUS_ORDER: Record<string, number> = { proven: 0, trial: 1 }
  // A catalog read is a local file scan; one still unsettled after this long
  // has hung (a stalled connection, a stub that never answers) and counts as
  // a failure, so the panel never sits on a bare heading (#1501)
  const CATALOG_TIMEOUT_MS = 8000

  const base = $derived(pipeline.from_pretrained_arguments?.model_name)
  const repo = $derived(
    typeof base === 'string' && REPO_ID.test(base) ? base : undefined,
  )
  const partitionValue = $derived(pipeline.from_pretrained_arguments?.workflow)
  // A reference (`variable:...`) is not a partition the server can filter by
  const partition = $derived(
    typeof partitionValue === 'string' &&
      partitionValue !== '' &&
      !partitionValue.includes(':')
      ? partitionValue
      : undefined,
  )

  let catalog = $state<LoraRow[]>([])
  let catalogState = $state<'idle' | 'loading' | 'ready' | 'error'>('idle')

  $effect(() => {
    const model = repo
    const workflow = partition
    catalog = []
    if (!model) {
      catalogState = 'idle'
      return
    }
    catalogState = 'loading'
    let stale = false
    let giveUp: ReturnType<typeof setTimeout> | undefined
    const timer = setTimeout(() => {
      giveUp = setTimeout(() => {
        if (!stale && catalogState === 'loading') catalogState = 'error'
      }, CATALOG_TIMEOUT_MS)
      api
        .listLoras(workflow ? { model, workflow } : { model })
        .then((body) => {
          if (stale) return
          catalog = body.loras
            .filter((row) => row.status !== 'rejected')
            .sort(
              (a, b) =>
                (STATUS_ORDER[a.status] ?? 2) - (STATUS_ORDER[b.status] ?? 2) ||
                a.name.localeCompare(b.name),
            )
          catalogState = 'ready'
        })
        .catch(() => {
          if (!stale) catalogState = 'error'
        })
        .finally(() => clearTimeout(giveUp))
    }, 300)
    return () => {
      stale = true
      clearTimeout(timer)
      clearTimeout(giveUp)
    }
  })

  function isAdded(row: LoraRow) {
    return loras.some(
      (lora: Record<string, any>) =>
        lora.model_name === row.model_name &&
        (lora.weight_name ?? undefined) === (row.weight_name ?? undefined),
    )
  }

  function addFromCatalog(row: LoraRow) {
    const item: Record<string, any> = { model_name: row.model_name }
    if (row.weight_name) item.weight_name = row.weight_name
    if (row.revision) item.revision = row.revision
    item.scale = row.scale?.default ?? 1.0
    pipeline.loras = [...(pipeline.loras ?? []), item]
  }

  function describeScale(row: LoraRow) {
    if (!row.scale) return ''
    const range = row.scale.range
    return range
      ? `scale ${row.scale.default} (${range[0]}–${range[1]})`
      : `scale ${row.scale.default}`
  }

  function describePartition(row: LoraRow) {
    const value = row.workflow
    if (value === null || value === undefined) return ''
    return `workflow: ${Array.isArray(value) ? value.join(', ') : value}`
  }

  function add() {
    pipeline.loras = [...(pipeline.loras ?? []), { model_name: '', scale: 1.0 }]
  }

  function remove(index: number) {
    pipeline.loras.splice(index, 1)
    if (pipeline.loras.length === 0) delete pipeline.loras
  }
</script>

<div class="loras">
  <section class="catalog" aria-label="From the catalog">
    <h4>From the catalog</h4>
    {#if !repo}
      <p class="note">Catalog suggestions need a literal model repo.</p>
    {:else if catalogState === 'error'}
      <p class="note">Catalog suggestions are unavailable right now.</p>
    {:else if catalogState !== 'ready'}
      <p class="note">Loading catalog suggestions…</p>
    {:else if catalogState === 'ready' && catalog.length === 0}
      <p class="note">No catalog LoRAs for {repo}.</p>
    {:else if catalogState === 'ready'}
      <ul>
        {#each catalog as row (row.name)}
          {@const added = isAdded(row)}
          <li class="suggestion">
            <div class="text">
              <div class="title">
                <span class="name">{row.name}</span>
                <span class="status {row.status}">{row.status}</span>
                {#if describeScale(row)}
                  <span class="meta">{describeScale(row)}</span>
                {/if}
                {#if !partition && describePartition(row)}
                  <span class="meta partition">{describePartition(row)}</span>
                {/if}
              </div>
              <div class="description">{row.description}</div>
              <div class="meta">Use when: {row.use_when}</div>
              {#if row.trigger}
                <div class="meta">Trigger: <code>{row.trigger}</code></div>
              {/if}
            </div>
            <button
              class="quiet withicon"
              disabled={added}
              onclick={() => addFromCatalog(row)}
              title={added
                ? 'this LoRA is already on the step'
                : `add ${row.model_name} to this pipeline`}
              aria-label={added ? `${row.name} added` : `add ${row.name}`}
            >
              {#if added}added{:else}<Plus size={14} />add{/if}
            </button>
          </li>
        {/each}
      </ul>
    {/if}
  </section>

  {#each loras as lora, index (index)}
    <div class="row">
      <input bind:value={lora.model_name} placeholder="org/lora-repo or path" />
      <input
        value={lora.weight_name ?? ''}
        placeholder="weight file (optional)"
        onchange={(e) => {
          const v = e.currentTarget.value
          if (v) lora.weight_name = v
          else delete lora.weight_name
        }}
      />
      <input
        class="scale"
        value={lora.scale ?? ''}
        placeholder="scale"
        onchange={(e) => setNumber(lora, 'scale', e.currentTarget.value)}
      />
      <button
        class="quiet icon"
        onclick={() => remove(index)}
        title="remove this LoRA"
        aria-label="remove this LoRA"
      >
        <Trash2 size={14} />
      </button>
      {#if lora.revision}
        <div class="revision">revision <code>{lora.revision}</code></div>
      {/if}
    </div>
  {/each}
  <button
    class="quiet withicon"
    onclick={add}
    title="add a LoRA adapter to load onto this pipeline"
  >
    <Plus size={14} />add LoRA
  </button>
</div>

<style>
  .loras {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    align-items: flex-start;
  }
  .catalog {
    width: 100%;
  }
  /* Own ink, not inherited: .panel transitions colour (app.css), so text
     inheriting it lags a theme switch while the muted lines jump (#1480) */
  .catalog h4,
  .name,
  .description {
    color: var(--ink);
  }
  .catalog h4 {
    margin: 0 0 0.25rem;
    font-size: 0.85rem;
    font-weight: 600;
  }
  .catalog ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .suggestion {
    display: flex;
    gap: 0.5rem;
    align-items: flex-start;
    justify-content: space-between;
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 0.35rem 0.5rem;
  }
  .text {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
  }
  .title {
    display: flex;
    flex-wrap: wrap;
    gap: 0.2rem 0.5rem;
    align-items: baseline;
  }
  .name {
    font-weight: 600;
    overflow-wrap: anywhere;
  }
  .status {
    font-size: 0.75rem;
    color: var(--muted);
  }
  .status.proven {
    color: var(--good);
  }
  .meta,
  .note,
  .revision {
    font-size: 0.8rem;
    color: var(--muted);
  }
  .note {
    margin: 0;
  }
  .description {
    font-size: 0.85rem;
  }
  .row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) 80px auto;
    gap: 0.5rem;
    width: 100%;
    align-items: center;
  }
  .revision {
    grid-column: 1 / -1;
  }
  /* Two repo fields side by side need ~460px; below that they stack and the
     scale keeps the remove button company on the last line */
  @container (max-width: 460px) {
    .row {
      grid-template-columns: minmax(0, 1fr) auto;
    }
    .row > input:not(.scale) {
      grid-column: 1 / -1;
    }
  }
</style>
