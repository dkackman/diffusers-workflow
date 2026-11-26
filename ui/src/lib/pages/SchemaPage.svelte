<script lang="ts">
  import { api } from '../api'
  import { matchWords } from '../matchWords'
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import SchemaNode, { type SchemaValue } from '../schema/SchemaNode.svelte'

  let filter = $state('')

  // The schema the running server validates against: it changes only with
  // the server, whose restart raises the epoch and reads it again
  const read = resource(() => api.getSchema() as Promise<SchemaValue>, {
    watch: [],
  })
  const schema = $derived(read.data ?? null)

  // The raw JSON link is the schema already read, as a blob: a plain link to
  // /api/schema carries no Authorization header, so a server with a token
  // refuses it
  let raw = $state('')
  $effect(() => {
    if (!schema) return
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(schema, null, 2)], {
        type: 'application/json',
      }),
    )
    raw = url
    return () => {
      raw = ''
      URL.revokeObjectURL(url)
    }
  })

  const definitions = $derived(
    Object.entries(schema?.$defs ?? {}) as [string, SchemaValue][],
  )
  const shownDefinitions = $derived(
    definitions.filter(([name]) => matchWords(filter, name)),
  )
</script>

<PageHeader title="Schema">
  {#snippet actions()}{#if raw}<a
        class="muted"
        href={raw}
        target="_blank"
        rel="noopener"
      >
        raw JSON
      </a>{/if}{/snippet}
</PageHeader>

<LoadState of={read} label="the schema" />

{#if schema}
  <p class="muted intro">
    {schema.description} This is the schema the running server validates against —
    it also powers completion and hover documentation in the JSON editor views.
  </p>

  <div class="panel">
    <h2>Document root</h2>
    <SchemaNode name="workflow" node={schema} root={schema} open />
  </div>

  <div class="defshead">
    <h2>Definitions</h2>
    <input class="filter" placeholder="filter…" bind:value={filter} />
  </div>
  {#each shownDefinitions as [name, definition] (name)}
    <div class="panel def" id={'def-' + name}>
      <SchemaNode {name} node={definition} root={schema} />
    </div>
  {:else}
    <p class="muted">No definition matches "{filter}".</p>
  {/each}
{/if}

<style>
  .intro {
    max-width: 80ch;
  }
  .defshead {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.8rem;
    margin-top: 1.2rem;
  }
  .defshead input {
    max-width: 200px;
  }
  .panel.def {
    scroll-margin-top: 70px;
    margin-bottom: 0.6rem;
  }
</style>
