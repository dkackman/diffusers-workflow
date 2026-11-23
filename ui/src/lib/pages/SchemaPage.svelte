<script lang="ts">
  import { api } from '../api'
  import { resource } from '../resource.svelte'
  import LoadState from '../ui/LoadState.svelte'
  import SchemaNode, { type SchemaValue } from '../schema/SchemaNode.svelte'

  let filter = $state('')

  // The schema the running server validates against: it changes only with
  // the server, whose restart raises the epoch and reads it again
  const read = resource(() => api.getSchema() as Promise<SchemaValue>, {
    watch: [],
  })
  const schema = $derived(read.data ?? null)

  const definitions = $derived(
    Object.entries(schema?.$defs ?? {}) as [string, SchemaValue][],
  )
  const shownDefinitions = $derived(
    definitions.filter(([name]) =>
      name.toLowerCase().includes(filter.trim().toLowerCase()),
    ),
  )
</script>

<div class="head">
  <h1>Workflow Schema</h1>
  <span class="flex"></span>
  <a class="muted" href="/api/schema" target="_blank" rel="noopener">
    raw JSON
  </a>
</div>

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
    <input placeholder="filter…" bind:value={filter} />
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
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.6rem;
  }
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
