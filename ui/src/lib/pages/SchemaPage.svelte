<script lang="ts">
  import { api } from '../api'
  import SchemaNode, { type SchemaValue } from '../schema/SchemaNode.svelte'

  let schema = $state<SchemaValue | null>(null)
  let error = $state('')
  let filter = $state('')

  $effect(() => {
    api
      .getSchema()
      .then((s) => (schema = s))
      .catch((e) => (error = e instanceof Error ? e.message : String(e)))
  })

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

{#if error}<p class="error">{error}</p>{/if}

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
{:else if !error}
  <p class="muted">loading schema…</p>
{/if}

<style>
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.6rem;
  }
  .flex {
    flex: 1;
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
