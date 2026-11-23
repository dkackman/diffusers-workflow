<script module lang="ts">
  /** The start-from value for a definition handed over from elsewhere (an
   * image's embedded metadata) rather than chosen here. */
  export const IMPORTED = '(imported)'
</script>

<script lang="ts">
  import { groupOf } from '../grouping'

  // A workflow not yet saved: which workspace it saves into and what it
  // starts from - blank, or one of the read-only examples. Two rows of the
  // file bar's grid.
  let {
    workspace,
    workspaces,
    startFrom,
    examples,
    onworkspace,
    onstartfrom,
  }: {
    workspace: string
    workspaces: string[]
    startFrom: string
    examples: string[]
    onworkspace: (workspace: string) => void
    onstartfrom: (example: string) => void
  } = $props()

  // The examples by folder, for the start-from picker's option groups
  const exampleGroups = $derived.by(() => {
    const groups: Record<string, string[]> = {}
    for (const example of examples)
      (groups[groupOf(example)] ??= []).push(example)
    return Object.entries(groups)
  })
</script>

<label for="wf-workspace">workspace</label>
<div>
  <select
    id="wf-workspace"
    class="pick"
    value={workspace}
    onchange={(e) => {
      const picked = e.currentTarget.value
      // the route change re-renders it; until then show the
      // workspace the draft is still in
      e.currentTarget.value = workspace
      onworkspace(picked)
    }}
    title="the workspace this workflow saves into"
  >
    {#each workspaces as name (name)}<option value={name}>{name}</option>{/each}
  </select>
</div>

<label for="wf-startfrom">start from</label>
<div>
  <select
    id="wf-startfrom"
    class="pick"
    value={startFrom}
    onchange={(e) => {
      const picked = e.currentTarget.value
      // held at the current start until the swap lands - a
      // cancelled confirm leaves it where it was
      e.currentTarget.value = startFrom
      onstartfrom(picked)
    }}
    title="a blank workflow, or a copy of one of the examples"
  >
    <option value="">blank</option>
    {#if startFrom && !examples.includes(startFrom)}
      <!-- handed over from elsewhere: an image's metadata, or a copy
           of one of this workspace's own workflows -->
      <option value={startFrom} disabled>
        {startFrom === IMPORTED ? 'image metadata' : `copy of ${startFrom}`}
      </option>
    {/if}
    {#each exampleGroups as [group, names] (group)}
      <optgroup label={group ? `${group}/` : 'examples'}>
        {#each names as example (example)}<option value={example}>
            {example}
          </option>{/each}
      </optgroup>
    {/each}
  </select>
</div>

<style>
  .pick {
    max-width: 320px;
  }
</style>
