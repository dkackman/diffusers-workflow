<script module lang="ts">
  import { codec, hashState } from '../router.svelte'
  import { GROUPINGS, SORTS } from './order'

  /** The Gallery's search text, sort and grouping, each in the hash so
   * opening a file and pressing Back - or sharing the link - keeps them.
   * Made once, while the page sets up (a hashState watches the hash) */
  export function findState() {
    return {
      text: hashState('q', ''),
      sort: hashState('sort', SORTS[0], codec.oneOf(SORTS)),
      groupBy: hashState('group', GROUPINGS[0], codec.oneOf(GROUPINGS)),
    }
  }
  export type FindState = ReturnType<typeof findState>
</script>

<script lang="ts">
  // The Gallery's header: its search box, and Sort and Group by after the
  // page's own filters (`children`)
  import type { Snippet } from 'svelte'
  import PageHeader from '../ui/PageHeader.svelte'
  import Segmented from '../ui/Segmented.svelte'
  import { GROUP_OPTIONS, SORT_OPTIONS } from './order'

  let {
    find,
    count,
    children,
  }: {
    find: FindState
    count: { n: number; of: number }
    children: Snippet
  } = $props()
</script>

<PageHeader title="Gallery" baseline count={{ ...count, noun: 'file' }}>
  {#snippet filter()}
    <input
      class="filter"
      placeholder="search names, prompts and notes…"
      bind:value={find.text.value}
    />
  {/snippet}
  {#snippet actions()}
    {@render children()}
    <Segmented
      label="Sort"
      size="sm"
      options={SORT_OPTIONS}
      value={find.sort.value}
      onselect={(v) => (find.sort.value = v)}
    />
    <Segmented
      label="Group by"
      size="sm"
      options={GROUP_OPTIONS}
      value={find.groupBy.value}
      onselect={(v) => (find.groupBy.value = v)}
    />
  {/snippet}
</PageHeader>
