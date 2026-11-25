<script module lang="ts">
  import { compareNames } from '../names'
  import { codec, hashState } from '../router.svelte'

  /** How a library list is ordered: by name (its folders, as before), or
   * the most recently changed first - the Workflows, Prompts and Assets
   * pages' one choice, the Gallery's Name/Newest in its words */
  export const LIST_SORTS = ['name', 'newest'] as const
  export type ListSortValue = (typeof LIST_SORTS)[number]

  /** The page's sort, in the hash so Back and a shared link keep it. Made
   * once, while the page sets up (a hashState watches the hash) */
  export const listSort = () =>
    hashState('sort', LIST_SORTS[0], codec.oneOf(LIST_SORTS))

  /** `names` with the most recently changed first; a name the listing gives
   * no time goes last, and a tie is by name */
  export function newestFirst(
    names: readonly string[],
    at: (name: string) => number | undefined,
  ): string[] {
    const stamp = (name: string) => at(name) ?? -Infinity
    return [...names].sort(
      (left, right) => stamp(right) - stamp(left) || compareNames(left, right),
    )
  }
</script>

<script lang="ts">
  import Segmented from './Segmented.svelte'

  let { sort }: { sort: { value: ListSortValue } } = $props()
</script>

<Segmented
  label="Sort"
  size="sm"
  options={[
    { value: 'name', label: 'Name', title: 'by name, in folders' },
    { value: 'newest', label: 'Newest', title: 'most recently changed first' },
  ]}
  value={sort.value}
  onselect={(v) => (sort.value = v)}
/>
