<script lang="ts" generics="A extends unknown[], R">
  import type { Snippet } from 'svelte'
  import type { Action } from '../action.svelte'

  // A button that runs an action(): disabled and aria-busy while it runs
  let {
    act,
    args = [] as unknown as A,
    children,
    danger = false,
    disabled = false,
    title,
    class: cls = 'quiet withicon',
  }: {
    act: Action<A, R>
    args?: A
    children: Snippet
    danger?: boolean
    disabled?: boolean
    title?: string
    class?: string
  } = $props()
</script>

<button
  type="button"
  class={[cls, danger && 'danger']}
  {title}
  disabled={disabled || act.busy}
  aria-busy={act.busy}
  onclick={() => void act.run(...args)}
>
  {@render children()}
</button>
