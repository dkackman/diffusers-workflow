<script lang="ts">
  import type { Snippet } from 'svelte'
  import { Popover } from 'bits-ui'
  import { holdLayer } from './layers.svelte'

  let {
    open = $bindable(false),
    label,
    anchor,
    align = 'start',
    children,
  }: {
    open?: boolean
    label: string
    /** The trigger: the popover sits against it, and Bits leaves a press
     * on it to the trigger's own toggle rather than treating it as outside */
    anchor: HTMLElement | null
    align?: 'start' | 'center' | 'end'
    children: Snippet
  } = $props()

  $effect(() => {
    if (open) return holdLayer()
  })
</script>

<!-- Non-modal: the page stays usable, so there is no aria-modal and no
     focus trap; focus moves in on open, and Escape or a press outside
     closes it. A popover holding controls is a non-modal dialog, a role
     Bits leaves to the caller -->
<Popover.Root bind:open>
  <Popover.Portal>
    <Popover.Content
      class="ui-popover panel"
      role="dialog"
      aria-label={label}
      customAnchor={anchor}
      side="bottom"
      {align}
      sideOffset={4}
      trapFocus={false}
    >
      {@render children()}
    </Popover.Content>
  </Popover.Portal>
</Popover.Root>
