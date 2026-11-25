<script lang="ts">
  import type { Snippet } from 'svelte'
  import { Popover } from 'bits-ui'
  import { holdLayer, portalTarget } from './layers.svelte'

  let {
    open = $bindable(false),
    label,
    anchor,
    align = 'start',
    menu = false,
    children,
  }: {
    open?: boolean
    label: string
    /** The trigger: the popover sits against it, and Bits leaves a press
     * on it to the trigger's own toggle rather than treating it as outside */
    anchor: HTMLElement | null
    align?: 'start' | 'center' | 'end'
    /** A short list of actions: as wide as its words, not a panel's width */
    menu?: boolean
    children: Snippet
  } = $props()

  $effect(() => {
    if (open) return holdLayer()
  })
  const portal = portalTarget(() => open)
</script>

<!-- Non-modal: the page stays usable, so there is no aria-modal and no
     focus trap; focus moves in on open, and Escape or a press outside
     closes it. A popover holding controls is a non-modal dialog, a role
     Bits leaves to the caller -->
<Popover.Root bind:open>
  <Popover.Portal to={portal.to}>
    <Popover.Content
      class={menu ? 'ui-popover ui-popover-menu panel' : 'ui-popover panel'}
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
