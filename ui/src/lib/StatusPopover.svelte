<script lang="ts">
  import Popover from './ui/Popover.svelte'
  import ServerHealthList from './ServerHealthList.svelte'

  let {
    open = $bindable(false),
    anchor = null,
  }: {
    open?: boolean
    anchor?: HTMLElement | null
  } = $props()
</script>

<Popover bind:open label="server status" {anchor}>
  <ServerHealthList onnavigate={() => (open = false)} />
</Popover>
