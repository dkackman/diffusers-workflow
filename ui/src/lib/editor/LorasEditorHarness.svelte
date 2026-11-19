<script lang="ts">
  // Test harness: LorasEditor binds `pipeline`, so it needs reactive state
  import LorasEditor from './LorasEditor.svelte'

  let {
    initial,
    onchange,
  }: {
    initial: Record<string, any>
    onchange?: (pipeline: Record<string, any>) => void
  } = $props()

  // svelte-ignore state_referenced_locally
  let pipeline = $state<Record<string, any>>(initial)

  $effect(() => {
    onchange?.($state.snapshot(pipeline))
  })
</script>

<LorasEditor bind:pipeline />
