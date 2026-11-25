<script lang="ts">
  // Test harness: PipelineOptions binds `pipeline`, so it needs reactive state
  import PipelineOptions from './PipelineOptions.svelte'

  let { initial }: { initial: Record<string, any> } = $props()

  // svelte-ignore state_referenced_locally
  let pipeline = $state<Record<string, any>>(initial)
</script>

<PipelineOptions
  bind:pipeline
  index={0}
  openSection="components"
  references={[]}
/>
