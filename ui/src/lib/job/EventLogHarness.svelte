<script lang="ts">
  // Test harness: holds EventLog's props as state, so a test can change one
  // the way the page does, without the others being re-read as changed
  import EventLog from './EventLog.svelte'
  let { start }: { start: { live: boolean; pulse: number } } = $props()
  // svelte-ignore state_referenced_locally
  let current = $state({ jobId: 'j1', ...start })
  export function set(next: Partial<typeof current>) {
    Object.assign(current, next)
  }
</script>

<!-- Keyed by job, as JobPage keys it -->
{#key current.jobId}
  <EventLog {...current} />
{/key}
